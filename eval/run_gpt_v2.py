"""批量抽帧并使用 GPT 评测 videos 根目录下的 v2 视频。"""

import argparse
import base64
import io
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import fetch_video


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEOS_ROOT = PROJECT_ROOT / "videos"
ANCHOR_VIDEOS_ROOT = VIDEOS_ROOT / "NSA_anchor"
API_KEY_FILE = PROJECT_ROOT / "api_key.txt"
SAVE_PATH = PROJECT_ROOT / "results" / "gpt-5.6-sol-v2.jsonl"
TEST_SAVE_PATH = PROJECT_ROOT / "results" / "gpt-5.6-sol-v2-test.jsonl"
DEFAULT_RETRY_ERRORS_FILE = (
    PROJECT_ROOT / "results" / "gpt-5.6-sol-v2_label.jsonl"
)
FRAMES_ROOT = PROJECT_ROOT / "gpt_frames"

TRANS_PROMPT = (
    "不要借助现写的代码工具，你直接看这个视频，告诉我视频里的图案是怎么移动的？"
    "请在回答时按以下格式输出："
    "1. 先进行自由分析和推理。"
    "2. 推理结束后，在最后一行用 XML 标签包裹你的最终结论，格式如下："
    "<final_answer>你的最终结论写在这里</final_answer> "
    "注意： final_answer 标签内必须且只能包含最终结论，不要有任何额外解释。"
)
ROT_PROMPT = (
    "不要借助现写的代码工具，你直接看这个视频，告诉我视频里的球体是怎么旋转的？"
    "请在回答时按以下格式输出："
    "1. 先进行自由分析和推理。"
    "2. 推理结束后，在最后一行用 XML 标签包裹你的最终结论，格式如下："
    "<final_answer>你的最终结论写在这里</final_answer> "
    "注意： final_answer 标签内必须且只能包含最终结论，不要有任何额外解释。"
)

MODEL = "gpt-5.6-sol"
NUM_FRAMES = 10

# 文件名中的方向后缀与 ground truth 的硬编码对应关系。
GROUND_TRUTH = {
    "rot": {
        "up": "球体向上旋转",
        "down": "球体向下旋转",
        "left": "球体向左旋转",
        "right": "球体向右旋转",
        "clockwise": "球体顺时针旋转",
        "counterclockwise": "球体逆时针旋转",
    },
    "trans": {
        "up": "图案向上平移",
        "down": "图案向下平移",
        "left": "图案向左平移",
        "right": "图案向右平移",
        "clockwise": "图案顺时针旋转",
        "counterclockwise": "图案逆时针旋转",
    },
}

TASK_ORDER = {"rot": 0, "trans": 1}
CATEGORY_ORDER = {"NSA": 0, "NSA_ANCHOR": 1, "SA": 2}
DIRECTION_ORDER = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "clockwise": 4,
    "counterclockwise": 5,
}
VIDEO_NAME_PATTERN = re.compile(
    r"^(?P<task>rot|trans)_(?P<category>NSA_ANCHOR|NSA|SA)_"
    r"(?P<source>.+)_(?P<direction>up|down|left|right|clockwise|counterclockwise)\.mp4$",
    re.IGNORECASE,
)
FINAL_ANSWER_PATTERN = re.compile(
    r"<final_answer>\s*(.*?)\s*</final_answer>",
    re.IGNORECASE | re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量评测 videos 根目录下的 rot/trans v2 视频"
    )
    parser.add_argument(
        "--videos-root",
        default=str(VIDEOS_ROOT),
        help=(
            f"只扫描该目录直接下级的 MP4（默认：{VIDEOS_ROOT}；"
            f"锚点实验：{ANCHOR_VIDEOS_ROOT}）"
        ),
    )
    parser.add_argument(
        "--api-key-file",
        default=str(API_KEY_FILE),
        help=f"保存 API key 的 txt 文件（默认：{API_KEY_FILE}）",
    )
    parser.add_argument(
        "--frames-root",
        default=str(FRAMES_ROOT),
        help=f"抽帧输出根目录（默认：{FRAMES_ROOT}）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            f"结果 JSONL 路径（正式运行默认：{SAVE_PATH}；"
            f"试运行默认：{TEST_SAVE_PATH}）"
        ),
    )
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--test",
        action="store_true",
        help="试运行：只推理排序后的第一组六个变体",
    )
    test_group.add_argument(
        "--test-group",
        metavar="GROUP",
        help=(
            "试运行指定组六个变体，例如 rot_NSA_000、trans_SA_006 或 "
            "rot_NSA_ANCHOR_000_anchor_rot"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出视频、结果路径和断点状态，不读取 API key、不调用模型",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help=(
            "只复测本模型旧 label JSONL 中标为 incorrect 的 NSA 题目；"
            f"默认读取 {DEFAULT_RETRY_ERRORS_FILE}"
        ),
    )
    parser.add_argument(
        "--retry-errors-from",
        type=Path,
        default=DEFAULT_RETRY_ERRORS_FILE,
        help="--retry-errors 使用的旧标注 JSONL",
    )
    parser.add_argument(
        "--confirm-api-call",
        action="store_true",
        help="显式确认允许调用模型 API；--retry-errors 非 dry-run 时必须提供",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="忽略 JSONL 中已有结果并重新推理所选视频",
    )
    return parser.parse_args()


def natural_key(value: str) -> tuple:
    """让 2 排在 10 前面，同时兼容非纯数字的源图片名。"""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def parse_video_path(video_path: Path) -> dict | None:
    """解析 v2 视频文件名并返回排序和 GT 所需的元数据。"""
    match = VIDEO_NAME_PATTERN.fullmatch(video_path.name)
    if match is None:
        return None

    task = match.group("task").lower()
    category = match.group("category").upper()
    source = match.group("source")
    direction = match.group("direction").lower()
    return {
        "path": video_path,
        "task": task,
        "category": category,
        "source": source,
        "direction": direction,
        "group": f"{task}_{category}_{source}",
        "ground_truth": GROUND_TRUTH[task][direction],
    }


def video_sort_key(video: dict) -> tuple:
    """确保同一原图、同一任务的六个方向连续排列。"""
    return (
        TASK_ORDER[video["task"]],
        CATEGORY_ORDER[video["category"]],
        natural_key(video["source"]),
        DIRECTION_ORDER[video["direction"]],
    )


def discover_videos(videos_root: Path = VIDEOS_ROOT) -> list[dict]:
    """只扫描 videos 的直接子文件，不递归扫描 v1 等归档目录。"""
    if not videos_root.is_dir():
        raise FileNotFoundError(f"找不到视频目录：{videos_root}")

    videos = []
    ignored = []
    for path in videos_root.iterdir():
        if not path.is_file() or path.suffix.lower() != ".mp4":
            continue
        metadata = parse_video_path(path)
        if metadata is None:
            ignored.append(path.name)
        else:
            videos.append(metadata)

    if ignored:
        print(f"忽略 {len(ignored)} 个不符合 v2 命名规则的一级 MP4：")
        for name in sorted(ignored):
            print(f"  - {name}")

    videos.sort(key=video_sort_key)
    return videos


def default_output_path(
    videos_root: Path, test_mode: bool, retry_errors: bool = False
) -> Path:
    """根目录使用旧文件名；自定义视频目录自动使用独立结果文件。"""
    if videos_root.resolve() == VIDEOS_ROOT.resolve():
        return TEST_SAVE_PATH if test_mode else SAVE_PATH
    slug = re.sub(r"[^a-z0-9]+", "-", videos_root.name.lower()).strip("-") or "custom"
    if retry_errors:
        return PROJECT_ROOT / "results" / f"{MODEL}-{slug}-error-retry-v2.jsonl"
    suffix = "-test" if test_mode else ""
    return PROJECT_ROOT / "results" / f"{MODEL}-{slug}-v2{suffix}.jsonl"


def original_anchor_source(video: dict) -> str:
    """从 000_anchor_rot/trans 等派生 source 中取得原始 NSA 编号。"""
    match = re.match(r"^(?P<source>.+?)_anchor_(?:rot|trans)(?:_|$)", video["source"])
    return match.group("source") if match else video["source"]


def select_previous_errors(
    videos: list[dict], label_path: Path
) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """按 task、原始 source、direction 精确选择本模型的旧 NSA 错题。"""
    if not label_path.is_file():
        raise FileNotFoundError(f"找不到旧标注 JSONL：{label_path}")
    errors = {}
    with label_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label_path}:{line_number} 不是有效 JSON") from exc
            if (
                record.get("category") != "NSA"
                or record.get("human_answer_status") != "incorrect"
            ):
                continue
            key = (
                str(record.get("task", "")),
                str(record.get("source", "")),
                str(record.get("direction", "")),
            )
            errors[key] = record

    selected = []
    found = set()
    for video in videos:
        if video.get("category") != "NSA_ANCHOR":
            continue
        key = (video["task"], original_anchor_source(video), video["direction"])
        previous = errors.get(key)
        if previous is None:
            continue
        copied = dict(video)
        copied["previous_error"] = previous
        selected.append(copied)
        found.add(key)
    missing = sorted(set(errors) - found)
    return selected, missing


def previous_error_fields(video: dict) -> dict:
    """把复测对应的旧答案保存到新结果，便于直接做前后对照。"""
    previous = video.get("previous_error")
    if not previous:
        return {}
    return {
        "retry_of_video_name": previous.get("video_name"),
        "retry_of_video_path": previous.get("video_path"),
        "previous_model_answer": previous.get("model_answer"),
        "previous_reasoning": previous.get("reasoning"),
        "previous_final_answer": previous.get("final_answer"),
        "previous_answer_format_valid": previous.get("answer_format_valid"),
        "previous_human_answer_status": previous.get("human_answer_status"),
    }


def select_test_group(videos: list[dict], requested_group: str | None) -> list[dict]:
    """选择指定组，未指定时选择排序后的第一个完整六方向组。"""
    grouped = defaultdict(list)
    for video in videos:
        grouped[video["group"]].append(video)

    if requested_group is not None:
        normalized = requested_group.removesuffix(".mp4")
        selected = grouped.get(normalized)
        if not selected:
            available = ", ".join(grouped) if grouped else "无"
            raise ValueError(
                f"找不到测试组 {requested_group!r}。可用组包括：{available}"
            )
    else:
        selected = next(
            (group for group in grouped.values() if len(group) == len(DIRECTION_ORDER)),
            None,
        )
        if selected is None:
            raise RuntimeError("没有找到包含六个方向变体的完整测试组")

    selected = sorted(selected, key=video_sort_key)
    found_directions = {video["direction"] for video in selected}
    missing = set(DIRECTION_ORDER) - found_directions
    if missing:
        raise RuntimeError(
            f"测试组 {selected[0]['group']} 不完整，缺少方向：{sorted(missing)}"
        )
    return selected


def read_api_key(key_file: str) -> str:
    """从指定 txt 文件读取 API key，文件中仅需放置 key 本身。"""
    path = Path(key_file).expanduser()
    try:
        api_key = path.read_text(encoding="utf-8-sig").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(f"API key 文件不存在：{path}") from exc
    except OSError as exc:
        raise RuntimeError(f"无法读取 API key 文件：{path}（{exc}）") from exc

    if not api_key:
        raise RuntimeError(f"API key 文件为空：{path}")
    return api_key


def frame_to_pil(frame) -> Image.Image:
    """将 fetch_video 返回的 CHW tensor 转换为 RGB PIL Image。"""
    array = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def frame_to_data_url(frame) -> str:
    """将一帧编码为无损 PNG data URL。"""
    buffer = io.BytesIO()
    frame_to_pil(frame).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def get_frame_timestamps(video_path: str | Path, num_frames: int) -> list[dict]:
    """取得所有原始帧 PTS，并按 fetch_video 的均匀索引规则选出时间戳。"""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("未找到 ffprobe，无法获取抽帧时间戳") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "未知错误"
        raise RuntimeError(f"ffprobe 读取视频失败：{detail}") from exc

    probe_result = json.loads(completed.stdout)
    all_timestamps = [
        float(frame["best_effort_timestamp_time"])
        for frame in probe_result.get("frames", [])
        if frame.get("best_effort_timestamp_time") is not None
    ]
    if not all_timestamps:
        raise RuntimeError(f"没有从视频中读取到有效帧时间戳：{video_path}")

    indices = np.linspace(0, len(all_timestamps) - 1, num_frames).round().astype(int)
    return [
        {
            "source_frame_index": int(index),
            "timestamp_seconds": all_timestamps[index],
        }
        for index in indices
    ]


def format_timestamp(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS.mmm。"""
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def get_label_font(frame_height: int):
    font_size = max(16, frame_height // 18)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def save_frames(
    frames,
    video_path: str | Path,
    frame_timestamps: list[dict],
    save_root: str | Path = FRAMES_ROOT,
) -> Path:
    """保存 10 张无损单帧、带编号的 2x5 无损拼图及时间戳 JSONL。"""
    if len(frames) != 10:
        raise RuntimeError(f"2x5 拼图需要恰好 10 帧，实际抽取到 {len(frames)} 帧")
    if len(frame_timestamps) != len(frames):
        raise RuntimeError("抽取帧数量与时间戳数量不一致")

    video_name = Path(video_path).stem
    save_dir = Path(save_root) / video_name
    save_dir.mkdir(parents=True, exist_ok=True)

    pil_frames = [frame_to_pil(frame) for frame in frames]
    timestamp_records = []
    for frame_number, (image, timestamp) in enumerate(
        zip(pil_frames, frame_timestamps), start=1
    ):
        filename = f"frame_{frame_number:02d}.png"
        image.save(save_dir / filename, format="PNG")
        timestamp_records.append(
            {
                "frame_number": frame_number,
                "filename": filename,
                "source_frame_index": timestamp["source_frame_index"],
                "timestamp_seconds": round(timestamp["timestamp_seconds"], 6),
                "timestamp": format_timestamp(timestamp["timestamp_seconds"]),
            }
        )

    width, height = pil_frames[0].size
    cols, rows = 5, 2
    gap = max(6, min(width, height) // 80)
    border_width = max(2, gap // 3)
    grid_width = width * cols + gap * (cols - 1)
    grid_height = height * rows + gap * (rows - 1)
    grid = Image.new("RGB", (grid_width, grid_height), (235, 235, 235))
    font = get_label_font(height)
    draw = ImageDraw.Draw(grid)
    for index, image in enumerate(pil_frames):
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        row, column = divmod(index, cols)
        x = column * (width + gap)
        y = row * (height + gap)
        grid.paste(image, (x, y))
        draw.rectangle(
            (x, y, x + width - 1, y + height - 1),
            outline=(30, 30, 30),
            width=border_width,
        )

        label = f"Frame {index + 1:02d}"
        box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
        label_width = box[2] - box[0]
        label_height = box[3] - box[1]
        padding = max(4, height // 100)
        draw.rectangle(
            (x, y, x + label_width + padding * 2, y + label_height + padding * 2),
            fill=(0, 0, 0),
        )
        draw.text(
            (x + padding, y + padding),
            label,
            fill=(255, 255, 255),
            font=font,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )

    grid.save(save_dir / "grid_2x5.png", format="PNG")
    with (save_dir / "timestamps.jsonl").open("w", encoding="utf-8") as file:
        for record in timestamp_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return save_dir


def build_video_content(frames, prompt: str) -> list[dict]:
    """使用当前任务对应的提示词和同一批已落盘 PNG 帧构造消息。"""
    content = [{"type": "text", "text": prompt}]
    for frame in frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": frame_to_data_url(frame)},
            }
        )
    return content


def project_relative_path(path: str | Path) -> str:
    """将路径转换为相对于项目根目录的 POSIX 路径。"""
    absolute_path = Path(path).expanduser().resolve()
    return Path(os.path.relpath(absolute_path, PROJECT_ROOT)).as_posix()


def parse_model_answer(model_answer: str | None) -> dict:
    """按 prompt 要求拆分推理和 final_answer，并检查回答格式。"""
    text = model_answer or ""
    matches = list(FINAL_ANSWER_PATTERN.finditer(text))

    # 必须恰好有一个非空标签，且闭合标签之后只能有空白，才视为格式正确。
    if len(matches) == 1:
        match = matches[0]
        reasoning = text[: match.start()].strip()
        final_answer = match.group(1).strip()
        trailing_text = text[match.end() :].strip()
        format_valid = bool(final_answer) and not trailing_text
    else:
        reasoning = text.strip()
        final_answer = None
        format_valid = False

    return {
        "reasoning": reasoning,
        "final_answer": final_answer,
        "answer_format_valid": format_valid,
    }


def result_key(record: dict) -> str | None:
    """取得结果对应的视频文件名，用于断点续跑和去重。"""
    video_name = record.get("video_name")
    if video_name:
        return Path(str(video_name)).name
    video_path = record.get("video_path")
    if video_path:
        return Path(str(video_path)).name
    return None


def read_existing_results(output_path: Path) -> dict[str, dict]:
    """读取已有 JSONL；重复记录保留最后一条有效 JSON 对象。"""
    if not output_path.is_file():
        return {}
    results = {}
    with output_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"警告：忽略已有结果中的无效 JSON（第 {line_number} 行）")
                continue
            if not isinstance(record, dict):
                continue
            key = result_key(record)
            if key:
                results[key] = record
    return results


def write_results_in_order(
    output_path: Path,
    videos: list[dict],
    results: dict[str, dict],
) -> None:
    """按视频排序重写 JSONL，避免续跑结果追加到文件末尾。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for video in videos:
            record = results.get(video["path"].name)
            if record is not None:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)


def evaluate_video(
    client: OpenAI,
    video: dict,
    frames_root: str | Path,
) -> dict:
    """抽帧、保存可视化文件并评测一个视频。"""
    video_path = video["path"]
    prompt = ROT_PROMPT if video["task"] == "rot" else TRANS_PROMPT
    frames = fetch_video(
        {
            "type": "video",
            "video": str(video_path),
            "nframes": NUM_FRAMES,
        }
    )
    frame_timestamps = get_frame_timestamps(video_path, len(frames))
    frames_dir = save_frames(frames, video_path, frame_timestamps, frames_root)

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": build_video_content(frames, prompt),
            }
        ],
    )
    model_answer = completion.choices[0].message.content or ""
    parsed_answer = parse_model_answer(model_answer)
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video_path": project_relative_path(video_path),
        "video_name": video_path.name,
        "group": video["group"],
        "task": video["task"],
        "category": video["category"],
        "source": video["source"],
        "direction": video["direction"],
        "ground_truth": video["ground_truth"],
        "question": prompt,
        "model_answer": model_answer,
        **parsed_answer,
        "num_frames": len(frames),
        "frames_dir": project_relative_path(frames_dir),
        "finish_reason": completion.choices[0].finish_reason,
        **previous_error_fields(video),
    }


def main() -> None:
    args = parse_args()
    videos_root = Path(args.videos_root).expanduser().resolve()
    all_videos = discover_videos(videos_root)
    if not all_videos:
        raise RuntimeError(f"{videos_root} 直接下级没有符合 v2 命名规则的 MP4")
    videos = all_videos

    if args.retry_errors:
        if args.test or args.test_group:
            raise ValueError("--retry-errors 不能和 --test/--test-group 同时使用")
        if not args.dry_run and not args.confirm_api_call:
            raise RuntimeError(
                "旧错误复测会调用模型 API；请先使用 --dry-run 核对，"
                "确认后显式加入 --confirm-api-call"
            )
        videos, missing = select_previous_errors(
            all_videos, args.retry_errors_from.expanduser().resolve()
        )
        if missing:
            details = "\n".join(f"  - {task}/{source}/{direction}" for task, source, direction in missing)
            raise RuntimeError(
                f"旧错题对应的新视频缺少 {len(missing)} 个；未调用 API：\n{details}"
            )
        if not videos:
            raise RuntimeError("旧标注文件中没有可复测的 NSA incorrect 题目")
        print(f"旧错误复测模式：精确选择 {len(videos)} 个新锚点视频")

    if args.test or args.test_group:
        videos = select_test_group(videos, args.test_group)
        print(f"试运行模式：仅处理 {videos[0]['group']} 的 {len(videos)} 个变体")

    group_count = len({video["group"] for video in videos})
    print(f"待评测：{len(videos)} 个视频，共 {group_count} 组")
    print(f"第一个视频：{videos[0]['path'].name}")
    print(f"最后一个视频：{videos[-1]['path'].name}")

    default_output = default_output_path(
        videos_root, bool(args.test or args.test_group), args.retry_errors
    )
    output_path = Path(args.output).expanduser() if args.output else default_output
    existing_results = read_existing_results(output_path)
    if not args.overwrite:
        pending_videos = [
            video for video in videos if video["path"].name not in existing_results
        ]
    else:
        pending_videos = videos

    skipped = len(videos) - len(pending_videos)
    print(f"视频目录：{videos_root}")
    print(f"结果文件：{output_path}")
    print(f"已有结果：{skipped} 个，待处理：{len(pending_videos)} 个")
    if args.dry_run:
        print("dry-run：未读取 API key，未调用模型。待处理视频：")
        for video in pending_videos:
            print(f"  - {video['path'].name}")
        return
    if not pending_videos:
        # 即使本次没有新任务，也整理一次已有文件的顺序并去除重复记录。
        write_results_in_order(output_path, all_videos, existing_results)
        print(f"所有视频均已有结果，已确认 JSONL 顺序：{output_path}")
        return

    api_key = read_api_key(args.api_key_file)
    failures = []
    with httpx.Client(verify=False) as httpx_client:
        client = OpenAI(
            http_client=httpx_client,
            api_key=api_key,
            base_url=os.environ.get(
                "OPENAI_API_BASE", "http://115.120.87.159:8082/v1"
            ),
        )
        for index, video in enumerate(pending_videos, start=1):
            print(
                f"\n[{index}/{len(videos)}] {video['path'].name} "
                f"GT={video['ground_truth']}"
            )
            try:
                result = evaluate_video(client, video, args.frames_root)
                existing_results[video["path"].name] = result
                write_results_in_order(output_path, all_videos, existing_results)
                print(json.dumps(result, indent=2, ensure_ascii=False))
            except Exception as exc:
                failures.append((video["path"], exc))
                print(f"评测失败：{exc}")

    print(f"\n评测完成：新成功 {len(pending_videos) - len(failures)}，跳过 {skipped}，失败 {len(failures)}")
    print(f"结果已保存到：{output_path}")
    if failures:
        print("失败列表：")
        for video_path, error in failures:
            print(f"  - {video_path.name}: {error}")


if __name__ == "__main__":
    main()
