"""批量使用 Gemini 直接评测 videos/v3 下的三类视频。"""

import argparse
import base64
import json
import os
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEOS_ROOT = PROJECT_ROOT / "videos" / "v3"
API_URL = (
    "http://115.120.87.159:8082/v1beta/models/"
    "gemini-3.5-flash:generateContent"
)
API_KEY_FILE = PROJECT_ROOT / "api_key.txt"
SAVE_PATH = PROJECT_ROOT / "results" / "gemini-3.5-flash-v3.jsonl"
TEST_SAVE_PATH = PROJECT_ROOT / "results" / "gemini-3.5-flash-v3-test.jsonl"
DEFAULT_MAX_UPLOAD_MIB = 80.0
DEFAULT_TRANSCODE_MAX_DIMENSION = 1600
DEFAULT_TRANSCODE_CRF = 23

TRANS_PROMPT = (
    "Watch the video directly without using code or tools. Describe how the pattern moves. "
    "Write your analysis and reasoning freely in English. Then state your final conclusion "
    "in your own words on the final line using this format: "
    "<final_answer>Your conclusion</final_answer>. The final_answer tag must contain only "
    "your concise conclusion, with no additional explanation after the tag."
)
ROT_PROMPT = (
    "Watch the video directly without using code or tools. Describe how the sphere rotates. "
    "Write your analysis and reasoning freely in English. Then state your final conclusion "
    "in your own words on the final line using this format: "
    "<final_answer>Your conclusion</final_answer>. The final_answer tag must contain only "
    "your concise conclusion, with no additional explanation after the tag."
)

# 文件名中的方向后缀与 ground truth 的硬编码对应关系。
GROUND_TRUTH = {
    "rot": {
        "up": "The sphere rotates upward.",
        "down": "The sphere rotates downward.",
        "left": "The sphere rotates leftward.",
        "right": "The sphere rotates rightward.",
        "clockwise": "The sphere rotates clockwise.",
        "counterclockwise": "The sphere rotates counterclockwise.",
    },
    "trans": {
        "up": "The pattern translates upward.",
        "down": "The pattern translates downward.",
        "left": "The pattern translates leftward.",
        "right": "The pattern translates rightward.",
        "clockwise": "The pattern rotates clockwise.",
        "counterclockwise": "The pattern rotates counterclockwise.",
    },
}

TASK_ORDER = {"rot": 0, "trans": 1}
CATEGORY_ORDER = {"NSA": 0, "SA": 1, "NSA_ANCHOR": 2}
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
        description="批量使用 Gemini 评测 videos/v3 下三类 rot/trans 视频"
    )
    parser.add_argument(
        "--videos-root",
        default=str(VIDEOS_ROOT),
        help=f"只扫描该目录直接下级的 MP4（默认：{VIDEOS_ROOT}）",
    )
    parser.add_argument(
        "--api-key-file",
        default=str(API_KEY_FILE),
        help=f"保存 API key 的 txt 文件（默认：{API_KEY_FILE}）",
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
        help="试运行：从 NSA、SA、NSA_ANCHOR 各选择排序后的第一个视频",
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
        "--max-upload-mib",
        type=float,
        default=DEFAULT_MAX_UPLOAD_MIB,
        help=(
            "Gemini inline_data 原始 MP4 大小阈值；超过时临时转码，"
            f"默认 {DEFAULT_MAX_UPLOAD_MIB:g} MiB"
        ),
    )
    parser.add_argument(
        "--transcode-max-dimension",
        type=int,
        default=DEFAULT_TRANSCODE_MAX_DIMENSION,
        help=(
            "临时转码视频的最长边上限，"
            f"默认 {DEFAULT_TRANSCODE_MAX_DIMENSION} 像素"
        ),
    )
    parser.add_argument(
        "--transcode-crf",
        type=int,
        default=DEFAULT_TRANSCODE_CRF,
        help=f"临时 H.264 转码的 CRF，默认 {DEFAULT_TRANSCODE_CRF}",
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
    """解析 v3 视频文件名并返回排序和 GT 所需的元数据。"""
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
        print(f"忽略 {len(ignored)} 个不符合 v3 命名规则的一级 MP4：")
        for name in sorted(ignored):
            print(f"  - {name}")

    videos.sort(key=video_sort_key)
    return videos


def default_output_path(videos_root: Path, test_mode: bool) -> Path:
    """根目录使用旧文件名；自定义视频目录自动使用独立结果文件。"""
    if videos_root.resolve() == VIDEOS_ROOT.resolve():
        return TEST_SAVE_PATH if test_mode else SAVE_PATH
    slug = re.sub(r"[^a-z0-9]+", "-", videos_root.name.lower()).strip("-") or "custom"
    suffix = "-test" if test_mode else ""
    return PROJECT_ROOT / "results" / f"gemini-3.5-flash-{slug}-v3{suffix}.jsonl"


def select_test_samples(videos: list[dict]) -> list[dict]:
    """从三个类别中各选择排序后的第一个视频。"""
    selected = []
    for category in CATEGORY_ORDER:
        video = next(
            (item for item in videos if item["category"] == category),
            None,
        )
        if video is None:
            raise RuntimeError(f"测试模式找不到 {category} 类别的视频")
        selected.append(video)
    return selected


def select_test_group(videos: list[dict], requested_group: str) -> list[dict]:
    """选择指定的完整六方向组。"""
    grouped = defaultdict(list)
    for video in videos:
        grouped[video["group"]].append(video)

    normalized = requested_group.removesuffix(".mp4")
    selected = grouped.get(normalized)
    if not selected:
        available = ", ".join(grouped) if grouped else "无"
        raise ValueError(
            f"找不到测试组 {requested_group!r}。可用组包括：{available}"
        )

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


def project_relative_path(path: str | Path) -> str:
    """将路径转换为相对于项目根目录的 POSIX 路径。"""
    absolute_path = Path(path).expanduser().resolve()
    return Path(os.path.relpath(absolute_path, PROJECT_ROOT)).as_posix()


def parse_model_answer(model_answer: str | None) -> dict:
    """按 prompt 要求拆分推理和 final_answer，并检查回答格式。"""
    text = model_answer or ""
    matches = list(FINAL_ANSWER_PATTERN.finditer(text))
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


def extract_gemini_text(response_data: dict) -> str:
    """从 Gemini generateContent 响应中提取文本，并给出可读错误。"""
    try:
        parts = response_data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Gemini 响应中没有 candidates/content/parts：{response_data}"
        ) from exc
    text_parts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    text = "".join(text_parts).strip()
    if not text:
        raise RuntimeError(f"Gemini 响应中没有文本内容：{response_data}")
    return text


def format_mib(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.2f} MiB"


def transcode_for_upload(
    source_path: Path,
    output_path: Path,
    max_dimension: int,
    crf: int,
) -> None:
    """将超限视频临时缩小为 H.264；不修改原始视频。"""
    scale_filter = (
        f"scale='min({max_dimension},iw)':'min({max_dimension},ih)':"
        "force_original_aspect_ratio=decrease,"
        "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    )
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        scale_filter,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("未找到 ffmpeg，无法转码超限视频") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"退出码 {completed.returncode}"
        raise RuntimeError(f"临时转码失败：{detail}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("临时转码没有生成有效 MP4")


def evaluate_video(
    video: dict,
    api_key: str,
    max_upload_bytes: int,
    transcode_max_dimension: int,
    transcode_crf: int,
) -> dict:
    """读取整个 MP4 并通过 Gemini inline_data 接口评测一个视频。"""
    video_path = video["path"]
    prompt = ROT_PROMPT if video["task"] == "rot" else TRANS_PROMPT
    original_size = video_path.stat().st_size
    transcoded = original_size > max_upload_bytes
    temporary_directory = None
    upload_path = video_path
    try:
        if transcoded:
            temporary_directory = tempfile.TemporaryDirectory(
                prefix="gemini-v3-upload-"
            )
            upload_path = Path(temporary_directory.name) / video_path.name
            print(
                f"  文件超过 {format_mib(max_upload_bytes)}，临时转码："
                f"{format_mib(original_size)} -> 最长边 {transcode_max_dimension}px，"
                f"CRF {transcode_crf}"
            )
            transcode_for_upload(
                video_path,
                upload_path,
                transcode_max_dimension,
                transcode_crf,
            )
            upload_size = upload_path.stat().st_size
            print(f"  临时文件大小：{format_mib(upload_size)}")
            if upload_size > max_upload_bytes:
                raise RuntimeError(
                    "临时转码后仍超过上传阈值："
                    f"{format_mib(upload_size)} > {format_mib(max_upload_bytes)}；"
                    "请减小 --transcode-max-dimension 或增大 --transcode-crf"
                )
        else:
            upload_size = original_size

        estimated_base64_size = ((upload_size + 2) // 3) * 4
        if estimated_base64_size >= 128 * 1024 * 1024:
            raise RuntimeError(
                "预计 Base64 请求体已达到 128 MiB 上限："
                f"{format_mib(estimated_base64_size)}；"
                "请减小 --max-upload-mib 或调整转码参数"
            )

        try:
            video_bytes = upload_path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"读取视频失败：{upload_path}（{exc}）") from exc

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "video/mp4",
                                "data": base64.b64encode(video_bytes).decode("ascii"),
                            }
                        }
                    ]
                }
            ]
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        try:
            response = requests.post(API_URL, json=payload, headers=headers, timeout=None)
        except requests.RequestException as exc:
            raise RuntimeError(f"Gemini 请求失败：{exc}") from exc

        try:
            response_data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Gemini 返回了非 JSON 响应（HTTP {response.status_code}）："
                f"{response.text[:500]}"
            ) from exc
        if response.status_code != 200:
            raise RuntimeError(
                f"Gemini 请求失败（HTTP {response.status_code}）：{response_data}"
            )

        model_answer = extract_gemini_text(response_data)
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
            "finish_reason": response_data["candidates"][0].get("finishReason"),
            "upload_transcoded": transcoded,
            "original_video_size_bytes": original_size,
            "uploaded_video_size_bytes": upload_size,
            "transcode_max_dimension": (
                transcode_max_dimension if transcoded else None
            ),
            "transcode_crf": transcode_crf if transcoded else None,
        }
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()


def main() -> None:
    args = parse_args()
    if args.max_upload_mib <= 0:
        raise ValueError("--max-upload-mib 必须大于 0")
    if args.transcode_max_dimension < 2:
        raise ValueError("--transcode-max-dimension 必须至少为 2")
    if not 0 <= args.transcode_crf <= 51:
        raise ValueError("--transcode-crf 必须在 0 到 51 之间")
    max_upload_bytes = round(args.max_upload_mib * 1024 * 1024)

    videos_root = Path(args.videos_root).expanduser().resolve()
    all_videos = discover_videos(videos_root)
    if not all_videos:
        raise RuntimeError(f"{videos_root} 直接下级没有符合 v3 命名规则的 MP4")
    videos = all_videos

    if args.test:
        videos = select_test_samples(videos)
        categories = ", ".join(video["category"] for video in videos)
        print(f"试运行模式：从三个类别各选择 1 个视频（{categories}）")
    elif args.test_group:
        videos = select_test_group(videos, args.test_group)
        print(f"试运行模式：仅处理 {videos[0]['group']} 的 {len(videos)} 个变体")

    group_count = len({video["group"] for video in videos})
    print(f"待评测：{len(videos)} 个视频，共 {group_count} 组")
    print(f"第一个视频：{videos[0]['path'].name}")
    print(f"最后一个视频：{videos[-1]['path'].name}")

    default_output = default_output_path(videos_root, bool(args.test or args.test_group))
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
    oversized_pending = [
        video
        for video in pending_videos
        if video["path"].stat().st_size > max_upload_bytes
    ]
    print(
        f"上传阈值：{args.max_upload_mib:g} MiB；"
        f"待处理视频中需临时转码：{len(oversized_pending)} 个"
    )
    if args.dry_run:
        print("dry-run：未转码、未读取 API key、未调用模型。待处理视频：")
        for video in pending_videos:
            size = video["path"].stat().st_size
            action = "临时转码" if size > max_upload_bytes else "原文件上传"
            print(f"  - {video['path'].name}：{format_mib(size)}，{action}")
        return
    if not pending_videos:
        write_results_in_order(output_path, all_videos, existing_results)
        print(f"所有视频均已有结果，已确认 JSONL 顺序：{output_path}")
        return

    api_key = read_api_key(args.api_key_file)
    failures = []
    for index, video in enumerate(pending_videos, start=1):
        print(
            f"\n[{index}/{len(videos)}] {video['path'].name} "
            f"GT={video['ground_truth']}"
        )
        try:
            result = evaluate_video(
                video,
                api_key,
                max_upload_bytes,
                args.transcode_max_dimension,
                args.transcode_crf,
            )
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
