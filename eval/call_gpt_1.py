import argparse
import base64
import io
import json
import os
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import fetch_video


# 可在命令行中通过 --api-key-file 指定其他 txt 文件。
API_KEY_FILE = "./api_key.txt"

# VIDEO_PATH = "./videos/demo_008.mp4"
# VIDEO_PATH = "./videos/demo_007.mp4"
# VIDEO_PATH = "./videos/demo_005.mp4"
VIDEO_PATH = "./videos/translation_006.mp4"
# PROMPT = (
#     "不要借助现写的代码工具，你直接看这个视频，告诉我视频里的球体是不是静止的？"
#     "如果不是，具体是怎么样运动的，往哪个方向？"
# )
PROMPT = (
    "不要借助现写的代码工具，你直接看这个视频，告诉我视频里的画面是不是静止的？"
    "如果不是，具体是怎么样运动的，往哪个方向？"
)
MODEL = "gpt-5.6-sol"
NUM_FRAMES = 10
SAVE_PATH = "./results/gpt-5.6-sol.jsonl"
FRAMES_ROOT = "./gpt_frames"


def parse_args():
    parser = argparse.ArgumentParser(description="均匀抽帧并调用 GPT 分析视频")
    parser.add_argument(
        "--api-key-file",
        default=API_KEY_FILE,
        help=f"保存 API key 的 txt 文件（默认：{API_KEY_FILE}）",
    )
    parser.add_argument("--video-path", default=VIDEO_PATH, help="待分析的视频路径")
    parser.add_argument(
        "--frames-root",
        default=FRAMES_ROOT,
        help=f"抽帧输出根目录（默认：{FRAMES_ROOT}）",
    )
    return parser.parse_args()


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


def get_frame_timestamps(video_path: str, num_frames: int) -> list[dict]:
    """取得所有原始帧的 PTS，并按 fetch_video 的均匀索引规则选出时间戳。"""
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
        video_path,
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
    all_timestamps = []
    for frame in probe_result.get("frames", []):
        value = frame.get("best_effort_timestamp_time")
        if value is not None:
            all_timestamps.append(float(value))

    if not all_timestamps:
        raise RuntimeError(f"没有从视频中读取到有效帧时间戳：{video_path}")

    # qwen_vl_utils 对显式 nframes 使用从首帧到末帧的均匀采样。
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
    video_path: str,
    frame_timestamps: list[dict],
    save_root: str = FRAMES_ROOT,
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
    # 使用浅灰色间隔和深色边框分隔相邻帧，避免相似画面连在一起。
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

    timestamp_path = save_dir / "timestamps.jsonl"
    with timestamp_path.open("w", encoding="utf-8") as file:
        for record in timestamp_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return save_dir


def build_video_content(frames):
    """使用同一批已落盘帧构造 OpenAI 消息，避免重复抽帧产生偏差。"""
    content = [{"type": "text", "text": PROMPT}]
    for frame in frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": frame_to_data_url(frame)},
            }
        )
    return content


def main():
    args = parse_args()
    api_key = read_api_key(args.api_key_file)

    frames = fetch_video(
        {
            "type": "video",
            "video": args.video_path,
            "nframes": NUM_FRAMES,
        }
    )
    frame_timestamps = get_frame_timestamps(args.video_path, len(frames))
    frames_dir = save_frames(
        frames,
        args.video_path,
        frame_timestamps,
        save_root=args.frames_root,
    )
    print(f"抽帧文件已保存到：{frames_dir}")

    httpx_client = httpx.Client(verify=False)
    client = OpenAI(
        http_client=httpx_client,
        api_key=api_key,
        base_url=os.environ.get("OPENAI_API_BASE", "http://115.120.87.159:8082/v1"),
    )

    messages = [
        {
            "role": "user",
            "content": build_video_content(frames),
        }
    ]

    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    text = completion.choices[0].message.content
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video_path": args.video_path,
        "question": PROMPT,
        "model_answer": text,
        "num_frames": len(frames),
        "frames_dir": str(frames_dir),
        "frame_timestamps": frame_timestamps,
        "finish_reason": completion.choices[0].finish_reason,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    save_path = Path(SAVE_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\nGPT 结果已保存到 {save_path}")


if __name__ == "__main__":
    main()
