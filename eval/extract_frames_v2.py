"""Reproduce run_gpt_v2.py frame extraction without calling a model API."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import fetch_video


NUM_FRAMES = 10


def get_frame_timestamps(video_path: Path, num_frames: int) -> list[dict]:
    """Select timestamps using the same uniform frame-index rule as run_gpt_v2.py."""
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
            command, check=True, capture_output=True, text=True
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


def frame_to_pil(frame) -> Image.Image:
    array = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def format_timestamp(seconds: float) -> str:
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
    frames, video_path: Path, frame_timestamps: list[dict], frames_root: Path
) -> Path:
    if len(frames) != NUM_FRAMES:
        raise RuntimeError(
            f"2x5 拼图需要恰好 {NUM_FRAMES} 帧，实际抽取到 {len(frames)} 帧"
        )
    if len(frame_timestamps) != len(frames):
        raise RuntimeError("抽取帧数量与时间戳数量不一致")

    save_dir = frames_root / video_path.stem
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
    grid = Image.new(
        "RGB",
        (width * cols + gap * (cols - 1), height * rows + gap * (rows - 1)),
        (235, 235, 235),
    )
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
        padding = max(4, height // 100)
        draw.rectangle(
            (x, y, x + box[2] - box[0] + padding * 2, y + box[3] - box[1] + padding * 2),
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


def extract_video(video_path: Path, frames_root: Path) -> Path:
    frames = fetch_video(
        {"type": "video", "video": str(video_path), "nframes": NUM_FRAMES}
    )
    timestamps = get_frame_timestamps(video_path, len(frames))
    return save_frames(frames, video_path, timestamps, frames_root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="仅复现 run_gpt_v2.py 的抽帧和可视化输出，不调用模型 API"
    )
    parser.add_argument(
        "--videos-root", type=Path, default=Path("videos"), help="直接扫描一级 MP4 的目录"
    )
    parser.add_argument(
        "--frames-root", type=Path, default=Path("gpt_frames"), help="抽帧输出根目录"
    )
    parser.add_argument(
        "--video", type=Path, help="只处理一个 MP4；不提供时处理目录下所有一级 MP4"
    )
    args = parser.parse_args()
    videos_root = args.videos_root.expanduser()
    frames_root = args.frames_root.expanduser()
    if args.video:
        videos = [args.video.expanduser()]
    else:
        if not videos_root.is_dir():
            raise FileNotFoundError(f"找不到视频目录：{videos_root}")
        videos = sorted(
            path for path in videos_root.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"
        )
    if not videos:
        raise RuntimeError("没有找到 MP4 视频")

    for index, video_path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video_path}")
        output = extract_video(video_path, frames_root)
        print(f"  输出：{output}")


if __name__ == "__main__":
    main()
