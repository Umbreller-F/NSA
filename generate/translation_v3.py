"""使用 textures/v3 的 NSA、SA 和 NSA_anchor 纹理生成六方向运动视频。"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXTURES_ROOT = PROJECT_ROOT / "textures" / "v3"
OUTPUT_ROOT = PROJECT_ROOT / "videos" / "v3"
CATEGORY_DIRS = {
    "NSA": TEXTURES_ROOT / "NSA",
    "SA": TEXTURES_ROOT / "SA",
    "NSA_ANCHOR": TEXTURES_ROOT / "NSA_anchor",
}
CATEGORIES = tuple(CATEGORY_DIRS)
DIRECTIONS = ("up", "down", "left", "right", "clockwise", "counterclockwise")

FPS = 30
DURATION_SECONDS = 5
VIEW_RATIO = 0.65
ROTATION_DEG = 180


class FFmpegWriter:
    """把 RGB numpy 帧通过标准输入流式写入 H.264 MP4。"""

    def __init__(self, output_path: Path, width: int, height: int, fps: int):
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("未找到 ffmpeg，请先安装 ffmpeg") from exc

    def append_data(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg 标准输入不可用")
        try:
            self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        except BrokenPipeError as exc:
            error = self.process.stderr.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg 编码失败：{error or '管道意外关闭'}") from exc

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        error = self.process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg 编码失败：{error or f'退出码 {return_code}'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "批量为 textures/v3/NSA、textures/v3/SA、textures/v3/NSA_anchor "
            "中的 JPG/PNG 生成六方向运动视频，并保存到 videos/v3"
        )
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=CATEGORIES,
        default=list(CATEGORIES),
        help="要生成的实验类别；默认 NSA SA NSA_ANCHOR",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的视频；默认跳过非空的已有视频",
    )
    return parser.parse_args()


def find_jpg_images(directory: Path) -> list[Path]:
    """返回目录下按文件名排序的 JPG/JPEG/PNG 图片。"""
    if not directory.is_dir():
        raise FileNotFoundError(f"找不到纹理目录：{directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def even_size(value: int) -> int:
    """返回不大于 value 的正偶数，满足 H.264 yuv420p 编码要求。"""
    return max(2, value - value % 2)


def smoothstep(progress: float) -> float:
    """缓入缓出，让运动的起步与停止更加自然。"""
    return progress * progress * (3.0 - 2.0 * progress)


def camera_path(
    direction: str,
    image_width: int,
    image_height: int,
    view_width: int,
    view_height: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """根据画面中图案的运动方向返回裁剪窗口的反向运动路径。"""
    max_x = image_width - view_width
    max_y = image_height - view_height
    center_x = max_x // 2
    center_y = max_y // 2
    paths = {
        # 裁剪窗口（镜头）与画面内容的视运动方向相反。
        "right": ((max_x, center_y), (0, center_y)),
        "left": ((0, center_y), (max_x, center_y)),
        "down": ((center_x, max_y), (center_x, 0)),
        "up": ((center_x, 0), (center_x, max_y)),
    }
    return paths[direction]


def center_crop(image: Image.Image, width: int, height: int) -> np.ndarray:
    """从 PIL 图片中心裁出固定尺寸并返回 RGB 数组。"""
    left = (image.width - width) // 2
    top = (image.height - height) // 2
    return np.asarray(image.crop((left, top, left + width, top + height)))


def generate_video(image_path: Path, output_path: Path, direction: str) -> None:
    """为单张图片生成图案沿指定方向运动或旋转的视频。"""
    if not 0 < VIEW_RATIO < 1:
        raise ValueError("VIEW_RATIO 必须在 0 和 1 之间")
    if direction not in DIRECTIONS:
        raise ValueError(f"不支持的方向：{direction}")

    with Image.open(image_path) as source:
        pil_image = source.convert("RGB")

    image = np.asarray(pil_image)
    image_height, image_width = image.shape[:2]
    view_width = even_size(int(image_width * VIEW_RATIO))
    view_height = even_size(int(image_height * VIEW_RATIO))
    frame_count = max(2, round(FPS * DURATION_SECONDS))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = FFmpegWriter(output_path, view_width, view_height, FPS)
    try:
        if direction in {"up", "down", "left", "right"}:
            start, end = camera_path(
                direction,
                image_width,
                image_height,
                view_width,
                view_height,
            )
            for frame_index in range(frame_count):
                progress = smoothstep(frame_index / (frame_count - 1))
                x = round(start[0] + (end[0] - start[0]) * progress)
                y = round(start[1] + (end[1] - start[1]) * progress)
                frame = image[y : y + view_height, x : x + view_width]
                writer.append_data(frame)
        else:
            # Pillow 中正角度为逆时针，负角度为顺时针。
            sign = 1 if direction == "counterclockwise" else -1
            for frame_index in range(frame_count):
                progress = smoothstep(frame_index / (frame_count - 1))
                angle = sign * ROTATION_DEG * progress
                rotated = pil_image.rotate(
                    angle,
                    resample=Image.Resampling.BICUBIC,
                    expand=False,
                    fillcolor=(0, 0, 0),
                )
                writer.append_data(center_crop(rotated, view_width, view_height))
    finally:
        writer.close()


def main() -> None:
    args = parse_args()
    jobs = []
    for category in args.categories:
        directory = CATEGORY_DIRS[category]
        if not directory.is_dir():
            print(f"警告：纹理目录不存在，跳过：{directory}")
            continue
        images = find_jpg_images(directory)
        if category == "NSA_ANCHOR":
            images = [path for path in images if "_anchor_trans" in path.stem]
        if not images:
            print(f"警告：{directory} 中没有适用于 translation 的 JPG/PNG 图片")
        for image_path in images:
            for direction in DIRECTIONS:
                output_path = (
                    OUTPUT_ROOT / f"trans_{category}_{image_path.stem}_{direction}.mp4"
                )
                jobs.append((image_path, output_path, direction))

    if not jobs:
        raise RuntimeError("没有找到任何待处理的 JPG 图片")

    print(f"共发现 {len(jobs)} 个运动视频任务；全部输出到 {OUTPUT_ROOT}")
    generated = 0
    skipped = 0
    failures = []
    for job_index, (image_path, output_path, direction) in enumerate(jobs, start=1):
        if output_path.is_file() and output_path.stat().st_size > 0 and not args.overwrite:
            skipped += 1
            print(f"[{job_index}/{len(jobs)}] 跳过已有：{output_path}")
            continue

        print(
            f"[{job_index}/{len(jobs)}] 生成 {image_path.parent.name}/{image_path.name} "
            f"-> {direction}"
        )
        try:
            generate_video(image_path, output_path, direction)
            generated += 1
            print(f"已保存：{output_path}")
        except Exception as exc:  # 记录失败后继续完成其余批次。
            output_path.unlink(missing_ok=True)
            failures.append((output_path, exc))
            print(f"失败：{output_path}：{exc}")

    print(f"完成：新生成 {generated} 个，跳过 {skipped} 个，失败 {len(failures)} 个")
    if failures:
        details = "\n".join(f"- {path}: {error}" for path, error in failures)
        raise RuntimeError(f"以下视频生成失败：\n{details}")


if __name__ == "__main__":
    main()
