"""预处理 ``textures/v3`` 下以 ``raw`` 结尾的纹理目录。

默认会将 ``textures/v3/*_raw`` 中的 JPG/PNG 图片中心裁剪为正方形，
并保存到去掉 ``_raw`` 后的同级目录。例如：

    textures/v3/NSA_raw/000.jpg -> textures/v3/NSA/000.jpg
    textures/v3/SA_raw/019.png  -> textures/v3/SA/019.jpg

PNG 会转换为 JPG；透明区域在转换前使用白色背景合成。

在项目根目录运行：

    python3 generate/process_v3_textures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEXTURES_ROOT = PROJECT_ROOT / "textures" / "v3"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="中心裁剪 textures/v3 下的原始纹理，并将 PNG 转换为 JPG"
    )
    parser.add_argument(
        "--textures-root",
        type=Path,
        default=DEFAULT_TEXTURES_ROOT,
        help="纹理根目录，默认是项目中的 textures/v3",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG 保存质量（1-100），默认 95",
    )
    return parser.parse_args()


def output_directory(source_directory: Path) -> Path:
    """返回原始目录对应的输出目录。"""
    name = source_directory.name
    if not name.lower().endswith("raw"):
        raise ValueError(f"目录名不是以 raw 结尾：{source_directory}")

    output_name = name[:-3].rstrip("_-")
    if not output_name:
        raise ValueError(f"无法从目录名推导输出目录：{source_directory}")
    return source_directory.with_name(output_name)


def center_crop_square(image: Image.Image) -> Image.Image:
    """按照短边从图片中心裁出正方形。"""
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def flatten_to_rgb(image: Image.Image) -> Image.Image:
    """转换为 RGB；如果包含透明通道，则用白色填充透明区域。"""
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def destination_path(source_path: Path, target_directory: Path) -> Path:
    if source_path.suffix.lower() == ".png":
        return target_directory / f"{source_path.stem}.jpg"
    return target_directory / source_path.name


def process_image(source_path: Path, destination: Path, quality: int) -> None:
    with Image.open(source_path) as opened_image:
        # 先应用相机/手机图片可能携带的 EXIF 方向，确保按视觉方向裁剪。
        image = ImageOps.exif_transpose(opened_image)
        square = center_crop_square(image)
        rgb = flatten_to_rgb(square)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(destination, format="JPEG", quality=quality, optimize=True)


def process_directory(source_directory: Path, quality: int) -> int:
    target_directory = output_directory(source_directory)
    source_images = sorted(
        path
        for path in source_directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )

    destinations: dict[Path, Path] = {}
    for source_path in source_images:
        destination = destination_path(source_path, target_directory)
        if destination in destinations:
            raise RuntimeError(
                "输出文件名冲突："
                f"{destinations[destination]} 和 {source_path} 都会写入 {destination}"
            )
        destinations[destination] = source_path

    for destination, source_path in destinations.items():
        process_image(source_path, destination, quality)
        print(f"{source_path} -> {destination}")
    return len(source_images)


def main() -> None:
    args = parse_args()
    if not 1 <= args.quality <= 100:
        raise ValueError("--quality 必须在 1 到 100 之间")
    if not args.textures_root.is_dir():
        raise FileNotFoundError(f"找不到纹理根目录：{args.textures_root}")

    raw_directories = sorted(
        path
        for path in args.textures_root.iterdir()
        if path.is_dir() and path.name.lower().endswith("raw")
    )
    if not raw_directories:
        raise RuntimeError(f"{args.textures_root} 下没有以 raw 结尾的目录")

    total = sum(process_directory(directory, args.quality) for directory in raw_directories)
    print(f"处理完成：{len(raw_directories)} 个目录，共 {total} 张图片")


if __name__ == "__main__":
    main()
