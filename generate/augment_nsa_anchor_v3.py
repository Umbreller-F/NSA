"""从 NSA 纹理中构造低语义、高独特性的局部纹理锚点。

脚本会处理全部 NSA 源图，并从另一张 NSA 纹理中裁取一个小 patch，经过
可选的局部颜色匹配后，以软边界融合到目标纹理中。原图不会被覆盖；默认
读取 ``textures/v3/NSA``，输出到
``textures/v3/NSA_anchor``，并写出
``manifest.jsonl`` 记录每个改造样本的来源和参数。

因为当前两个视频生成器的可见区域不同，脚本默认会为每张源图生成两种布局：

* ``*_anchor_trans_*``：锚点随机放在中心可视区域，保证在平移视频中出现；
* ``*_anchor_rot_*``：锚点随机放在等距柱状纹理左右接缝附近的安全区域，
  映射后位于球体正面。

示例（在 non-semantic-anchor 目录执行）：

    python generate/augment_nsa_anchor_v3.py
    python generate/augment_nsa_anchor_v3.py --no-color-match --seed 42

输出目录不是当前两个批量视频脚本的默认输入目录。这样可以避免误把改造
纹理混入原始 NSA 集合；后续生成视频时可以显式指定/复制该目录，或扩展
生成脚本的输入类别。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXTURES_ROOT = PROJECT_ROOT / "textures" / "v3"
DEFAULT_NSA_DIR = TEXTURES_ROOT / "NSA"
DEFAULT_OUTPUT_DIR = TEXTURES_ROOT / "NSA_anchor"
DEFAULT_VISIBLE_RATIO = 0.65
DEFAULT_ROT_SEAM_RATIO = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为全部 NSA 源图构造局部纹理锚点")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_NSA_DIR,
        help="NSA 原始纹理目录，默认 textures/v3/NSA",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="改造纹理输出目录，默认 textures/v3/NSA_anchor",
    )
    parser.add_argument(
        "--patch-ratio",
        type=float,
        default=0.12,
        help="patch 边长占目标图最短边的比例，默认 0.12",
    )
    parser.add_argument(
        "--feather-ratio",
        type=float,
        default=0.12,
        help="patch 边缘羽化宽度占 patch 边长的比例，默认 0.12",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="随机裁剪/放置的种子，默认 42"
    )
    parser.add_argument(
        "--visible-ratio",
        type=float,
        default=DEFAULT_VISIBLE_RATIO,
        help="锚点安全可视区域占图片宽高的比例，默认 0.65",
    )
    parser.add_argument(
        "--rot-seam-ratio",
        type=float,
        default=DEFAULT_ROT_SEAM_RATIO,
        help="rot 锚点在 UV 接缝每侧的安全宽度比例，默认 0.20",
    )
    parser.add_argument(
        "--candidates-per-donor",
        type=int,
        default=6,
        help="从每张候选 donor 采样多少个 patch，再选择结构差异最大的一个",
    )
    parser.add_argument(
        "--donor-selection",
        choices=("balanced", "random"),
        default="balanced",
        help="donor 选择策略；默认均衡轮换，避免集中来自一张纹理",
    )
    parser.add_argument(
        "--no-color-match",
        action="store_true",
        help="不把 patch 的每通道均值/标准差匹配到目标局部区域",
    )
    parser.add_argument(
        "--output-format",
        choices=("png", "jpg"),
        default="png",
        help="输出格式；默认 PNG，避免额外的有损压缩成为实验混杂",
    )
    parser.add_argument(
        "--layout",
        choices=("rot", "trans", "both"),
        default="both",
        help="生成哪种锚点布局，默认 both（每张源图同时生成 rot 和 trans）",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="覆盖已有改造纹理和 manifest"
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 JSONL 文件：{path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是有效 JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} 不是 JSON 对象")
            records.append(record)
    return records


def find_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"找不到纹理目录：{directory}")
    images = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(images) < 2:
        raise RuntimeError("NSA 纹理至少需要两张，才能从另一张图提取 patch")
    return images


def stable_seed(text: str, seed: int) -> int:
    # 不使用 Python 内置 hash，保证不同进程/机器得到相同结果。
    value = seed & 0xFFFFFFFF
    for char in text.encode("utf-8"):
        value = (value * 16777619) ^ char
        value &= 0xFFFFFFFF
    return value


def crop_patch(
    donor: Image.Image, width: int, height: int, rng: random.Random
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """在 donor 上随机取 patch；donor 太小时先按比例放大。"""
    if donor.width < width or donor.height < height:
        scale = max(width / donor.width, height / donor.height)
        resized = donor.resize(
            (max(width, round(donor.width * scale)), max(height, round(donor.height * scale))),
            Image.Resampling.LANCZOS,
        )
    else:
        resized = donor
    left = rng.randint(0, resized.width - width)
    top = rng.randint(0, resized.height - height)
    box = (left, top, left + width, top + height)
    return resized.crop(box), box


def match_local_color(patch: np.ndarray, target: np.ndarray) -> np.ndarray:
    """按 RGB 通道匹配局部均值/标准差，保留 patch 的纹理结构。"""
    patch_float = patch.astype(np.float32)
    target_float = target.astype(np.float32)
    patch_mean = patch_float.reshape(-1, 3).mean(axis=0)
    patch_std = patch_float.reshape(-1, 3).std(axis=0)
    target_mean = target_float.reshape(-1, 3).mean(axis=0)
    target_std = target_float.reshape(-1, 3).std(axis=0)
    scale = np.divide(
        target_std,
        patch_std,
        out=np.ones_like(target_std),
        where=patch_std >= 1.0,
    )
    matched = (patch_float - patch_mean) * scale + target_mean
    return np.clip(matched, 0, 255).astype(np.uint8)


def target_region(
    target: Image.Image,
    layout: str,
    size: int,
    trans_position: tuple[int, int] | None,
    rot_position: tuple[int, int] | None,
) -> np.ndarray:
    """取得目标纹理中与某种布局对应的 patch 大小局部区域。"""
    array = np.asarray(target.convert("RGB"))
    if layout == "trans":
        if trans_position is None:
            raise ValueError("trans 布局缺少目标位置")
        left, top = trans_position
        return array[top : top + size, left : left + size]
    if rot_position is None:
        raise ValueError("rot 布局缺少目标位置")
    left, top = rot_position
    x_indices = (left + np.arange(size)) % target.width
    return array[top : top + size, x_indices]


def random_visible_position(
    width: int,
    height: int,
    patch_size: int,
    visible_ratio: float,
    rng: random.Random,
) -> tuple[int, int]:
    """在中心可视区域内随机放置 patch，并保证 patch 完整落在区域中。"""
    visible_width = max(patch_size, round(width * visible_ratio))
    visible_height = max(patch_size, round(height * visible_ratio))
    visible_width = min(width, visible_width)
    visible_height = min(height, visible_height)
    min_left = (width - visible_width) // 2
    min_top = (height - visible_height) // 2
    max_left = min_left + visible_width - patch_size
    max_top = min_top + visible_height - patch_size
    return rng.randint(min_left, max_left), rng.randint(min_top, max_top)


def random_rot_position(
    width: int,
    height: int,
    patch_size: int,
    visible_ratio: float,
    seam_ratio: float,
    rng: random.Random,
) -> tuple[int, int]:
    """在 UV 接缝两侧的球体正面安全区域内随机放置 patch。"""
    seam_width = round(width * seam_ratio)
    if patch_size > 2 * seam_width:
        raise ValueError(
            "rot 锚点大于 UV 接缝安全区域；请增大 --rot-seam-ratio "
            "或减小 --patch-ratio"
        )
    _, top = random_visible_position(
        width, height, patch_size, visible_ratio, rng
    )
    max_center_distance = seam_width - patch_size // 2
    center_offset = rng.randint(-max_center_distance, max_center_distance)
    left = (center_offset - patch_size // 2) % width
    return left, top


def structural_difference(patch: np.ndarray, target: np.ndarray) -> float:
    """颜色匹配后计算局部纹理结构差异，越大越容易形成独特锚点。"""
    matched = match_local_color(patch, target)
    patch_gray = matched.astype(np.float32).mean(axis=2)
    target_gray = target.astype(np.float32).mean(axis=2)
    pixel_difference = np.mean(np.abs(patch_gray - target_gray)) / 255.0

    patch_dx = np.diff(patch_gray, axis=1)
    patch_dy = np.diff(patch_gray, axis=0)
    target_dx = np.diff(target_gray, axis=1)
    target_dy = np.diff(target_gray, axis=0)
    gradient_difference = (
        np.mean(np.abs(patch_dx - target_dx))
        + np.mean(np.abs(patch_dy - target_dy))
    ) / (2.0 * 255.0)
    return float(0.6 * pixel_difference + 0.4 * gradient_difference)


def choose_distinctive_patch(
    donors: list[Path],
    target: Image.Image,
    size: int,
    layouts: list[str],
    rng: random.Random,
    candidates_per_donor: int,
    trans_position: tuple[int, int] | None,
    rot_position: tuple[int, int] | None,
) -> tuple[Image.Image, Path, tuple[int, int, int, int], float]:
    """从所有 donor 的候选 patch 中选择颜色匹配后结构差异最大的一个。"""
    target_regions = [
        target_region(target, layout, size, trans_position, rot_position)
        for layout in layouts
    ]
    best: tuple[Image.Image, Path, tuple[int, int, int, int], float] | None = None
    for donor_path in donors:
        with Image.open(donor_path) as donor_source:
            donor = donor_source.convert("RGB")
            for _ in range(candidates_per_donor):
                patch, box = crop_patch(donor, size, size, rng)
                patch_array = np.asarray(patch.convert("RGB"))
                score = sum(
                    structural_difference(patch_array, region)
                    for region in target_regions
                ) / len(target_regions)
                if best is None or score > best[3]:
                    best = (patch.copy(), donor_path, box, score)
    if best is None:
        raise RuntimeError("没有生成有效的 donor patch 候选")
    return best


def choose_donor(
    donors: list[Path], usage: dict[str, int], rng: random.Random
) -> Path:
    """优先选择当前使用次数最少的 donor；平局时用固定随机数打散顺序。"""
    minimum = min(usage.get(path.stem, 0) for path in donors)
    candidates = [path for path in donors if usage.get(path.stem, 0) == minimum]
    return candidates[rng.randrange(len(candidates))]


def feather_mask(width: int, height: int, feather: int) -> Image.Image:
    """生成软边矩形 mask，避免引入圆点/星形等人为语义形状。"""
    if feather <= 0:
        return Image.new("L", (width, height), 255)
    # 到最近边界的距离决定透明度：边缘为 0，进入 feather 像素后为 255。
    y, x = np.ogrid[:height, :width]
    distance = np.minimum.reduce(
        (x + np.zeros_like(y), width - 1 - x + np.zeros_like(y),
         y + np.zeros_like(x), height - 1 - y + np.zeros_like(x))
    ).astype(np.float32)
    progress = np.clip(distance / feather, 0.0, 1.0)
    smooth = progress * progress * (3.0 - 2.0 * progress)
    return Image.fromarray(np.rint(smooth * 255).astype(np.uint8), mode="L")


def paste_patch(
    target: Image.Image,
    patch: Image.Image,
    left: int,
    top: int,
    feather: int,
    color_match: bool,
) -> None:
    target_array = np.asarray(target.convert("RGB"))
    patch_array = np.asarray(patch.convert("RGB"))
    height, width = patch_array.shape[:2]
    if color_match:
        local = target_array[top : top + height, left : left + width]
        patch_array = match_local_color(patch_array, local)
    patch_image = Image.fromarray(patch_array, mode="RGB")
    mask = feather_mask(width, height, feather)
    target.paste(patch_image, (left, top), mask)


def paste_wrapped_patch(
    target: Image.Image,
    patch: Image.Image,
    left: int,
    top: int,
    feather: int,
    color_match: bool,
) -> list[list[int]]:
    """在水平循环纹理上粘贴 patch，返回实际写入的一或两个矩形区域。"""
    patch_array = np.asarray(patch.convert("RGB"))
    height, width = patch_array.shape[:2]
    x_indices = (left + np.arange(width)) % target.width
    target_array = np.asarray(target.convert("RGB"))
    local = target_array[top : top + height, x_indices]
    if color_match:
        patch_array = match_local_color(patch_array, local)

    patch_image = Image.fromarray(patch_array, mode="RGB")
    mask = feather_mask(width, height, feather)
    first_width = min(width, target.width - left)
    first_patch = patch_image.crop((0, 0, first_width, height))
    first_mask = mask.crop((0, 0, first_width, height))
    target.paste(first_patch, (left, top), first_mask)

    second_width = width - first_width
    boxes = [[left, top, left + first_width, top + height]]
    if second_width > 0:
        second_patch = patch_image.crop((first_width, 0, width, height))
        second_mask = mask.crop((first_width, 0, width, height))
        target.paste(second_patch, (0, top), second_mask)
        boxes.append([0, top, second_width, top + height])
    return boxes


def save_image(image: Image.Image, path: Path, output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "png":
        image.save(path, format="PNG", compress_level=6)
    else:
        image.save(path, format="JPEG", quality=100, subsampling=0, optimize=False)


def project_relative_or_absolute(path: Path) -> str:
    """项目内写相对路径；显式指定到项目外时保留绝对路径。"""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def remove_previous_outputs(manifest_path: Path, output_dir: Path) -> int:
    """仅删除旧 manifest 记录且仍位于 output_dir 直接下级的派生图片。"""
    if not manifest_path.is_file():
        return 0
    removed = 0
    output_root = output_dir.resolve()
    for record in read_jsonl(manifest_path):
        value = record.get("output")
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        resolved = path.resolve()
        if resolved.parent != output_root:
            raise RuntimeError(f"拒绝删除输出目录之外的旧文件：{resolved}")
        if resolved.is_file():
            resolved.unlink()
            removed += 1
    return removed


def main() -> None:
    args = parse_args()
    if not 0 < args.patch_ratio < 0.5:
        raise ValueError("--patch-ratio 应在 (0, 0.5) 内")
    if not 0 <= args.feather_ratio < 0.5:
        raise ValueError("--feather-ratio 应在 [0, 0.5) 内")
    if args.candidates_per_donor < 1:
        raise ValueError("--candidates-per-donor 至少为 1")
    if not 0 < args.visible_ratio <= DEFAULT_VISIBLE_RATIO:
        raise ValueError(
            f"--visible-ratio 应在 (0, {DEFAULT_VISIBLE_RATIO}] 内，"
            "避免锚点落到视频可视区域之外"
        )
    if not 0 < args.rot_seam_ratio <= 0.25:
        raise ValueError(
            "--rot-seam-ratio 应在 (0, 0.25] 内，"
            "避免 rot 锚点离开球体正面可见半球"
        )

    source_images = find_images(args.source_dir)
    layouts = ["rot", "trans"] if args.layout == "both" else [args.layout]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"已存在 {manifest_path}；如需重做请使用 --overwrite")
    if args.overwrite:
        removed = remove_previous_outputs(manifest_path, args.output_dir)
        if removed:
            print(f"已清理上一轮 manifest 记录的 {removed} 张派生纹理")

    records: list[dict[str, Any]] = []
    generated = 0
    donor_usage: dict[str, int] = defaultdict(int)
    planned_count = len(source_images) * len(layouts)

    for target_path in source_images:
        donors = [path for path in source_images if path != target_path]
        rng = random.Random(stable_seed(target_path.name, args.seed))
        donor = (
            choose_donor(donors, donor_usage, rng)
            if args.donor_selection == "balanced"
            else donors[rng.randrange(len(donors))]
        )
        donor_usage[donor.stem] += 1

        with Image.open(target_path) as target_source:
            original_target = target_source.convert("RGB")
            patch_size = max(
                8, round(min(original_target.width, original_target.height) * args.patch_ratio)
            )
            feather = round(patch_size * args.feather_ratio)
            trans_position = (
                random_visible_position(
                    original_target.width,
                    original_target.height,
                    patch_size,
                    args.visible_ratio,
                    rng,
                )
                if "trans" in layouts
                else None
            )
            rot_position = (
                random_rot_position(
                    original_target.width,
                    original_target.height,
                    patch_size,
                    args.visible_ratio,
                    args.rot_seam_ratio,
                    rng,
                )
                if "rot" in layouts
                else None
            )

            patch, donor_path, donor_box, distinctiveness_score = (
                choose_distinctive_patch(
                    [donor],
                    original_target,
                    patch_size,
                    layouts,
                    rng,
                    args.candidates_per_donor,
                    trans_position,
                    rot_position,
                )
            )

            for layout in layouts:
                target = original_target.copy()
                if layout == "trans":
                    if trans_position is None:
                        raise RuntimeError("trans 布局缺少目标位置")
                    left, top = trans_position
                    paste_patch(
                        target,
                        patch,
                        left,
                        top,
                        feather,
                        not args.no_color_match,
                    )
                    target_boxes = [
                        [left, top, left + patch_size, top + patch_size]
                    ]
                    placement = "random_visible_region"
                else:
                    if rot_position is None:
                        raise RuntimeError("rot 布局缺少目标位置")
                    left, top = rot_position
                    target_boxes = paste_wrapped_patch(
                        target,
                        patch,
                        left,
                        top,
                        feather,
                        not args.no_color_match,
                    )
                    placement = "equirectangular_front_seam_region"

                suffix = ".png" if args.output_format == "png" else ".jpg"
                output_name = f"{target_path.stem}_anchor_{layout}{suffix}"
                output_path = args.output_dir / output_name
                if output_path.exists() and not args.overwrite:
                    raise FileExistsError(f"已存在输出文件：{output_path}")
                save_image(target, output_path, args.output_format)

                records.append(
                    {
                        "output": project_relative_or_absolute(output_path),
                        "target_source": target_path.stem,
                        "target_path": project_relative_or_absolute(target_path),
                        "layout": layout,
                        "placement": placement,
                        "donor_source": donor_path.stem,
                        "donor_path": project_relative_or_absolute(donor_path),
                        "donor_box_xyxy": list(donor_box),
                        "target_boxes_xyxy": target_boxes,
                        "patch_size": patch_size,
                        "distinctiveness_score": distinctiveness_score,
                        "candidates_per_donor": args.candidates_per_donor,
                        "feather": feather,
                        "color_match": not args.no_color_match,
                        "output_format": args.output_format,
                        "seed": args.seed,
                        "visible_ratio": args.visible_ratio,
                        "rot_seam_ratio": args.rot_seam_ratio,
                    }
                )
                generated += 1
                print(
                    f"[{generated}/{planned_count}] "
                    f"{target_path.stem}/{layout}: {output_path}"
                )

    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(manifest_path)
    print(f"完成：生成 {generated} 张纹理，清单：{manifest_path}")


if __name__ == "__main__":
    main()
