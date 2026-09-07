"""从原始图片生成带重叠区域的图像对，用于 overlap grounding GRPO 数据。

对每张原始图片，随机划出两个一样大的矩形区域，保证两者的重叠面积占比
（交集面积 / 单个区域面积）落在 [overlap_min, overlap_max] 区间内。区域
尺寸、宽高比、位置、偏移方向和重叠比例都随机，但整体由固定随机种子控
制，完全可复现。

输出目录结构（默认 ``textures/overlap/pairs``）：

    pairs/
        train/
            images/pair_00000_a.jpg
            images/pair_00000_b.jpg
            pair_00000.json
        val/
            ...

每个 JSON 的格式与 ``generate/grpo_prepare.py`` 的 ``load_annotation``
一致（图片路径为相对 data dir 的相对路径）：

    {"image_a": "images/pair_00000_a.jpg",
     "image_b": "images/pair_00000_b.jpg",
     "bbox_a": [x1, y1, x2, y2],
     "bbox_b": [x1, y1, x2, y2]}

bbox 是重叠区域在各自裁剪图中的坐标。train/val 按源图划分，同一张原图
的所有图像对只会落在同一个 split 中，避免泄漏。

示例（在 non-semantic-anchor 目录执行）：

    python generate/overlap_pairs.py
    python generate/overlap_pairs.py --seed 7 --pairs-per-image 8

随后可运行：

    python generate/grpo_prepare.py \
        --train_dir textures/overlap/pairs/train \
        --val_dir textures/overlap/pairs/val
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC_DIR = PROJECT_ROOT / "textures" / "overlap" / "original"
DEFAULT_OUT_DIR = PROJECT_ROOT / "textures" / "overlap" / "pairs"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从原始图片生成带重叠区域的图像对及 grpo_prepare 格式标注"
    )
    parser.add_argument("--src-dir", type=Path, default=DEFAULT_SRC_DIR, help="原始图片目录")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="输出根目录")
    parser.add_argument("--seed", type=int, default=0, help="全局随机种子")
    parser.add_argument("--pairs-per-image", type=int, default=4, help="每张原图生成的图像对数")
    parser.add_argument("--val-split", type=float, default=0.25, help="划入 val 的原图比例（按图划分）")
    parser.add_argument(
        "--box-scale",
        type=float,
        nargs=2,
        default=[0.4, 0.7],
        metavar=("MIN", "MAX"),
        help="区域边长占原图短边比例的随机范围",
    )
    parser.add_argument("--overlap-min", type=float, default=0.3, help="重叠面积占比下限")
    parser.add_argument("--overlap-max", type=float, default=0.8, help="重叠面积占比上限")
    parser.add_argument(
        "--max-side",
        type=int,
        default=1024,
        help="保存裁剪图的最长边像素（等比缩放，bbox 同步换算）；<=0 表示不缩放",
    )
    return parser.parse_args()


def sample_pair_geometry(rng: random.Random, img_w: int, img_h: int, args) -> dict:
    """为一对区域采样几何参数，返回裁剪框与重叠 bbox（均为原图/裁剪图坐标）。

    两个区域尺寸相同。偏移量 (dx, dy) 通过先采目标重叠比例 r，再分解为
    x/y 两个方向的重叠分数 ux * uy = r 得到，保证重叠占比精确落在区间内。
    """
    short = min(img_w, img_h)
    scale_lo, scale_hi = args.box_scale
    box_w = int(short * rng.uniform(scale_lo, scale_hi))
    box_h = int(short * rng.uniform(scale_lo, scale_hi))

    # 目标重叠比例 r = (重叠宽/区域宽) * (重叠高/区域高)
    r = rng.uniform(args.overlap_min, args.overlap_max)
    ux = rng.uniform(r, 1.0)
    uy = r / ux
    dx = box_w - max(1, round(ux * box_w))
    dy = box_h - max(1, round(uy * box_h))
    # 随机偏移方向
    off_x = dx * rng.choice((-1, 1))
    off_y = dy * rng.choice((-1, 1))

    # 区域 A 的可放置范围需同时容纳区域 B = A + (off_x, off_y)
    x_lo = max(0, -off_x)
    x_hi = img_w - box_w - max(0, off_x)
    y_lo = max(0, -off_y)
    y_hi = img_h - box_h - max(0, off_y)
    x_a = rng.randint(x_lo, max(x_lo, x_hi))
    y_a = rng.randint(y_lo, max(y_lo, y_hi))
    x_b = x_a + off_x
    y_b = y_a + off_y

    # 重叠区域在各自裁剪图中的坐标
    inter_w = box_w - abs(off_x)
    inter_h = box_h - abs(off_y)
    bbox_a = [max(0, off_x), max(0, off_y), max(0, off_x) + inter_w, max(0, off_y) + inter_h]
    bbox_b = [max(0, -off_x), max(0, -off_y), max(0, -off_x) + inter_w, max(0, -off_y) + inter_h]

    return {
        "crop_a": (x_a, y_a, x_a + box_w, y_a + box_h),
        "crop_b": (x_b, y_b, x_b + box_w, y_b + box_h),
        "bbox_a": bbox_a,
        "bbox_b": bbox_b,
        "overlap_ratio": (inter_w * inter_h) / (box_w * box_h),
    }


def save_crop(img: Image.Image, box: tuple, path: Path, max_side: int) -> float:
    """裁剪并保存区域图，返回相对原裁剪图的缩放系数。"""
    crop = img.crop(box)
    scale = 1.0
    if max_side > 0 and max(crop.size) > max_side:
        scale = max_side / max(crop.size)
        new_size = (round(crop.width * scale), round(crop.height * scale))
        crop = crop.resize(new_size, Image.LANCZOS)
    crop.save(path, quality=95)
    return scale


def scale_bbox(bbox: list, scale: float) -> list:
    return [round(v * scale) for v in bbox]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    src_paths = sorted(
        p for p in args.src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not src_paths:
        raise SystemExit(f"在 {args.src_dir} 下没有找到图片")

    # 按源图划分 train/val
    shuffled = src_paths[:]
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * args.val_split)
    split_of = {p: ("val" if i < n_val else "train") for i, p in enumerate(shuffled)}

    pair_index = 0
    counts = {"train": 0, "val": 0}
    for src_path in src_paths:
        split = split_of[src_path]
        split_dir = args.out_dir / split
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(src_path).convert("RGB")
        for _ in range(args.pairs_per_image):
            geo = sample_pair_geometry(rng, img.width, img.height, args)
            pair_id = f"pair_{pair_index:05d}"

            path_a = images_dir / f"{pair_id}_a.jpg"
            path_b = images_dir / f"{pair_id}_b.jpg"
            scale_a = save_crop(img, geo["crop_a"], path_a, args.max_side)
            scale_b = save_crop(img, geo["crop_b"], path_b, args.max_side)

            ann = {
                "image_a": f"images/{path_a.name}",
                "image_b": f"images/{path_b.name}",
                "bbox_a": scale_bbox(geo["bbox_a"], scale_a),
                "bbox_b": scale_bbox(geo["bbox_b"], scale_b),
            }
            with open(split_dir / f"{pair_id}.json", "w") as f:
                json.dump(ann, f, indent=2)

            counts[split] += 1
            pair_index += 1
        img.close()

    print(f"共处理 {len(src_paths)} 张原图（val {n_val} 张，train {len(src_paths) - n_val} 张）")
    print(f"生成图像对：train {counts['train']}，val {counts['val']}，输出目录 {args.out_dir}")


if __name__ == "__main__":
    main()
