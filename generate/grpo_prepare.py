"""Data preparation script for image overlap grounding GRPO training.

Generates parquet files from image pairs with overlap bbox annotations.
The parquet follows verl's standard format: prompt with <image> placeholders,
images list, data_source, reward_model.ground_truth.
"""
import argparse
import io
import json
import os
from pathlib import Path

import pandas as pd
from PIL import Image


PROMPT_TEMPLATE = (
    "You are given two images that may have overlapping regions. "
    "Your task is to identify the overlapping area in each image. "
    "Output the bounding box for the overlap region in each image.\n\n"
    "Image A:\n<image>\n\nImage B:\n<image>\n\n"
    "Please output your answer in the following JSON format:\n"
    '{"bbox_a": [x1, y1, x2, y2], "bbox_b": [x1, y1, x2, y2]}'
)


def image_to_bytes(img_path: str) -> bytes:
    img = Image.open(img_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def load_annotation(ann_path: str):
    """Load overlap annotations from a JSON file.

    Expected format (one file per image pair):
    {
        "image_a": "path/to/img_a.jpg",
        "image_b": "path/to/img_b.jpg",
        "bbox_a": [x1, y1, x2, y2],
        "bbox_b": [x1, y1, x2, y2]
    }
    """
    with open(ann_path) as f:
        return json.load(f)


def build_parquet_rows(annotations: list[dict], start_index: int = 0) -> list[dict]:
    rows = []
    for i, ann in enumerate(annotations):
        img_a_bytes = image_to_bytes(ann["image_a"])
        img_b_bytes = image_to_bytes(ann["image_b"])

        row = {
            "prompt": [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE,
                }
            ],
            "images": [
                {"bytes": img_a_bytes},
                {"bytes": img_b_bytes},
            ],
            "data_source": "image_overlap",
            "reward_model": {
                "ground_truth": {
                    "bbox_a": ann["bbox_a"],
                    "bbox_b": ann["bbox_b"],
                }
            },
            "extra_info": {"index": start_index + i},
        }
        rows.append(row)
    return rows


def collect_annotations(data_dir: str) -> list[dict]:
    """Collect all annotation JSON files from a directory.

    Each .json file should contain one image pair annotation (see load_annotation).
    """
    annotations = []
    for json_path in sorted(Path(data_dir).glob("*.json")):
        ann = load_annotation(str(json_path))
        base_dir = Path(data_dir)
        ann["image_a"] = str(base_dir / ann["image_a"]) if not os.path.isabs(ann["image_a"]) else ann["image_a"]
        ann["image_b"] = str(base_dir / ann["image_b"]) if not os.path.isabs(ann["image_b"]) else ann["image_b"]
        annotations.append(ann)
    return annotations


def main():
    parser = argparse.ArgumentParser(description="Prepare parquet data for image overlap grounding")
    parser.add_argument("--train_dir", type=str, required=True, help="Directory containing training annotations")
    parser.add_argument("--val_dir", type=str, required=True, help="Directory containing validation annotations")
    parser.add_argument("--output_dir", type=str, default="./data/image_overlap", help="Output directory for parquet files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_anns = collect_annotations(args.train_dir)
    val_anns = collect_annotations(args.val_dir)

    print(f"Found {len(train_anns)} training pairs, {len(val_anns)} validation pairs")

    train_rows = build_parquet_rows(train_anns, start_index=0)
    val_rows = build_parquet_rows(val_anns, start_index=0)

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    print(f"Saved train parquet: {train_path} ({len(train_df)} rows)")
    print(f"Saved val parquet: {val_path} ({len(val_df)} rows)")


if __name__ == "__main__":
    main()