"""一次性修复 v2 translation 上下左右方向命名及已有结果。

旧版生成器以“镜头移动方向”命名平移视频，而评测问题询问的是画面中图案
的运动方向，两者相反。本脚本不重新编码视频，只交换已有文件名并同步修复
指定 JSONL。顺时针与逆时针不受影响。
"""

import argparse
import json
import shutil
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEOS_ROOT = PROJECT_ROOT / "videos"
FRAMES_ROOT = PROJECT_ROOT / "gpt_frames"
RESULTS_ROOT = PROJECT_ROOT / "results"
RESULT_FILES = (
    RESULTS_ROOT / "gemini-3.5-flash-v2.jsonl",
    RESULTS_ROOT / "gpt-5.6-sol-v2.jsonl",
    RESULTS_ROOT / "gemini-3.5-flash-v2_label.jsonl",
)
DONE_MARKER = RESULTS_ROOT / ".trans_v2_direction_fix_done"

DIRECTION_SWAP = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
DIRECTION_ORDER = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "clockwise": 4,
    "counterclockwise": 5,
}
TASK_ORDER = {"rot": 0, "trans": 1}
CATEGORY_ORDER = {"NSA": 0, "SA": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="修复已有 trans v2 视频文件名和三个结果 JSONL 的方向/GT"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅检查并打印修改数量，不写入任何文件",
    )
    return parser.parse_args()


def natural_key(value: str) -> tuple:
    """自然排序源图编号。"""
    parts = []
    current = ""
    numeric = None
    for char in value:
        is_numeric = char.isdigit()
        if numeric is not None and is_numeric != numeric:
            parts.append(int(current) if numeric else current.lower())
            current = ""
        current += char
        numeric = is_numeric
    if current:
        parts.append(int(current) if numeric else current.lower())
    return tuple(parts)


def result_sort_key(record: dict) -> tuple:
    """与评测脚本保持一致的固定结果顺序。"""
    return (
        TASK_ORDER.get(record.get("task"), 99),
        CATEGORY_ORDER.get(record.get("category"), 99),
        natural_key(str(record.get("source", ""))),
        DIRECTION_ORDER.get(record.get("direction"), 99),
        str(record.get("video_name", record.get("video_path", ""))),
    )


def replace_direction_suffix(value: str, old: str, new: str) -> str:
    """只替换路径或文件名末尾的方向，避免改动其他文本。"""
    suffixes = (f"_{old}.mp4", f"_{old}")
    for suffix in suffixes:
        if value.endswith(suffix):
            return value[: -len(suffix)] + suffix.replace(f"_{old}", f"_{new}")
    raise ValueError(f"路径方向与记录不一致：{value!r}，direction={old!r}")


def fix_record(record: dict) -> bool:
    """原地修复一条 trans 上下左右记录；返回是否修改。"""
    if record.get("task") != "trans":
        return False
    old_direction = record.get("direction")
    new_direction = DIRECTION_SWAP.get(old_direction)
    if new_direction is None:
        return False

    for field in ("video_path", "video_name", "frames_dir"):
        value = record.get(field)
        if value:
            record[field] = replace_direction_suffix(str(value), old_direction, new_direction)
    record["direction"] = new_direction
    record["ground_truth"] = f"图案向{'上' if new_direction == 'up' else '下' if new_direction == 'down' else '左' if new_direction == 'left' else '右'}平移"
    return True


def load_and_fix_jsonl(path: Path) -> tuple[list[dict], int]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少目标结果文件：{path}")
    records = []
    changed = 0
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行 JSON 无效：{exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path} 第 {line_number} 行不是 JSON 对象")
            changed += int(fix_record(record))
            records.append(record)
    records.sort(key=result_sort_key)
    return records, changed


def collect_swap_pairs(root: Path, suffix: str) -> list[tuple[Path, Path]]:
    """收集每个上下/左右文件或目录的旧路径和新路径。"""
    if not root.is_dir():
        return []
    pairs = []
    patterns = (f"trans_*_up{suffix}", f"trans_*_down{suffix}",
                f"trans_*_left{suffix}", f"trans_*_right{suffix}")
    for pattern in patterns:
        for source in root.glob(pattern):
            old_direction = source.name.removesuffix(suffix).rsplit("_", 1)[-1]
            new_name = replace_direction_suffix(source.name, old_direction, DIRECTION_SWAP[old_direction])
            pairs.append((source, source.with_name(new_name)))
    return sorted(pairs)


def validate_pairs(pairs: list[tuple[Path, Path]], expected: int, label: str) -> None:
    if len(pairs) != expected:
        raise RuntimeError(f"{label} 应有 {expected} 个上下左右目标，实际找到 {len(pairs)} 个")
    sources = {source for source, _ in pairs}
    targets = {target for _, target in pairs}
    if len(sources) != len(pairs) or len(targets) != len(pairs) or sources != targets:
        raise RuntimeError(f"{label} 交换集合不完整或存在冲突")


def swap_paths(pairs: list[tuple[Path, Path]]) -> None:
    """通过唯一临时名两阶段交换，防止 up/down 等目标互相覆盖。"""
    temporary = []
    for source, target in pairs:
        temp = source.with_name(f".{source.name}.direction-fix-{uuid.uuid4().hex}.tmp")
        source.rename(temp)
        temporary.append((temp, target))
    for temp, target in temporary:
        temp.rename(target)


def write_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_name(path.name + ".direction-fix.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if DONE_MARKER.exists():
        print(f"检测到完成标记 {DONE_MARKER}，修复已经执行过，本次不做任何修改。")
        return

    fixed_results = {}
    for path in RESULT_FILES:
        records, changed = load_and_fix_jsonl(path)
        fixed_results[path] = records
        print(f"JSONL：{path.name}，将修复 {changed} 条 trans 上下左右记录")
        if changed != 88:
            raise RuntimeError(f"{path.name} 预期修复 88 条，实际为 {changed} 条")

    video_pairs = collect_swap_pairs(VIDEOS_ROOT, ".mp4")
    frame_pairs = collect_swap_pairs(FRAMES_ROOT, "")
    validate_pairs(video_pairs, 88, "videos")
    validate_pairs(frame_pairs, 88, "gpt_frames")
    print(f"视频：将交换 {len(video_pairs)} 个文件名（不重新编码）")
    print(f"抽帧目录：将交换 {len(frame_pairs)} 个目录名")

    if args.dry_run:
        print("dry-run 检查通过，未修改任何文件。")
        return

    # JSONL 先留只读备份，保留人工标注和所有模型原始输出。
    for path in RESULT_FILES:
        backup = path.with_name(path.name + ".before_trans_direction_fix.bak")
        if backup.exists():
            raise RuntimeError(f"备份文件已存在，为避免覆盖而停止：{backup}")
        shutil.copy2(path, backup)

    swap_paths(video_pairs)
    swap_paths(frame_pairs)
    for path, records in fixed_results.items():
        write_jsonl(path, records)
    DONE_MARKER.write_text(
        "Fixed trans up/down/left/right from camera direction to pattern-motion direction.\n",
        encoding="utf-8",
    )
    print("修复完成。原 JSONL 备份后缀：.before_trans_direction_fix.bak")


if __name__ == "__main__":
    main()
