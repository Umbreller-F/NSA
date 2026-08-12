"""把一张图片生成为“摄像机平移”效果的视频。"""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


# ========== 硬编码配置 ==========
IMAGE_PATH = Path("/data/dataset/TempCompass/non-semantic-anchor/textures/texture_6.jpg")
OUTPUT_PATH = Path(
    "/data/dataset/TempCompass/non-semantic-anchor/videos/translation_006.mp4"
)

FPS = 30
DURATION_SECONDS = 5
VIEW_RATIO = 0.65  # 摄像机视野占原图宽、高的比例；越小越像放大
DIRECTION = "right"  # 可选：right、left、down、up


def even_size(value: int) -> int:
    """返回不大于 value 的正偶数，避免 H.264 对奇数尺寸编码失败。"""
    return max(2, value - value % 2)


def camera_path(
    image_width: int,
    image_height: int,
    view_width: int,
    view_height: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """根据移动方向返回裁剪窗口左上角的起点和终点。"""
    max_x = image_width - view_width
    max_y = image_height - view_height
    center_x = max_x // 2
    center_y = max_y // 2

    paths = {
        "right": ((0, center_y), (max_x, center_y)),
        "left": ((max_x, center_y), (0, center_y)),
        "down": ((center_x, 0), (center_x, max_y)),
        "up": ((center_x, max_y), (center_x, 0)),
    }
    if DIRECTION not in paths:
        raise ValueError(f"不支持的移动方向: {DIRECTION!r}，可选值为 {tuple(paths)}")
    return paths[DIRECTION]


def main() -> None:
    if not 0 < VIEW_RATIO < 1:
        raise ValueError("VIEW_RATIO 必须在 0 和 1 之间")
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"找不到输入图片: {IMAGE_PATH}")

    # 统一转为 RGB，避免灰度图或带透明通道的图片无法被编码。
    with Image.open(IMAGE_PATH) as source:
        image = np.asarray(source.convert("RGB"))

    image_height, image_width = image.shape[:2]
    view_width = even_size(int(image_width * VIEW_RATIO))
    view_height = even_size(int(image_height * VIEW_RATIO))
    start, end = camera_path(
        image_width, image_height, view_width, view_height
    )

    frame_count = max(2, round(FPS * DURATION_SECONDS))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 使用流式写入，避免把所有视频帧同时保存在内存里。
    with imageio.get_writer(
        str(OUTPUT_PATH),
        fps=FPS,
        codec="libx264",
        is_batch=False,
        out_pixel_format="yuv420p",
    ) as writer:
        for frame_index in range(frame_count):
            progress = frame_index / (frame_count - 1)
            # smoothstep 缓入缓出，让镜头起步和停止更自然。
            progress = progress * progress * (3.0 - 2.0 * progress)
            x = round(start[0] + (end[0] - start[0]) * progress)
            y = round(start[1] + (end[1] - start[1]) * progress)
            frame = image[y : y + view_height, x : x + view_width]
            writer.append_data(frame)

    print(
        f"已保存: {OUTPUT_PATH} "
        f"({view_width}x{view_height}, {frame_count} 帧, {FPS} FPS)"
    )


if __name__ == "__main__":
    main()
