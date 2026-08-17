"""使用 textures/v3 的 NSA、SA 和 NSA_anchor 纹理生成球面旋转视频。"""

import argparse
import os
import subprocess
from pathlib import Path


# 必须在导入 PyVista/VTK 前设置离屏渲染环境变量。
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.2")
os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFF_SCREEN", "true")

import numpy as np
import pyvista as pv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXTURES_ROOT = PROJECT_ROOT / "textures" / "v3"
OUTPUT_ROOT = PROJECT_ROOT / "videos" / "v3"
# ``NSA_anchor`` 是独立实验条件，不并入原始 NSA 集合。
CATEGORY_DIRS = {
    "NSA": TEXTURES_ROOT / "NSA",
    "SA": TEXTURES_ROOT / "SA",
    "NSA_ANCHOR": TEXTURES_ROOT / "NSA_anchor",
}
CATEGORIES = tuple(CATEGORY_DIRS)

FRAMES = 150
FPS = 30
ROTATION_DEG = 180
WINDOW_SIZE = (1440, 1440)

# 从位于 +Z 方向、Y 轴朝上的摄像机视角定义画面中的运动方向。
# up/down 绕 X 轴，left/right 绕 Y 轴，逆/顺时针绕 Z 轴。
ROTATIONS = {
    "up": ("x", -1),
    "down": ("x", 1),
    "left": ("y", -1),
    "right": ("y", 1),
    "counterclockwise": ("z", 1),
    "clockwise": ("z", -1),
}


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
            "中的 JPG/PNG 生成六方向球体旋转视频，并保存到 videos/v3"
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


def make_equirect_sphere(
    radius: float = 1.0,
    n_theta: int = 160,
    n_phi: int = 80,
) -> pv.PolyData:
    """构建具有标准等距柱状 UV、且接缝处复制点列的球面网格。"""
    theta = np.linspace(0, 2 * np.pi, n_theta + 1)
    phi = np.linspace(0, np.pi, n_phi + 1)

    points = []
    uvs = []
    for phi_index, phi_value in enumerate(phi):
        for theta_index, theta_value in enumerate(theta):
            points.append(
                [
                    radius * np.sin(phi_value) * np.sin(theta_value),
                    radius * np.cos(phi_value),
                    radius * np.sin(phi_value) * np.cos(theta_value),
                ]
            )
            uvs.append([theta_index / n_theta, phi_index / n_phi])

    row_width = n_theta + 1
    faces = []
    for phi_index in range(n_phi):
        for theta_index in range(n_theta):
            a = phi_index * row_width + theta_index
            b = a + 1
            c = a + row_width + 1
            d = a + row_width
            if phi_index == 0:
                faces.extend([3, a, c, d])
            elif phi_index == n_phi - 1:
                faces.extend([3, a, b, d])
            else:
                faces.extend([4, a, b, c, d])

    mesh = pv.PolyData(np.asarray(points), np.asarray(faces))
    mesh.active_texture_coordinates = np.asarray(uvs)
    return mesh


def rotate_mesh(mesh: pv.PolyData, axis: str, angle: float) -> None:
    """让 mesh 绕指定坐标轴原地旋转。"""
    rotate = {
        "x": mesh.rotate_x,
        "y": mesh.rotate_y,
        "z": mesh.rotate_z,
    }[axis]
    rotate(angle, point=(0, 0, 0), inplace=True)


def generate_video(
    texture_path: Path,
    output_path: Path,
    axis: str,
    sign: int,
) -> None:
    """为单张纹理生成一种方向的球体旋转视频。"""
    sphere = make_equirect_sphere(radius=1)
    texture = pv.read_texture(str(texture_path))
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    writer = None

    try:
        # 关闭材质光照，避免球面产生明暗、反光或阴影，直接显示纹理颜色。
        plotter.add_mesh(sphere, texture=texture, lighting=False)
        plotter.set_background("black")
        plotter.camera_position = [(0, 0, 5), (0, 0, 0), (0, 1, 0)]
        plotter.enable_anti_aliasing("ssaa")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = FFmpegWriter(output_path, WINDOW_SIZE[0], WINDOW_SIZE[1], FPS)

        # 第 0 帧保留初始姿态，最后一帧恰好达到设定的累计角度。
        angle_step = sign * ROTATION_DEG / (FRAMES - 1)
        for frame_index in range(FRAMES):
            if frame_index > 0:
                rotate_mesh(sphere, axis, angle_step)
            plotter.render()
            frame = plotter.screenshot(return_img=True)
            if frame is None or frame.size == 0:
                raise RuntimeError(f"第 {frame_index + 1} 帧截图失败")
            writer.append_data(frame[:, :, :3])
    finally:
        if writer is not None:
            writer.close()
        plotter.close()


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
            images = [path for path in images if "_anchor_rot" in path.stem]
        if not images:
            print(f"警告：{directory} 中没有适用于 rotation 的 JPG/PNG 图片")
        for image_path in images:
            for direction, (axis, sign) in ROTATIONS.items():
                output_path = (
                    OUTPUT_ROOT / f"rot_{category}_{image_path.stem}_{direction}.mp4"
                )
                jobs.append((image_path, output_path, direction, axis, sign))

    if not jobs:
        raise RuntimeError("没有找到任何待处理的 JPG 图片")

    print(f"共发现 {len(jobs)} 个旋转视频任务；全部输出到 {OUTPUT_ROOT}")
    generated = 0
    skipped = 0
    failures = []
    for job_index, (image_path, output_path, direction, axis, sign) in enumerate(
        jobs, start=1
    ):
        if output_path.is_file() and output_path.stat().st_size > 0 and not args.overwrite:
            skipped += 1
            print(f"[{job_index}/{len(jobs)}] 跳过已有：{output_path}")
            continue

        print(
            f"[{job_index}/{len(jobs)}] 生成 {image_path.parent.name}/{image_path.name} "
            f"-> {direction}（{axis.upper()} 轴）"
        )
        try:
            generate_video(image_path, output_path, axis, sign)
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
