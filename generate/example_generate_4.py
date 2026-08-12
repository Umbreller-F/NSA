import os
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'
os.environ['VTK_DEFAULT_RENDER_WINDOW_OFF_SCREEN'] = 'true'

# 如果系统有 xvfb，也可以自动启动
# import pyvista as pv
# pv.start_xvfb()  # 需要 pip install pyvista xvfbwrapper

import numpy as np
import pyvista as pv
import imageio

# ========== 配置 ==========
TEXTURE_PATH = "/data/dataset/TempCompass/case_analysis/texture_4.jpg"
OUTPUT = "/data/dataset/TempCompass/case_analysis/NSA-videos/demo_004.mp4"
FRAMES = 150
FPS = 30
ROTATION_DEG = 180  # 整段视频累计自转角度
WINDOW_SIZE = [1440, 1440]

# 创建球体并自动映射 UV
sphere = pv.Sphere(radius=1, theta_resolution=80, phi_resolution=80)
sphere.texture_map_to_sphere(inplace=True)
# 把纹理极点从正对相机转到顶部，避免星burst畸变位于画面中心
sphere.rotate_x(90, point=(0, 0, 0), inplace=True)

# 加载纹理
tex = pv.read_texture(TEXTURE_PATH)

# 离屏渲染器
plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
plotter.add_mesh(sphere, texture=tex, smooth_shading=True)
plotter.set_background('black')
plotter.camera_position = [(0, 0, 5), (0, 0, 0), (0, 1, 0)]
plotter.enable_anti_aliasing('ssaa')  # 超采样抗锯齿，提升清晰度

# 逐帧旋转并保存
images = []
for i in range(FRAMES):
    sphere.rotate_y(ROTATION_DEG / FRAMES, point=(0, 0, 0), inplace=True)  # 绕竖直轴自转
    plotter.render()
    img = plotter.screenshot()
    if img is not None and img.size > 0:
        images.append(img)
    else:
        print(f"警告：第 {i} 帧截图失败")

plotter.close()  # 及时释放资源

if len(images) == 0:
    raise RuntimeError("没有成功渲染任何帧，请检查显示后端配置")

imageio.mimsave(OUTPUT, images, fps=FPS, codec="libx264")
print(f"已保存: {OUTPUT}")