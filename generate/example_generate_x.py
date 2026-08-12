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
x = 7

TEXTURE_PATH = f"/data/dataset/TempCompass/case_analysis/texture_{x}.jpg"
OUTPUT = f"/data/dataset/TempCompass/case_analysis/NSA-videos/demo_00{x}.mp4"
FRAMES = 150
FPS = 30
ROTATION_DEG = 180  # 整段视频累计自转角度
WINDOW_SIZE = [1440, 1440]

def make_equirect_sphere(radius=1.0, n_theta=160, n_phi=80):
    """手工构建球面网格：接缝处复制一列点，UV 为标准等距柱状映射。

    pv.Sphere + texture_map_to_sphere 会把纹理以镜像方式在两个半球各铺一次，
    这里改为手工网格：经度 j 从 0 到 n_theta（含两端，接缝处 u 精确从 0 到 1），
    极点处每列各有一份重复点，因此接缝和两极都没有涂抹/畸变。
    """
    theta = np.linspace(0, 2 * np.pi, n_theta + 1)  # 含重复接缝列
    phi = np.linspace(0, np.pi, n_phi + 1)          # 0=北极, pi=南极

    points, uvs = [], []
    for i, p in enumerate(phi):
        for j, t in enumerate(theta):
            points.append([radius * np.sin(p) * np.sin(t),
                           radius * np.cos(p),
                           radius * np.sin(p) * np.cos(t)])
            uvs.append([j / n_theta, i / n_phi])

    W = n_theta + 1  # 每行点数
    faces = []
    for i in range(n_phi):
        for j in range(n_theta):
            a = i * W + j
            b = a + 1
            c = a + W + 1
            d = a + W
            if i == 0:
                faces += [3, a, c, d]        # 北极圈用三角形
            elif i == n_phi - 1:
                faces += [3, a, b, d]        # 南极圈用三角形
            else:
                faces += [4, a, b, c, d]
    mesh = pv.PolyData(np.array(points), np.array(faces))
    mesh.active_texture_coordinates = np.array(uvs)
    return mesh


# 创建球体（标准等距柱状 UV，极点在竖直方向）
sphere = make_equirect_sphere(radius=1)

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