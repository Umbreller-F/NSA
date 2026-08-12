import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ========== 配置 ==========
RES = 120              # 细分度（120 足够光滑无棱角）
FRAMES = 120
FPS = 30
OUTPUT = "/data/dataset/TempCompass/case_analysis/NSA-videos/demo_002.mp4"

PATTERN = "spiral"    # stripes(竖条纹) | spots(斑点) | spiral(螺旋) | wave(波浪) | checker(棋盘)

# 涂装颜色
COLOR_BASE = np.array([1.0, 1.0, 1.0, 1.0])    # 底色：白
COLOR_PAINT = np.array([0.0, 0.0, 0.0, 1.0])   # 花纹：黑

# ========== 生成球面 UV ==========
n, m = RES + 1, RES + 1
u = np.linspace(0, 2*np.pi, n)
v = np.linspace(0, np.pi, m)
u, v = np.meshgrid(u, v)

# ========== 花纹定义（在球面 UV 坐标上绘制） ==========
if PATTERN == "stripes":
    val = np.sin(10 * u)                          # 竖条纹（沿经线）
elif PATTERN == "spots":
    val = np.sin(8*u) * np.sin(8*v)               # 斑点
elif PATTERN == "spiral":
    val = np.sin(12 * (u + 0.3*v))                # 螺旋
elif PATTERN == "wave":
    val = np.sin(6*u) * np.cos(5*v)               # 波浪
elif PATTERN == "checker":
    val = np.sign(np.sin(10*u) * np.sin(10*v))    # 棋盘涂装
else:
    val = np.zeros_like(u)

# 二值化：硬边涂装感
mask = (val > 0).astype(float)[:-1, :-1]

# 构建 RGBA 颜色数组 (RES, RES, 4)
colors = np.zeros((RES, RES, 4))
colors[mask > 0.5] = COLOR_PAINT
colors[mask <= 0.5] = COLOR_BASE

# 球面基础坐标（静态，旋转通过矩阵实现）
theta = np.linspace(0, 2*np.pi, n)
phi = np.linspace(0, np.pi, m)
theta, phi = np.meshgrid(theta, phi)

x_base = np.sin(phi) * np.cos(theta)
y_base = np.sin(phi) * np.sin(theta)
z_base = np.cos(phi)

# ========== 渲染 ==========
fig = plt.figure(figsize=(6, 6), facecolor='white')
ax = fig.add_subplot(111, projection='3d', facecolor='white')

def animate(frame):
    ax.clear()
    ax.set_facecolor('white')
    ax.axis('off')
    
    # 绕 Z 轴旋转（Z 轴 = 屏幕竖直方向）
    angle = frame * (2 * np.pi / FRAMES)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    x = x_base * cos_a - y_base * sin_a
    y = x_base * sin_a + y_base * cos_a
    z = z_base
    
    # 绘制：facecolors 直接覆盖涂装，无网格线，完全不透明
    ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1,
                    shade=False, antialiased=True, linewidth=0)
    
    # 正面平视：相机在 X 轴正方向看向原点
    # elev=0（水平），azim=0（正对球心）
    # 此时 Z 轴竖直在屏幕中，球不倾斜
    ax.view_init(elev=0, azim=0)
    
    # 等比例约束，确保球是正圆的
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass

ani = FuncAnimation(fig, animate, frames=FRAMES, interval=1000//FPS)
writer = FFMpegWriter(fps=FPS, bitrate=5000)
ani.save(OUTPUT, writer=writer)
plt.close()
print(f"已保存: {OUTPUT}")