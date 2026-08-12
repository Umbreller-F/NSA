from manim import *

config.media_dir = "/data/dataset/TempCompass/case_analysis/NSA-videos"
config.output_file = "demo_001"

class RotatingTexturedSphere(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=90 * DEGREES, theta=0 * DEGREES)
        
        sphere = Sphere(
            center=ORIGIN,
            radius=2,
            resolution=(40, 40),
        )
        
        # 白球黑线
        sphere.set_fill(WHITE, opacity=1.0)
        sphere.set_stroke(BLACK, opacity=0.8, width=0.8)
        
        self.add(sphere)
        self.play(
            Rotate(sphere, angle=0.5*PI, axis=OUT, run_time=8, rate_func=linear),
        )
        self.wait(1)

# manim -ql example_generate_2.py RotatingTexturedSphere
# manim -qm /data/dataset/TempCompass/non-semantic-anchor/example_generate_1.py RotatingTexturedSphere

# | 参数    | 分辨率       | 帧率    | 用途   |
# | ----- | --------- | ----- | ---- |
# | `-ql` | 854×480   | 15fps | 快速预览 |
# | `-qm` | 1280×720  | 30fps | 平衡质量 |
# | `-qh` | 1920×1080 | 60fps | 高清输出 |
# | `-qp` | 3840×2160 | 60fps | 4K   |
