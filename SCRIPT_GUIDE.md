# 实验脚本时间线

以下命令均在项目根目录和 Conda 环境 `NSA` 中运行：

```bash
conda activate NSA
```

## v2：非语义纹理锚点实验

### 添加纹理锚点

添加低语义、高独特性的随机标记，patch 边长为图片短边的 8%：

```bash
python generate/augment_nsa_anchor.py --patch-ratio 0.08
```

不做颜色匹配：

```bash
python generate/augment_nsa_anchor.py --no-color-match
```

只根据 translation 错误筛选，并只生成 trans 布局：

```bash
python generate/augment_nsa_anchor.py --task trans --layout trans
```

### 渲染视频

```bash
python generate/rotation_v2.py --categories NSA_ANCHOR
python generate/translation_v2.py --categories NSA_ANCHOR
```

### 模型测试

```bash
python eval/run_gpt_v2.py \
  --videos-root videos/NSA_anchor

python eval/run_gemini_v2.py \
  --videos-root videos/NSA_anchor
```

重试错误样例：

```bash
python eval/run_gpt_v2.py \
  --videos-root videos/NSA_anchor \
  --retry-errors \
  --confirm-api-call

python eval/run_gemini_v2.py \
  --videos-root videos/NSA_anchor \
  --retry-errors \
  --confirm-api-call
```

### 查看测试结果

```bash
python visualization/case_viewer.py \
  --jsonl gemini-3.5-flash-v2.jsonl \
  --port 34800

python visualization/case_viewer.py \
  --jsonl gpt-5.6-sol-v2.jsonl \
  --port 34801

python visualization/case_viewer.py \
  --jsonl gpt-5.6-sol-nsa-anchor-error-retry-v2.jsonl \
  --port 34801
```

## v3：纹理处理与锚点生成（当前进度）

### 1. 纹理预处理

新增 `generate/process_v3_textures.py`，处理 `textures/v3` 下名称以 `raw` 结尾的目录：

- 非正方形图片按短边从中心裁剪为正方形。
- PNG 转换为 JPG，透明背景填充为白色。
- JPG 默认以质量 95 保存。

输入输出示例：

| 输入 | 输出 |
| --- | --- |
| `textures/v3/NSA_raw/000.jpg` | `textures/v3/NSA/000.jpg` |
| `textures/v3/SA_raw/019.png` | `textures/v3/SA/019.jpg` |

运行：

```bash
python generate/process_v3_textures.py
```

调整 JPG 质量：

```bash
python generate/process_v3_textures.py --quality 90
```

### 2. 生成 NSA 纹理锚点

`generate/augment_nsa_anchor_v3.py` 为全部 NSA 纹理添加局部锚点。锚点在安全可视区域内随机放置，位置多样且能在视频中出现；默认随机种子为 `42`，结果可复现。

默认输入：`textures/v3/NSA` 中的全部纹理。

默认输出：`textures/v3/NSA_anchor`

- `*_anchor_trans.png`：锚点随机放在中心可视区域，用于平移视频。
- `*_anchor_rot.png`：锚点随机放在左右 UV 接缝附近的安全区域，用于球面旋转视频。
- `manifest.jsonl`：记录每张输出纹理的来源和参数。

运行：

```bash
python generate/augment_nsa_anchor_v3.py
```

重新生成并覆盖已有结果：

```bash
python generate/augment_nsa_anchor_v3.py --overwrite
```

### 3. 生成三类视频

输入为 `textures/v3/NSA`、`textures/v3/SA` 和 `textures/v3/NSA_anchor`，视频统一保存到 `videos/v3`。

```bash
python generate/translation_v3.py
python generate/rotation_v3.py
```

默认生成 NSA、SA、NSA_ANCHOR 三类视频；rotation 视频关闭光照，按纹理原始颜色显示。已有视频会跳过，需要覆盖时添加 `--overwrite`。

### 4. v3 模型评测

评测 `videos/v3` 下的三类视频。模型使用英文进行开放式分析和回答，并将简洁结论写入 `<final_answer>` 标签：

```bash
python eval/run_gpt_v3.py --dry-run
python eval/run_gemini_v3.py --dry-run
```

添加 `--test` 时会从 NSA、SA、NSA_ANCHOR 各选择第一个视频，共测试 3 个。正式运行时去掉 `--dry-run`；结果默认保存到 `results/*-v3.jsonl`。

Gemini 上传前会检查视频大小。超过 80 MiB 的视频会临时转码（最长边缩放至 1600 像素，H.264 CRF 23压缩）后上传，原视频不变；可先用 `--dry-run` 查看哪些视频会被转码。
