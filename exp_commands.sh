# texture图片添加低语义、高独特性的随机标记
# patch 边长为短边的 8%
python generate/augment_nsa_anchor.py --patch-ratio 0.08

# 不做颜色匹配
python generate/augment_nsa_anchor.py --no-color-match

# 只根据 translation 错误筛选，只生成 trans 布局
python generate/augment_nsa_anchor.py --task trans --layout trans

# 渲染视频
python generate/rotation_v2.py --categories NSA_ANCHOR
python generate/translation_v2.py --categories NSA_ANCHOR

# 测试
python eval/run_gpt_v2.py \
  --videos-root videos/NSA_anchor

python eval/run_gemini_v2.py \
  --videos-root videos/NSA_anchor

# 重试错误样例
python eval/run_gpt_v2.py \
  --videos-root videos/NSA_anchor \
  --retry-errors \
  --confirm-api-call

python eval/run_gemini_v2.py \
  --videos-root videos/NSA_anchor \
  --retry-errors \
  --confirm-api-call

# case viewer
python visualization/case_viewer.py \
  --jsonl gemini-3.5-flash-v2-test.jsonl \
  --port 34800