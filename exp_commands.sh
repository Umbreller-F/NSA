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
  --jsonl gemini-3.5-flash-v2.jsonl \
  --port 34800

python visualization/case_viewer.py \
  --jsonl gpt-5.6-sol-v2.jsonl \
  --port 34801

python visualization/case_viewer.py \
  --jsonl gpt-5.6-sol-nsa-anchor-error-retry-v2.jsonl \
  --port 34801
  
mkdir -p nohup_logs

nohup python generate/translation_v3.py \
  > "nohup_logs/$(date +%Y%m%d_%H%M%S)_translation_v3.log" 2>&1 &

nohup python eval/run_gemini_v3.py \
  > "nohup_logs/$(date +%Y%m%d_%H%M%S)_run_gemini_v3.log" 2>&1 &

nohup python eval/run_gpt_v3.py \
  > "nohup_logs/$(date +%Y%m%d_%H%M%S)_run_gpt_v3.log" 2>&1 &