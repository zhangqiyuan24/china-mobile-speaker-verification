#!/bin/bash
# report_all_21.sh — #21 主线组分项报告（任务书第4条，纯 CPU，8.28）
# 流程：raw_clean 先跑取 EER 工作点作固定阈值 -> 全组带 --ref-thr 复用。
# 校验锚：raw_noise_0db 应复算出 EER 7.6139 / minDCF 0.5942。
set -e
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
S=$AI/8.27/exp/official_ns05ext2/eval/combo/scores
OUT=$AI/8.27/exp/analysis_21

$PY $AI/8.27/report_metrics.py --score $S/raw_clean/trials_cross_device.score \
    --name raw_clean --out-root $OUT
THR=$($PY -c "import json;print(json.load(open('$OUT/raw_clean/metrics.json'))['EER_thr'])")
echo "[ref] clean EER 工作点阈值 = $THR"
for n in raw_noise_0db raw_noise_m5db frontend_noise_0db frontend_noise_m5db; do
  $PY $AI/8.27/report_metrics.py --score $S/$n/trials_cross_device.score \
      --name $n --out-root $OUT --ref-thr $THR
done
echo "[all done] $OUT"
