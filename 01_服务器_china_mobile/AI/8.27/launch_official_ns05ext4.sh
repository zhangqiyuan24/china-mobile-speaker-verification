#!/bin/bash
# 等 ns10 (lambda-ns 1.0) 退出后, 在 GPU3 串联 ns05ext_ep4 官方链
while pgrep -u ligangrui -f "lambda-ns 1.0" > /dev/null; do sleep 300; done
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
$PY $AI/8.26/enhance_tse.py \
  --checkpoint $AI/8.27/exp/tse_v3_ns05ext/checkpoint_epoch1.tar \
  --gpu 3 --batch-size 8 \
  --out-root $AI/8.27/exp/tse_v3_ns05ext/enhanced_official && \
$PY $AI/8.21/evaluate_combo.py \
  --config $AI/8.27/eval_ns05ext4.toml \
  --checkpoint $AI/8.24/exp/soup5/checkpoints/checkpoint_soup5.tar \
  --gpu 3 --batch-size 32 \
  --enh-0db $AI/8.27/exp/tse_v3_ns05ext/enhanced_official/noise_0db \
  --enh-m5db $AI/8.27/exp/tse_v3_ns05ext/enhanced_official/noise_m5db
