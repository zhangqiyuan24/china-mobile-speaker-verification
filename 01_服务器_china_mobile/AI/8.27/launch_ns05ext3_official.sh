#!/bin/bash
# launch_ns05ext3_official.sh — 守望器 7（8.28 凌晨布防，ext3 加码链）
# 等 ns05ext3 训练（ep10-12）全部退出后，GPU3 接力：
#   enhance_tse（ext3 收官 best ckpt）→ evaluate_combo FE（soup5，缓存复用）
# 触发依据：ep11 hard 4.023 < ext2 ep9 的 4.381（预声明成立）。
while pgrep -u ligangrui -f 'exp-tag [n]s05ext3' > /dev/null; do sleep 180; done
sleep 60

PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
SRC=$AI/8.27/exp/official_ns05ext_best/eval/combo
DST=$AI/8.27/exp/official_ns05ext3/eval/combo
mkdir -p $DST
cp -r $SRC/raw $DST/ 2>/dev/null
cp -r $SRC/common $DST/ 2>/dev/null

$PY $AI/8.26/enhance_tse.py \
  --checkpoint $AI/8.27/exp/tse_v3_ns05ext3/checkpoint_best.tar \
  --gpu 3 --batch-size 8 \
  --out-root $AI/8.27/exp/tse_v3_ns05ext3/enhanced_official \
  > $AI/8.27/enhance_ns05ext3_official.log 2>&1 && \
$PY $AI/8.21/evaluate_combo.py \
  --config $AI/8.27/eval_ns05ext3.toml \
  --checkpoint $AI/8.24/exp/soup5/checkpoints/checkpoint_soup5.tar \
  --gpu 3 --batch-size 32 \
  --enh-0db $AI/8.27/exp/tse_v3_ns05ext3/enhanced_official/noise_0db \
  --enh-m5db $AI/8.27/exp/tse_v3_ns05ext3/enhanced_official/noise_m5db \
  > $AI/8.27/official_ns05ext3.log 2>&1
