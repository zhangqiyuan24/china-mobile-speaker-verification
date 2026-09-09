#!/bin/bash
# launch_ns05ext2_official.sh — 守望器 6（8.27 晚布防，#22 加码链）
# 等 ns05ext2 训练（ep7-9，含 DataLoader workers）全部退出后，GPU3 接力：
#   1) enhance_tse（ns05ext2 收官 best ckpt × 官方 0db/m5db 测试集）
#   2) evaluate_combo FE 模式（soup5 后端，raw/enrol 缓存自 official_ns05ext_best 复用）
# 触发依据：ep7 hard 4.795 < ep6 4.989（预声明"破 4.989 → 官方加码"成立）。
while pgrep -u ligangrui -f 'exp-tag [n]s05ext2' > /dev/null; do sleep 180; done
sleep 60

PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
SRC=$AI/8.27/exp/official_ns05ext_best/eval/combo
DST=$AI/8.27/exp/official_ns05ext2/eval/combo
mkdir -p $DST
cp -r $SRC/raw $DST/ 2>/dev/null
cp -r $SRC/common $DST/ 2>/dev/null

$PY $AI/8.26/enhance_tse.py \
  --checkpoint $AI/8.27/exp/tse_v3_ns05ext2/checkpoint_best.tar \
  --gpu 3 --batch-size 8 \
  --out-root $AI/8.27/exp/tse_v3_ns05ext2/enhanced_official \
  > $AI/8.27/enhance_ns05ext2_official.log 2>&1 && \
$PY $AI/8.21/evaluate_combo.py \
  --config $AI/8.27/eval_ns05ext2.toml \
  --checkpoint $AI/8.24/exp/soup5/checkpoints/checkpoint_soup5.tar \
  --gpu 3 --batch-size 32 \
  --enh-0db $AI/8.27/exp/tse_v3_ns05ext2/enhanced_official/noise_0db \
  --enh-m5db $AI/8.27/exp/tse_v3_ns05ext2/enhanced_official/noise_m5db \
  > $AI/8.27/official_ns05ext2.log 2>&1
