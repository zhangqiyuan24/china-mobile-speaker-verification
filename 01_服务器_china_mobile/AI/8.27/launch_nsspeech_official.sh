#!/bin/bash
# launch_nsspeech_official.sh — 守望器 5（8.27 傍晚布防）
# 等口径闭合评测（official_pretrained 链）退出后，GPU1 接力跑 nsspeech 官方链：
#   1) enhance_tse（nsspeech best ckpt × 官方 0db/m5db 测试集）
#   2) evaluate_combo FE 模式（soup5 后端，raw/enrol 缓存自 official_ns05ext_best 复用，
#      守卫按 ckpt 指纹自动校验）
# 判定规则修正见 eval_nsspeech.toml 头注（proxy 域反向，一次性假设检验）。
while pgrep -u ligangrui -f 'official_[p]retrained' > /dev/null; do sleep 120; done
sleep 60

PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
SRC=$AI/8.27/exp/official_ns05ext_best/eval/combo
DST=$AI/8.27/exp/official_nsspeech/eval/combo
mkdir -p $DST
cp -r $SRC/raw $DST/ 2>/dev/null
cp -r $SRC/common $DST/ 2>/dev/null

$PY $AI/8.26/enhance_tse.py \
  --checkpoint $AI/8.27/exp/tse_v3_nsspeech/checkpoint_best.tar \
  --gpu 1 --batch-size 8 \
  --out-root $AI/8.27/exp/tse_v3_nsspeech/enhanced_official \
  > $AI/8.27/enhance_nsspeech_official.log 2>&1 && \
$PY $AI/8.21/evaluate_combo.py \
  --config $AI/8.27/eval_nsspeech.toml \
  --checkpoint $AI/8.24/exp/soup5/checkpoints/checkpoint_soup5.tar \
  --gpu 1 --batch-size 32 \
  --enh-0db $AI/8.27/exp/tse_v3_nsspeech/enhanced_official/noise_0db \
  --enh-m5db $AI/8.27/exp/tse_v3_nsspeech/enhanced_official/noise_m5db \
  > $AI/8.27/official_nsspeech.log 2>&1
