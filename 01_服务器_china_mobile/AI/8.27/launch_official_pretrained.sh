#!/bin/bash
# launch_official_pretrained.sh — 守望器 4（8.27 傍晚布防，#18 口径闭合）
# 等 nsspeech 训练退出后，GPU1 跑：原始 ERes2Net 官方 ckpt × ns05ext ep6 best 前端。
# 预声明：raw 复现官方基线（±0.05）→ FE 双档按 -30% 公式判定；不复现则 UNVERIFIED 待分析。
# 注：nsspeech 官方链（若 ep3 hard<6.64 触发）走 GPU3（ns05ext2 之后），与本链不冲突。
while pgrep -u ligangrui -f 'exp-tag [n]sspeech' > /dev/null; do sleep 120; done
sleep 60

PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI

$PY $AI/8.27/eval_official_pretrained.py \
  --config $AI/8.27/eval_pretrained_eres2net.toml \
  --gpu 1 --batch-size 32 \
  --enh-0db $AI/8.27/exp/tse_v3_ns05ext/enhanced_official_best/noise_0db \
  --enh-m5db $AI/8.27/exp/tse_v3_ns05ext/enhanced_official_best/noise_m5db \
  > $AI/8.27/official_pretrained_eres2net.log 2>&1
