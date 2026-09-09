#!/bin/bash
# 等 ns05ext 训练(exp-tag ns05ext)退出 + 第一官方链(launch_official_ns05ext4)退出后,
# 在 GPU3 用 checkpoint_best.tar 跑二次官方链。
# 触发依据: ep5 hard 3.500/7.739 avg=5.619 < 6.0（预声明规则）。
# 注意 pgrep 模式: "exp-tag ns05ext" 不匹配本脚本自身(路径为 ns05ext_best);
# "launch_official_ns05ext4" 不匹配本脚本(后缀 _best)。
while pgrep -u ligangrui -f "exp-tag ns05ext" > /dev/null \
   || pgrep -u ligangrui -f "launch_official_ns05ext4" > /dev/null; do
  sleep 300
done
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
# 缓存: raw/common 自 official_ns05ext4 拷入(同后端 soup5, meta 守卫通过);
# frontend 不拷, --enh-* 路径不同强制重提。
DST=$AI/8.27/exp/official_ns05ext_best/eval/combo
mkdir -p $DST
cp -r $AI/8.27/exp/official_ns05ext4/eval/combo/raw $DST/
cp -r $AI/8.27/exp/official_ns05ext4/eval/combo/common $DST/
$PY $AI/8.26/enhance_tse.py \
  --checkpoint $AI/8.27/exp/tse_v3_ns05ext/checkpoint_best.tar \
  --gpu 3 --batch-size 8 \
  --out-root $AI/8.27/exp/tse_v3_ns05ext/enhanced_official_best && \
$PY $AI/8.21/evaluate_combo.py \
  --config $AI/8.27/eval_ns05ext_best.toml \
  --checkpoint $AI/8.24/exp/soup5/checkpoints/checkpoint_soup5.tar \
  --gpu 3 --batch-size 32 \
  --enh-0db $AI/8.27/exp/tse_v3_ns05ext/enhanced_official_best/noise_0db \
  --enh-m5db $AI/8.27/exp/tse_v3_ns05ext/enhanced_official_best/noise_m5db
