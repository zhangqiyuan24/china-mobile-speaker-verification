#!/bin/bash
# 等 best 官方链退出后, 在 GPU3 从 checkpoint_best.tar(=ep6) 续训 ns05ext2 (总 ep7-9)。
# 用户授权(8.27 傍晚): 训练曲线未到头(三连降匀速), 空卡续训与官方评测/汇报线并行,
# 不占官方名额; GPU0 空卡铁律不动。
# 预声明: ep7-9 hard 曲线对照 ns05ext(ep4-6: 6.239/5.619/4.989);
# 收官 best 若破 4.989, 是否花官方名额由 ep6 官方兑现情况+用户决定。
# pgrep 自匹配检查: 模式串 "official_ns05ext_best" 不出现在本脚本路径/
#   续训进程 cmdline(evaluate 进程 --config 含此串, 是有意等待对象) ✓
while pgrep -u ligangrui -f "official_ns05ext_best" > /dev/null; do
  sleep 120
done
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
$PY $AI/8.27/train_tse_v3.py \
  --lambda-ns 0.5 \
  --base-ckpt $AI/8.27/exp/tse_v3_ns05ext/checkpoint_best.tar \
  --exp-tag ns05ext2 --epochs 3 --gpu 3 \
  > $AI/8.27/tse_v3_ns05ext2.log 2>&1
