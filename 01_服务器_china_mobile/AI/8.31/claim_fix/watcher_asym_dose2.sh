#!/bin/bash
# watcher_asym_dose2.sh — #38 第二批中间剂量臂（8.31 深夜自动接力，无需人工）
# 依据 ep2 读数：10% 安全但无效（claim 4.125 不动）、30% 有效但 oracle 红线
# （+0.72>0.5）→ 补 15%/20% 两点找双绿剂量；反拉两臂净负，不再加。
# 双门槛不变：① oracle hard avg ≤3.073 绿（>3.273 红）；② claim eer 显著低于 4.125
AI=/data/home/ligangrui/china_mobile/AI
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
FIX=$AI/8.31/claim_fix
BASE=$AI/8.27/exp/tse_v3_ns05duraug_r5/checkpoint_best.tar
cd $AI/8.27

# GPU0：等 asym10 退出 -> asym20（20%，无反拉）
nohup bash -c "
  while pgrep -u ligangrui -f -- '--exp-tag asym10 --' > /dev/null; do sleep 120; done
  sleep 60
  $PY train_tse_v5_asym.py --imp-prob 0.20 --lambda-ns 0.5 --base-ckpt $BASE \
    --exp-tag asym20 --epochs 3 --gpu 0 --dur-aug > $FIX/asym20.log 2>&1
" > /dev/null 2>&1 &

# GPU1：等 asym30（B 臂，勿匹配 asym30ap）退出 -> asym15（15%，无反拉）
nohup bash -c "
  while pgrep -u ligangrui -f -- '--exp-tag asym30 --' > /dev/null; do sleep 120; done
  sleep 60
  $PY train_tse_v5_asym.py --imp-prob 0.15 --lambda-ns 0.5 --base-ckpt $BASE \
    --exp-tag asym15 --epochs 3 --gpu 1 --dur-aug > $FIX/asym15.log 2>&1
" > /dev/null 2>&1 &

echo "[$(date '+%m-%d %H:%M')] dose2 守望器已布：GPU0->asym20（等asym10）、GPU1->asym15（等asym30）" >> $FIX/state.txt
echo "watchers armed"
