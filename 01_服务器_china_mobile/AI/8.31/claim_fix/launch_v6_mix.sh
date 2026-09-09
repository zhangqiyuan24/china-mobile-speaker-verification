#!/bin/bash
# launch_v6_mix.sh — #38 决策阶梯第二格：A+B 混叠 + 非对称臂点火
# 用户 8.31 方案（训练混入 A+B，注册=A，约束提取 A）+ 深夜授权：大幅未过线自动接力不请示
# 用法: launch_v6_mix.sh [IMP_PROB] [MIX_PROB] [GPU]
#   IMP 默认 0.30（ep2 读数：单位 claim 收益的 oracle 代价最优剂量；剂量曲线
#       若显示 15/20 更优，由巡检传入替换）
#   MIX 默认 0.40；GPU 缺省选当前显存占用最低卡
set -e
AI=/data/home/ligangrui/china_mobile/AI
FIX=$AI/8.31/claim_fix
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
BASE=$AI/8.27/exp/tse_v3_ns05duraug_r5/checkpoint_best.tar
IMP=${1:-0.30}
MIX=${2:-0.40}
TAG=mix$($PY -c "print(int(round($MIX*100)))")i$($PY -c "print(int(round($IMP*100)))")
if [ -n "$3" ]; then GPU=$3; else
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
fi
if pgrep -u ligangrui -f -- "--exp-tag $TAG " > /dev/null; then
  echo "[skip] $TAG 已在跑"; exit 0
fi
[ -f "$BASE" ] || { echo "[err] base ckpt 不存在: $BASE"; exit 1; }
echo "[$(date '+%m-%d %H:%M')] v6点火 tag=$TAG imp=$IMP mix=$MIX GPU=$GPU" >> $FIX/state.txt
cd $AI/8.27
nohup $PY train_tse_v6_mix.py --imp-prob $IMP --mix-prob $MIX \
  --lambda-ns 0.5 --base-ckpt $BASE --exp-tag $TAG --epochs 3 --gpu $GPU --dur-aug \
  > $FIX/$TAG.log 2>&1 &
echo "PID=$! log=$FIX/$TAG.log"
# 训练后官方链由巡检 cron 用: EXP_PREFIX=tse_v6 bash $FIX/run_claim_eval_arm.sh $TAG GPU
