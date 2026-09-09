#!/bin/bash
# launch_asym_arms.sh — #38 冒认防御三臂（8.31，用户批准全卡提速）
# 别人进程全无 → GPU3 按规则留空；A=10%(GPU0，r5基线评测后链式起)
# B=30%(GPU1)；C=30%+反拉τ0.1·λ0.3(GPU2)。r5 配方不动（λ_ns0.5+dur-aug）。
# 双门槛：① oracle hard avg ≤ 基线+0.3（红灯线 +0.5 回退 r5）
#        ② proxy claim eer 显著低于基线（基线值见 base_r5_claimval.log）
AI=/data/home/ligangrui/china_mobile/AI
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
FIX=$AI/8.31/claim_fix
BASE=$AI/8.27/exp/tse_v3_ns05duraug_r5/checkpoint_best.tar
mkdir -p $FIX
cd $AI/8.27

# 0) r5 基线 claim-proxy 实测（GPU0）→ 完成后链式起 A 臂
nohup bash -c "
  $PY train_tse_v5_asym.py --imp-prob 0 --eval-only --lambda-ns 0.5 \
      --base-ckpt $BASE --gpu 0 > $FIX/base_r5_claimval.log 2>&1
  $PY train_tse_v5_asym.py --imp-prob 0.10 --lambda-ns 0.5 --base-ckpt $BASE \
      --exp-tag asym10 --epochs 3 --gpu 0 --dur-aug > $FIX/asym10.log 2>&1
" > /dev/null 2>&1 &

# B 臂 30%（GPU1）
nohup $PY train_tse_v5_asym.py --imp-prob 0.30 --lambda-ns 0.5 --base-ckpt $BASE \
    --exp-tag asym30 --epochs 3 --gpu 1 --dur-aug > $FIX/asym30.log 2>&1 &

# C 臂 30% + 反拉（GPU2）
nohup $PY train_tse_v5_asym.py --imp-prob 0.30 --lam-imp 0.3 --tau-imp 0.10 \
    --lambda-ns 0.5 --base-ckpt $BASE \
    --exp-tag asym30ap --epochs 3 --gpu 2 --dur-aug > $FIX/asym30ap.log 2>&1 &

echo "[$(date '+%m-%d %H:%M')] 三臂已发：base评测+A=10%(GPU0) B=30%(GPU1) C=30%+反拉(GPU2)；GPU3留空" >> $FIX/state.txt
echo "launched"
