#!/bin/bash
# watcher_official_claim.sh — #38 官方 claim 链自动接力（8.31 深夜，claim 优先口径）
# 用户指示：claim 降即可、oracle 可让价、评测口径切 claim；尽快确认方案可行性
# GPU2：等 asym30ap（占卡）与 asym30（训练终，ckpt 定格）都退 → asym30 官方链
#        （30% 纯剂量 = ep2 读数里单位 claim 收益 oracle 代价最小的臂）
# GPU3：等 asym50ap 退 → asym50ap 官方链（冒认压最狠臂，对照）
# 与 dose2 守望器（GPU0→asym20 / GPU1→asym15）互不冲突。
AI=/data/home/ligangrui/china_mobile/AI
FIX=$AI/8.31/claim_fix
mkdir -p $FIX/official_asym30 $FIX/official_asym50ap

nohup bash -c "
  while pgrep -u ligangrui -f -- '--exp-tag asym30ap --' > /dev/null; do sleep 120; done
  while pgrep -u ligangrui -f -- '--exp-tag asym30 --'  > /dev/null; do sleep 120; done
  sleep 60
  bash $FIX/run_claim_eval_arm.sh asym30 2 >> $FIX/official_asym30/watch.log 2>&1
" > /dev/null 2>&1 &

nohup bash -c "
  while pgrep -u ligangrui -f -- '--exp-tag asym50ap --' > /dev/null; do sleep 120; done
  sleep 60
  bash $FIX/run_claim_eval_arm.sh asym50ap 3 >> $FIX/official_asym50ap/watch.log 2>&1
" > /dev/null 2>&1 &

echo "[$(date '+%m-%d %H:%M')] 官方claim链守望器已布：GPU2->asym30、GPU3->asym50ap" >> $FIX/state.txt
echo "armed"
