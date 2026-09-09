#!/bin/bash
# watcher_v8_official.sh PID TAG GPU — 等 v8 训练主进程退出 → 自动官方 claim 评测全链
# v8d 特例：训练日志在 8.27/exp/nohup_v8d.log，触发时 cp 到 $FIX/$TAG.log 供选臂
PID=$1; TAG=$2; GPU=$3
FIX=/data/home/ligangrui/china_mobile/AI/8.31/claim_fix
while kill -0 $PID 2>/dev/null; do sleep 60; done
sleep 90
if [ "$TAG" = "v8d_sil" ]; then
  cp /data/home/ligangrui/china_mobile/AI/8.27/exp/nohup_v8d.log $FIX/$TAG.log
fi
cd $FIX
EXP_PREFIX=tse_v8 bash run_claim_eval_arm.sh $TAG $GPU > official_${TAG}_launch.log 2>&1
