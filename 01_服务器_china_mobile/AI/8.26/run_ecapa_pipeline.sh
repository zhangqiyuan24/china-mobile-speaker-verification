#!/bin/bash
# run_ecapa_pipeline.sh — ECAPA 线自驱动: 原型表 -> 冒烟 -> 全量训练 (8.26)
# 用法: bash run_ecapa_pipeline.sh <GPU_ID>   (默认 1, soup5 评测完的卡)
set -eu
GPU=${1:-1}
HERE=/data/home/ligangrui/china_mobile/AI/8.26
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
cd "$HERE"

echo "[pipe] $(date '+%F %T') 1/3 ECAPA 原型表(192 维)"
$PY build_ecapa_protos.py --config config_ecapa_robust.toml --gpu "$GPU" > build_ecapa_protos.log 2>&1
grep -q "\[done\]" build_ecapa_protos.log || { tail -5 build_ecapa_protos.log; exit 1; }
tail -1 build_ecapa_protos.log

echo "[pipe] $(date '+%F %T') 2/3 冒烟(64 样本 x 1 ep)"
$PY train_ecapa_robust.py --config config_ecapa_robust.toml --gpu "$GPU" \
    --max-train 64 --epochs 1 --exp-tag smoke > ecapa_smoke.log 2>&1 || { tail -8 ecapa_smoke.log; exit 1; }
grep -a "\[ecapa\]\|\[val\]" ecapa_smoke.log | head -4

echo "[pipe] $(date '+%F %T') 3/3 全量训练(90k x 3cond x 8ep)"
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 nohup $PY train_ecapa_robust.py \
    --config config_ecapa_robust.toml --gpu "$GPU" --batch-size 30 --accum 3 \
    > ecapa_train.log 2>&1 &
echo "[pipe] ECAPA_PID=$! — 日志 ecapa_train.log"
