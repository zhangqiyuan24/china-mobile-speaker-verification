#!/bin/bash
# run_moe_pipeline.sh — MoE-lite 线④管线（8.26）
# 用法: bash run_moe_pipeline.sh [GPU_ID]   (默认 3, 等 gate 判决释放后启动)
set -e
HERE=/data/home/ligangrui/china_mobile/AI/8.26
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
GPU=${1:-3}

cd "$HERE"
echo "[moe-pipeline] 冒烟 (64 行 x 1 ep)"
$PY train_moe.py --config config_moe.toml --gpu "$GPU" \
    --max-train 64 --epochs 1 --batch-size 15 --accum 6 --amp --exp-tag smoke \
    > moe_smoke.log 2>&1 || { echo "[moe-pipeline] 冒烟失败"; tail -30 moe_smoke.log; exit 1; }
tail -5 moe_smoke.log

echo "[moe-pipeline] 全量 (b15 accum6 x 8 ep)"
$PY train_moe.py --config config_moe.toml --gpu "$GPU" \
    --batch-size 15 --accum 6 --amp > moe_train.log 2>&1
echo "[moe-pipeline] 完成"
