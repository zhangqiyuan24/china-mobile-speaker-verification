#!/bin/bash
# run_tse_pipeline.sh — TSE 线自驱动: 建条件表 -> 冒烟 -> 全量训练 (8.26)
# 约束: GPU2(与外部进程共存), 只写 AI/8.26
set -eu
HERE=/data/home/ligangrui/china_mobile/AI/8.26
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
cd "$HERE"

echo "[pipe] $(date '+%F %T') 1/3 每说话人条件嵌入表(soup5)"
$PY build_spk_embs.py --config train_tse.toml --gpu 2 > build_spk_embs.log 2>&1
grep -q "\[done\]" build_spk_embs.log || { tail -5 build_spk_embs.log; exit 1; }
tail -1 build_spk_embs.log

echo "[pipe] $(date '+%F %T') 2/3 冒烟(32 样本 x 1 ep)"
$PY train_tse.py --config train_tse.toml --gpu 2 --max-train 32 --epochs 1 --exp-tag smoke > tse_smoke.log 2>&1
grep -q "\[done\]" tse_smoke.log || { tail -8 tse_smoke.log; exit 1; }
grep -a "\[tse\]\|\[ep1\]" tse_smoke.log | head -5

echo "[pipe] $(date '+%F %T') 3/3 全量训练(30k x 3ep)"
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 nohup $PY train_tse.py --config train_tse.toml --gpu 2 > tse_train.log 2>&1 &
echo "[pipe] TSE_PID=$! — 日志 tse_train.log"
