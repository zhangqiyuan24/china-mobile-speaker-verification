#!/bin/bash
# wait_ecapa_eval.sh — ECAPA 训练退出后自动接官方评测(GPU1)
set -u
HERE=/data/home/ligangrui/china_mobile/AI/8.26
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
echo "[wait] $(date "+%F %T") 等 ECAPA 训练退出..."
while pgrep -f "train_ecapa_robust[.]py" > /dev/null; do sleep 300; done
sleep 60
echo "[wait] $(date "+%F %T") 训练退出, 启动官方评测"
cd "$HERE" && $PY evaluate_ecapa.py --config config_ecapa_robust.toml --gpu 1 \
    > ecapa_eval.log 2>&1
echo "[wait] $(date "+%F %T") 评测完毕(见 ecapa_eval.log)"
