#!/bin/bash
# watch_consist.sh — soup_ac 评测退出后自动在 GPU2 启动一致性微调(NAW-SV 平移)
cd /home/ligangrui/china_mobile/AI/8.24
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
while pgrep -f 'evaluate_combo.py --config config_soup_ac' > /dev/null; do
    sleep 30
done
echo "[watch] $(date '+%F %T') soup_ac 评测进程退出, 等 60s 释放显存"
sleep 60
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 nohup $PY train_consist.py     --config config_consist.toml --gpu 2     --init-ckpt exp/soup3_abc/checkpoints/checkpoint_soup3.tar     --batch-utts 9 --epochs 2 --amp     > /home/ligangrui/china_mobile/AI/8.24/consist.log 2>&1 &
echo "[watch] consist PID=$!"
