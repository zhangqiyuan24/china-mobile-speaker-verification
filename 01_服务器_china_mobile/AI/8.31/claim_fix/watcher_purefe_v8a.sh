#!/bin/bash
# watcher_purefe_v8a.sh — 守望 v8a（GPU0，PID 见下）训完退出 → 空卡即点火 #43 纯前端场景补测
V8A_PID=1813710
while kill -0 $V8A_PID 2>/dev/null; do sleep 60; done
sleep 30
mkdir -p /data/home/ligangrui/china_mobile/AI/8.28/scenes_purefe
bash /data/home/ligangrui/china_mobile/AI/8.31/claim_fix/scenes_purefe.sh 0 \
    > /data/home/ligangrui/china_mobile/AI/8.28/scenes_purefe_main.log 2>&1
