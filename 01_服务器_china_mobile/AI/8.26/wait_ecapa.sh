#!/bin/bash
# wait_ecapa.sh — 等 soup5 评测(GPU1)结束后自动启动 ECAPA 线(8.26 深夜)
# 模式串用括号技巧防 pgrep 自匹配
set -u
HERE=/data/home/ligangrui/china_mobile/AI/8.26
echo "[wait] $(date '+%F %T') 等 soup5 评测退出..."
while pgrep -f "evaluate_comb[o].py" > /dev/null; do sleep 300; done
sleep 60
echo "[wait] $(date '+%F %T') GPU1 已空, 启动 ECAPA 线"
bash "$HERE/run_ecapa_pipeline.sh" 1 > "$HERE/ecapa_pipeline.log" 2>&1
echo "[wait] $(date '+%F %T') ECAPA 管线接管完毕(见 ecapa_pipeline.log)"
