#!/bin/bash
# night_plan.sh — 夜间自动交接（8.24, 2026-08-24 晚挂机）
# 流程:
#   1. 等 proto_robust_90k 训练(GPU3)结束
#   2. 用 8.21 evaluate_combo 正式评测其 checkpoint_best
#   3. 在 GPU3 上启动 born-again 蒸馏(teacher = soup_821_90k 单模型, 输入同 fbank)
#      —— wavlm teacher 若明早达标, 再开第二路蒸馏(GPU2)
# 约束: 只写 AI/8.24; 不动他人进程; GPU0 始终留空
set -u
cd "$(dirname "$0")"
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
LOG_PREFIX="[night]"

echo "$LOG_PREFIX $(date '+%F %T') 值守启动: 等 proto_robust_90k 结束"

# ---- 1. 等训练进程消失(匹配 config_proto_90k 的 python 进程) ----
while pgrep -f "train_eres2net_robust_proto.py --config config_proto_90k" > /dev/null; do
    sleep 120
done
echo "$LOG_PREFIX $(date '+%F %T') proto_robust_90k 训练进程已结束"

BEST=exp/proto_robust_90k/checkpoints/checkpoint_best.tar
if [ ! -f "$BEST" ]; then
    echo "$LOG_PREFIX [ERROR] $BEST 不存在, 评测与蒸馏均不启动, 等人工处理"
    exit 1
fi

# ---- 2. 正式 open-set 评测(GPU3) ----
echo "$LOG_PREFIX $(date '+%F %T') 开始 evaluate_combo (proto_robust_90k best)"
$PY ../8.21/evaluate_combo.py --config config_proto_90k.toml \
    --checkpoint "$BEST" --gpu 3 \
    --enh-0db ../8.18/exp/spk_preserve/eval_e10/denoised/noise_0db \
    --enh-m5db ../8.18/exp/spk_preserve/eval_e10/denoised/noise_m5db \
    > eval_proto_90k.log 2>&1
echo "$LOG_PREFIX $(date '+%F %T') 评测完成(退出码 $?), 结果: exp/proto_robust_90k/eval/combo/combo_summary.json"

# ---- 3. born-again 蒸馏上线(GPU3, teacher=soup 吃同款 fbank, 双卡同物理卡) ----
TEACHER=exp/soup_821_90k/checkpoints/checkpoint_soup_w50.tar
if [ ! -f "$TEACHER" ]; then
    echo "$LOG_PREFIX [ERROR] teacher $TEACHER 不存在, 蒸馏不启动"
    exit 1
fi
echo "$LOG_PREFIX $(date '+%F %T') 启动 born-again 蒸馏 (GPU3, teacher=soup)"
nohup $PY train_eres2net_distill.py --config config_distill.toml \
    --gpu 3 --teacher-gpu 3 \
    --teacher-type eres2net --teacher-ckpt "$TEACHER" \
    --batch-size 24 --accum 4 --epochs 4 \
    > distill.log 2>&1 &
echo "$LOG_PREFIX $(date '+%F %T') 蒸馏 PID=$! ; 值守结束"
