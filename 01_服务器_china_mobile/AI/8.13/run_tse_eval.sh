#!/bin/bash
# ============================================================
# WeSep TSE + ERes2Net — 全量 overlap 评测 (类似 evaluate_all_finetune.sh)
#
# 跑完所有 overlap 条件:
#   overlap2_cn / overlap2_en / overlap3_cn / overlap3_en
#
# 结果保存到:
#   sv-ecapa/data/result/eres2net_wesep/{condition}/eer.txt
#
# 用法:
#   bash run_tse_eval.sh              # 全量 (慢, 百万 trial)
#   bash run_tse_eval.sh --balanced   # 平衡抽样 (200 trial/条件, 快速验证)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 结果目录 (相对 AI/8.13/ → sv-ecapa/data/result/eres2net_wesep)
RESULT_BASE="../../code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa/data/result/eres2net_wesep"
CONDITIONS="overlap2_cn overlap2_en overlap3_cn overlap3_en"

BALANCED_FLAG=""
if [ "$1" == "--balanced" ]; then
    BALANCED_FLAG="--balanced"
    echo "[模式] 平衡抽样 (200 trial/条件)"
else
    echo "[模式] 全量评测 (百万 trial/条件, 较慢)"
fi

echo "============================================"
echo "  WeSep TSE + ERes2Net overlap 全量评测"
echo "  结果目录: $RESULT_BASE"
echo "============================================"

for COND in $CONDITIONS; do
    echo ""
    echo "============================================"
    echo "  条件: $COND"
    echo "============================================"
    python tse_eres2net_eval.py \
        --condition "$COND" \
        $BALANCED_FLAG \
        --out_dir "$RESULT_BASE/$COND"
done

# ============================================================
# 汇总
# ============================================================
echo ""
echo "============================================"
echo "  评测完成！结果汇总:"
echo "============================================"
for COND in $CONDITIONS; do
    EER_FILE="$RESULT_BASE/$COND/eer.txt"
    if [ -f "$EER_FILE" ]; then
        EER=$(cat "$EER_FILE")
        printf "  %-14s  EER=%s%%\n" "$COND" "$EER"
    else
        printf "  %-14s  (无结果)\n" "$COND"
    fi
done
echo "============================================"
echo ""
echo "对比 baseline (evaluate_overlap.sh 的结果):"
echo "  TSE EER 应明显低于 baseline EER (尤其 overlap2_en / overlap3_en)"
