#!/bin/bash
# ============================================================
# E2-proto 正式训练（8 epoch）+ 正式 open-set 评测（含前端组合 ablation）
#
# 用法：
#   bash run_train.sh                       # 默认 GPU 3 共享卡（batch12×accum8+AMP）
#   GPU=1 SHARED=0 bash run_train.sh        # 独占卡：batch 96（config 默认）
#   EPOCHS=10 bash run_train.sh             # 覆盖 epoch 数（方案允许 5~10）
#   SKIP_TRAIN=1 bash run_train.sh          # 只跑评测
# 前端组合评测（可选，需先跑 enhance_testset.sh）：
#   ENH_0DB=exp/enhanced/noise_0db ENH_M5DB=exp/enhanced/noise_m5db bash run_train.sh
# ============================================================
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CFG="${CFG:-config_proto.toml}"
GPU="${GPU:-3}"
SHARED="${SHARED:-1}"              # 1=共享卡配方（batch12 accum8 amp）
EPOCHS="${EPOCHS:-}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
NPZ="exp/prototypes/prototypes.npz"
BEST="exp/proto_robust/checkpoints/checkpoint_best.tar"
MANIFEST_820="../8.20/exp/robust/data/manifest.json"

[ -f "$MANIFEST_820" ] || { echo "[ERROR] 缺 $MANIFEST_820（先在 8.20 跑 prepare）"; exit 1; }
[ -f "$NPZ" ] || { echo "==== Step 0: 生成 prototypes ===="; \
    python prepare_prototypes.py --config "$CFG" --gpu "$GPU"; }

if [ "$SKIP_TRAIN" != "1" ]; then
    echo "==== Step 1: 正式训练（prototype init，8 epoch） ===="
    EXTRA=""
    [ -n "$EPOCHS" ] && EXTRA="$EXTRA --epochs $EPOCHS"
    if [ "$SHARED" = "1" ]; then
        PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
        python train_eres2net_robust_proto.py --config "$CFG" --gpu "$GPU" \
            --amp --batch-size 12 --accum 8 --num-workers 4 --val-batch-size 8 $EXTRA
    else
        python train_eres2net_robust_proto.py --config "$CFG" --gpu "$GPU" $EXTRA
    fi
fi

[ -f "$BEST" ] || { echo "[ERROR] 没有 $BEST"; exit 1; }

echo "==== Step 2: 正式 open-set 评测（同一 checkpoint 重提两侧，官方打分） ===="
ENH_FLAGS=""
if [ -n "${ENH_0DB:-}" ] && [ -n "${ENH_M5DB:-}" ]; then
    ENH_FLAGS="--enh-0db $ENH_0DB --enh-m5db $ENH_M5DB"
    echo "前端组合 ablation 开启：$ENH_FLAGS"
else
    echo "（未设 ENH_0DB/ENH_M5DB，只评 raw；组合评测先跑 enhance_testset.sh）"
fi
python evaluate_combo.py --config "$CFG" --checkpoint "$BEST" --gpu "$GPU" $ENH_FLAGS

echo ""
echo "==== 完成 ===="
echo "结果：exp/proto_robust/eval/combo/{scores,combo_summary.json}"
echo "验收：0dB EER ≤ 5.874 且 -5dB EER ≤ 7.739（相对官方基线 -30%）"
echo "     clean 退化 <1pp 以 combo 汇总里的 open-set clean EER 为准（非验证集）"
