#!/bin/bash
# ============================================================
# E2 正式训练 + 正式评测 + AS-Norm（可选）
#
# 用法：
#   bash run_train.sh                      # 单卡，默认 GPU 1
#   GPUS=1,3 bash run_train.sh             # 双卡 DDP（GPU 1,3，同 8.19）
#   EPOCHS=10 LR=1e-5 bash run_train.sh    # 覆盖默认超参
# 可选：
#   EVAL_PRETRAINED=1   附带 pretrained 基线评测（验证管线可复现官方数字）
#   SKIP_TRAIN=1        只跑评测（训练已完成时）
#   RUN_ASNORM=1        训练评测后接着跑 AS-Norm 全流程
# ============================================================
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CFG="${CFG:-config_robust.toml}"
GPU="${GPU:-1}"
GPUS="${GPUS:-}"                 # 非空则 DDP，如 "1,3"
EPOCHS="${EPOCHS:-}"
LR="${LR:-}"
MANIFEST="exp/robust/data/manifest.json"
BEST="exp/robust/checkpoints/checkpoint_best.tar"

if [ ! -f "$MANIFEST" ]; then
    echo "==== Step 0: 数据准备 ===="
    python prepare_multicond_data.py --config "$CFG"
fi

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    echo "==== Step 1: 正式训练 ===="
    EXTRA=""
    [ -n "$EPOCHS" ] && EXTRA="$EXTRA --epochs $EPOCHS"
    [ -n "$LR" ] && EXTRA="$EXTRA --lr $LR"
    if [ -n "$GPUS" ]; then
        echo "DDP: GPUs=$GPUS"
        CUDA_VISIBLE_DEVICES="$GPUS" python -m torch.distributed.run --standalone \
            --nproc_per_node=2 train_eres2net_robust.py --config "$CFG" $EXTRA
    else
        python train_eres2net_robust.py --config "$CFG" --gpu "$GPU" $EXTRA
    fi
fi

if [ ! -f "$BEST" ]; then
    echo "[ERROR] 没有 $BEST（训练未产生满足门限的 checkpoint？）"
    exit 1
fi

echo "==== Step 2: 正式评测（同一 checkpoint 重提注册/测试两侧） ===="
PRE_FLAG=""
[ "${EVAL_PRETRAINED:-0}" = "1" ] && PRE_FLAG="--also-pretrained"
python evaluate_robust.py --config "$CFG" --checkpoint "$BEST" --gpu "$GPU" $PRE_FLAG

if [ "${RUN_ASNORM:-0}" = "1" ]; then
    echo "==== Step 3: AS-Norm（cohort = 训练说话人，K 在验证集上选） ===="
    python asnorm.py --config "$CFG" --make-cohort --cohort-spks 500 --cohort-utts 2
    python evaluate_robust.py --config "$CFG" --checkpoint "$BEST" --gpu "$GPU" \
        --extract-only --crop-seconds 3 \
        --wav-list exp/robust/data/asnorm/cohort_wavs.txt \
        --out-dir exp/robust/eval/finetuned/embeddings/cohort
    python asnorm.py --config "$CFG" --select-k --gpu "$GPU"
    python asnorm.py --config "$CFG" --apply
    echo "（--apply 的 raw EER 应与 Step 2 官方打分一致，AS EER 为归一化后结果）"
fi

echo ""
echo "==== 完成 ===="
echo "结果：exp/robust/eval/finetuned/scores/*/result.metrics"
echo "验收：0dB EER ≤ 5.874 且 -5dB EER ≤ 7.739（相对官方基线 -30%）"
