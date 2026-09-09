#!/bin/bash
# ============================================================
# E2-matched 一条龙：数据(可自动补) -> 语料准备 -> 五条件训练 -> 正式评测
#
# 用法：
#   bash run_train.sh                        # 全流程，默认 90k（config_matched_90k.toml）
#   CFG=config_matched.toml bash run_train.sh  # 30k 版（对照/归因用）
#   SANITY=1 bash run_train.sh               # 冒烟：语料只做前 2000 条 + 3 epoch 训练
#   SKIP_PREP=1 bash run_train.sh            # 语料已就绪，只训练+评测
#   SKIP_PREP=1 SKIP_TRAIN=1 bash run_train.sh   # 只评测（训练完之后）
#   GPU=1 SHARED=0 bash run_train.sh         # 独占卡：batch 100 × accum 1（config 默认，五路各 20）
#   GPU=1 BS=20 ACC=5 SKIP_PREP=1 bash run_train.sh   # 换卡重跑训练：语料复用，加大 micro-batch 提速
#   EPOCHS=6 SKIP_PREP=1 bash run_train.sh   # 覆盖 epoch 数
#
# Step 0：config 的 data_dir_820 下没有 train_utts.csv 时，若 config 带 [prep].generator
#         （90k 版有），自动调 8.20 的 prepare_multicond_data.py 生成（输出落本目录 exp/）。
# 断点续跑：prepare_enh_data.py 按块跳过已完成部分；训练/评测不支持续跑只能重跑。
# 磁盘预算：90k 语料（noisy+enhanced）约 34GB；30k 约 11GB。跑前 df -h 确认。
# ============================================================
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CFG="${CFG:-config_matched_90k.toml}"
GPU="${GPU:-3}"
SHARED="${SHARED:-1}"                # 1=共享卡配方（batch10 accum10 amp）
EPOCHS="${EPOCHS:-}"
SANITY="${SANITY:-0}"
SKIP_PREP="${SKIP_PREP:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

# 从 config 读路径（base = config 所在目录）
py_get() {  # py_get <config> <dotted-key>
    python - "$1" "$2" <<'PYEOF'
import os, sys, toml
cfg = toml.load(sys.argv[1])
base = os.path.dirname(os.path.abspath(sys.argv[1]))
val = cfg
for k in sys.argv[2].split("."):
    val = val[k]
p = val if os.path.isabs(val) else os.path.normpath(os.path.join(base, val))
print(p)
PYEOF
}

DATA_DIR="$(py_get "$CFG" paths.data_dir_820)"
EXP_DIR="$(py_get "$CFG" paths.exp_dir)"
GEN_CFG="$(python - "$CFG" <<'PYEOF'
import sys, toml
try:
    print(toml.load(sys.argv[1]).get("prep", {}).get("generator", ""))
except Exception:
    print("")
PYEOF
)"
NPZ="../8.21/exp/prototypes/prototypes.npz"
BEST="$EXP_DIR/checkpoints/checkpoint_best.tar"

# ---- Step 0: 训练数据（8.20 产物或本目录 90k 生成；只读检查，缺了按需生成）----
if [ ! -f "$DATA_DIR/train_utts.csv" ]; then
    if [ -n "$GEN_CFG" ] && [ -f "$GEN_CFG" ]; then
        echo "==== Step 0: 生成 90k 训练数据（8.20 prepare，输出到 $DATA_DIR） ===="
        python ../8.20/prepare_multicond_data.py --config "$GEN_CFG"
    else
        echo "[ERROR] 缺 $DATA_DIR/train_utts.csv（先跑 8.20 的 prepare_multicond_data.py）"; exit 1
    fi
else
    echo "==== Step 0: 训练数据就绪（$DATA_DIR） ===="
fi

[ -f "$NPZ" ] || {
    echo "[ERROR] 缺 $NPZ（先在 8.21 跑 prepare_prototypes.py，或跑一次 8.21 的 run_sanity.sh）"; exit 1; }

# ---- Step 1: enh 语料（混噪 + 前端增强 + bench；块级断点续跑）----
if [ "$SKIP_PREP" != "1" ]; then
    echo "==== Step 1: enh 语料准备（部署前端 = 8.18 spk-preserve, 冻结） ===="
    LIMIT_ARGS=""
    [ "$SANITY" = "1" ] && LIMIT_ARGS="--limit 2000"
    python prepare_enh_data.py --config "$CFG" --gpu "$GPU" --all $LIMIT_ARGS
else
    echo "==== Step 1: 跳过（SKIP_PREP=1，语料应已就绪） ===="
fi

# ---- Step 2: 五条件混合训练 ----
if [ "$SKIP_TRAIN" != "1" ]; then
    echo "==== Step 2: 五条件训练（clean/raw0/raw-5/enh0/enh-5 均匀 1/5） ===="
    EXTRA=""
    [ -n "$EPOCHS" ] && EXTRA="$EXTRA --epochs $EPOCHS"
    if [ "$SANITY" = "1" ]; then
        EXTRA="$EXTRA --sanity --allow-partial-corpus"
    fi
    # 小显存配方默认 micro 10 × accum 10（有效 100）。显存富余可加大 micro-batch 提速：
    #   BS=20 ACC=5 / BS=50 ACC=2 / BS=100 ACC=1 —— 有效 batch 保持 100，训练数学不变；
    #   BS 必须是 5 的倍数（五路均分）
    BS="${BS:-10}" ACC="${ACC:-10}" WORKERS="${WORKERS:-4}" VBS="${VBS:-8}"
    if [ $((BS * ACC)) -ne 100 ]; then
        echo "[WARN] BS*ACC = $((BS * ACC)) != 100：有效 batch 变了，与默认配方不可比"
    fi
    if [ "$SHARED" = "1" ]; then
        PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
        python train_matched.py --config "$CFG" --gpu "$GPU" \
            --amp --batch-size "$BS" --accum "$ACC" \
            --num-workers "$WORKERS" --val-batch-size "$VBS" $EXTRA
    else
        python train_matched.py --config "$CFG" --gpu "$GPU" $EXTRA
    fi
fi

[ -f "$BEST" ] || { echo "[ERROR] 没有 $BEST"; exit 1; }

# ---- Step 3: 正式 open-set 评测（复用 8.21 的 evaluate_combo，结果落本目录 exp/）----
echo "==== Step 3: 正式 open-set 评测（trials_cross_device, 3s, 官方打分） ===="
# 前端测试侧默认 = 8.18 eval_e10 已增强的官方测试集（与 7.7902 那次逐位一致）
ENH_0DB="${ENH_0DB:-../8.18/exp/spk_preserve/eval_e10/denoised/noise_0db}"
ENH_M5DB="${ENH_M5DB:-../8.18/exp/spk_preserve/eval_e10/denoised/noise_m5db}"
python ../8.21/evaluate_combo.py --config "$CFG" --checkpoint "$BEST" --gpu "$GPU" \
    --enh-0db "$ENH_0DB" --enh-m5db "$ENH_M5DB"

echo ""
echo "==== 完成 ===="
echo "结果：$EXP_DIR/eval/combo/{scores,combo_summary.json}"
echo "对照（8.21 实测）：raw robust 7.6911/9.8781；级联 7.5000/9.5085 (-10.6%/-14.0%)"
echo "验收：0dB EER ≤ 5.874 且 -5dB EER ≤ 7.739（相对官方基线 -30%）"
echo "     clean 退化 <1pp 以 combo 汇总里的 open-set clean EER 为准"
