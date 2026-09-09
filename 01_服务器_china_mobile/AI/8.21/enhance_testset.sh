#!/bin/bash
# ============================================================
# 用 8.17/8.18 的 FullSubNet+ 前端离线增强正式测试集的噪声两档
# （evaluate_combo.py 的 --enh-0db/--enh-m5db 输入，只读 8.17 代码与 ckpt）
#
# 用法：
#   bash enhance_testset.sh
#   GPU=1 bash enhance_testset.sh
#   # 用 speaker-preserve 微调版前端（推荐，8.18/8.19 训练产物）：
#   FULLSUBNET_CKPT=../8.19/exp/spk_preserve/<best>.tar bash enhance_testset.sh
# ============================================================
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

GPU="${GPU:-3}"
FSN_ROOT="${FSN_ROOT:-../8.17/code}"                 # FullSubNet+ 代码（只读）
CKPT="${FULLSUBNET_CKPT:-../8.17/models/best_model.tar}"
# 默认 = 原版 FullSubNet+；8.18/8.19 的 speaker-preserve 微调 ckpt 在
# ../8.19/exp/spk_preserve/ 下，确认文件名后用 FULLSUBNET_CKPT=... 指过去

SV_ECAPA="${SV_ECAPA:-../../code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa}"
OUT_ROOT="${OUT_ROOT:-exp/enhanced}"

# 全部转绝对路径，避免 cd 进 FullSubNet+ 目录后相对路径失效
FSN_ROOT_ABS="$(cd "$FSN_ROOT" && pwd)"
CKPT_ABS="$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")"
SV_ECAPA_ABS="$(cd "$SV_ECAPA" && pwd)"
CFG_INFERENCE="$FSN_ROOT_ABS/config/inference.toml"

[ -d "$FSN_ROOT_ABS" ] || { echo "[ERROR] 缺 FullSubNet+ 代码目录 $FSN_ROOT_ABS"; exit 1; }
[ -f "$CKPT_ABS" ] || { echo "[ERROR] 缺前端 ckpt $CKPT_ABS"; exit 1; }
[ -f "$CFG_INFERENCE" ] || { echo "[ERROR] 缺推理配置 $CFG_INFERENCE"; exit 1; }

for COND in noise_0db noise_m5db; do
    IN_DIR="$SV_ECAPA_ABS/data/data_for_test/$COND/3s"   # 含 A/ D/ 子目录，工具递归处理
    OUT_DIR="$HERE/$OUT_ROOT/$COND"
    [ -d "$IN_DIR" ] || { echo "[ERROR] 缺测试目录 $IN_DIR"; exit 1; }
    if find "$OUT_DIR" -name "*.wav" 2>/dev/null | grep -q .; then
        echo "==== $COND 已增强（$OUT_DIR 有 wav），跳过；重跑请删除该目录 ===="
        continue
    fi
    echo "==== 增强 $COND：$IN_DIR -> $OUT_DIR ===="
    mkdir -p "$OUT_DIR"
    (cd "$FSN_ROOT_ABS" && CUDA_VISIBLE_DEVICES="$GPU" \
        python -m speech_enhance.tools.inference \
            -C "$CFG_INFERENCE" -M "$CKPT_ABS" -I "$IN_DIR" -O "$OUT_DIR")
done

echo ""
echo "==== 完成 ===="
echo "增强 wav：$OUT_ROOT/noise_0db  $OUT_ROOT/noise_m5db"
echo "下一步组合评测："
echo "  ENH_0DB=$OUT_ROOT/noise_0db ENH_M5DB=$OUT_ROOT/noise_m5db bash run_train.sh"
echo "  （或 SKIP_TRAIN=1 只跑评测）"
