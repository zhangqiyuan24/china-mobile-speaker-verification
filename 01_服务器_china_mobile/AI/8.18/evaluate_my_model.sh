#!/bin/bash
# ============================================================
# 自定义模型评测 (8.18 专用)
# 模型：exp/spk_preserve/checkpoint_epoch3.tar
# 输出：exp/spk_preserve/eval_results/
# 代码依赖：8.17/code (speech_enhance)
# ============================================================
set -e

SKIP_DENOISE=false
EMB_BATCH=32
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-denoise) SKIP_DENOISE=true; shift ;;
        --batch-size) EMB_BATCH="$2"; shift 2 ;;
        *) echo "用法: $0 [--skip-denoise] [--batch-size N]"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- GPU ----
DENOISE_GPU="1"
EMB_GPU="0,3"

# ---- 模型路径（8.18）：可用环境变量 MODEL 覆盖，相对路径按脚本目录解析 ----
MODEL="${MODEL:-$SCRIPT_DIR/exp/spk_preserve/checkpoint_epoch3_fixed.tar}"
case "$MODEL" in /*) ;; *) MODEL="$SCRIPT_DIR/$MODEL" ;; esac
if [ ! -f "$MODEL" ]; then
    echo "[ERROR] 模型不存在: $MODEL"
    exit 1
fi

# ---- 输出根目录（8.18）：可用环境变量 OUTPUT_ROOT 覆盖（多模型对比时避免互相覆盖） ----
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/exp/spk_preserve/eval_results}"
case "$OUTPUT_ROOT" in /*) ;; *) OUTPUT_ROOT="$SCRIPT_DIR/$OUTPUT_ROOT" ;; esac
mkdir -p "$OUTPUT_ROOT"

# ---- 代码路径（指向 8.17/code） ----
CODE_DIR="$(cd "$SCRIPT_DIR/../8.17/code" && pwd)"
if [ ! -d "$CODE_DIR" ]; then
    echo "[ERROR] 代码目录不存在: $CODE_DIR"
    exit 1
fi

# 设置 PYTHONPATH，使 speech_enhance 可导入
export PYTHONPATH="$CODE_DIR:$CODE_DIR/speech_enhance:$PYTHONPATH"

# ---- 依赖路径 ----
SV_ECAPA="$(cd "$SCRIPT_DIR/../../code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa" && pwd)"
DURATION="3s"
# 可用环境变量 CONDITIONS 覆盖（如快速筛选只跑 "noise_0db"）
CONDITIONS="${CONDITIONS:-noise_0db noise_m5db}"

echo "============================================"
echo "  8.18 自定义模型评测"
echo "  模型: $MODEL"
echo "  输出: $OUTPUT_ROOT"
echo "  代码: $CODE_DIR"
echo "============================================"

# ---- Step 1: 降噪 ----
echo ""
echo "==== Step 1: FullSubNet+ 降噪 ===="
for COND in $CONDITIONS; do
    INPUT_DIR="$SV_ECAPA/data/data_for_test/$COND/$DURATION"
    OUTPUT_DIR="$OUTPUT_ROOT/denoised/$COND"
    [ ! -d "$INPUT_DIR" ] && echo "  [跳过] $COND 输入目录不存在" && continue

    echo "  --- $COND ---"
    WAV_LIST="$OUTPUT_DIR/wav_list.txt"
    if $SKIP_DENOISE && [ -f "$WAV_LIST" ] && [ $(wc -l < "$WAV_LIST") -eq 16401 ]; then
        echo "    [跳过降噪] 已完成"
        continue
    fi

    # 切换到代码目录，且 PYTHONPATH 已设置
    cd "$CODE_DIR"
    CUDA_VISIBLE_DEVICES=$DENOISE_GPU python -m speech_enhance.tools.inference \
        -C config/inference.toml \
        -M "$MODEL" \
        -I "$INPUT_DIR" \
        -O "$OUTPUT_DIR"
    cd "$SCRIPT_DIR"

    ENHANCED_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "enhanced_*" | head -1)
    [ -z "$ENHANCED_DIR" ] && echo "  [ERROR] 未找到增强输出" && exit 1
    find "$ENHANCED_DIR" -name "*.wav" | sort > "$WAV_LIST"
    echo "    降噪完成，共 $(wc -l < "$WAV_LIST") 条"
done

# ---- Step 2: 提 embedding + 打分 ----
echo ""
echo "==== Step 2: 提 embedding + 打分 ===="
cd "$SV_ECAPA"
source path.sh

for COND in $CONDITIONS; do
    WAV_LIST="$OUTPUT_ROOT/denoised/$COND/wav_list.txt"
    [ ! -f "$WAV_LIST" ] && echo "  [跳过] $COND 无 wav_list" && continue

    EMB_DIR="$OUTPUT_ROOT/embeddings/denoised_${COND}/eres2net/${DURATION}"
    CLEAN_EMB="$SV_ECAPA/exp/clean_bench/eres2net/embeddings_${DURATION}"
    SCORE_DIR="$OUTPUT_ROOT/scores/denoised_${COND}/eres2net/${DURATION}"
    TRIAL="$SV_ECAPA/data/data_for_test/clean/${DURATION}/trials_cross_device"

    echo "  --- $COND ---"
    [ -d "$EMB_DIR" ] && rm -rf "$EMB_DIR"

    CUDA_VISIBLE_DEVICES=$EMB_GPU python ../../../speakerlab/bin/infer_sv_batch.py \
        --model_id iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k \
        --wavs "$WAV_LIST" \
        --feat_out_dir "$EMB_DIR" \
        --feat_out_format ark \
        --batch_size "$EMB_BATCH"

    mkdir -p "$SCORE_DIR"
    python ../../../speakerlab/bin/compute_score_metrics.py \
        --enrol_data "$CLEAN_EMB" \
        --test_data "$EMB_DIR" \
        --scores_dir "$SCORE_DIR" \
        --trials "$TRIAL"

    echo "  结果 → $SCORE_DIR/result.metrics"
done

echo ""
echo "============================================"
echo "  评测完成！结果汇总："
echo "============================================"
for COND in $CONDITIONS; do
    METRICS="$OUTPUT_ROOT/scores/denoised_${COND}/eres2net/${DURATION}/result.metrics"
    [ -f "$METRICS" ] && echo "  --- $COND ---" && grep -E "EER|minDCF" "$METRICS" | head -2
done