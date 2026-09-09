#!/bin/bash
# ============================================================
# FullSubNet+ 降噪 + ERes2Net 声纹评测
#
# 流程:
#   1. FullSubNet+ 对 noise_0db / noise_m5db 的 3s wav 降噪
#   2. 对降噪后 wav 提 ERes2Net embedding
#   3. 与 clean enrollment embedding 打分 → EER
#
# GPU 分配: 降噪用 GPU 1 (单卡), 提 embedding 用 GPU 1,2,3
#
# 前提:
#   - bash setup_offline.sh 已跑 (依赖 + checkpoint 就位)
#   - bash evaluate_clean.sh 已跑 (clean enrollment embedding 已提取)
#
# 用法: bash evaluate_denoise.sh
# ============================================================
set -e

# ---- 可选参数 ----
SKIP_DENOISE=false
EMB_BATCH=32

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-denoise)
            SKIP_DENOISE=true
            shift
            ;;
        --batch-size)
            EMB_BATCH="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] 未知参数: $1"
            echo "用法: bash evaluate_denoise_resume.sh [--skip-denoise] [--batch-size N]"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- GPU ----
DENOISE_GPU="1"
EMB_GPU="1,3"

# ---- 路径 (全部绝对路径) ----
SV_ECAPA="$(cd "$SCRIPT_DIR/../../code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa" && pwd)"
CODE_DIR="$SCRIPT_DIR/code"
MODEL="$SCRIPT_DIR/models/best_model.tar"
DURATION="3s"
CONDITIONS="noise_0db noise_m5db"

echo "============================================"
echo "  FullSubNet+ 降噪 + ERes2Net 评测"
echo "  降噪 GPU: $DENOISE_GPU  提embedding GPU: $EMB_GPU"
echo "  embedding batch size: $EMB_BATCH"
echo "  skip denoise: $SKIP_DENOISE"
echo "============================================"

# ============================================================
# Step 1: 降噪 (FullSubNet+)
# ============================================================
echo ""
echo "==== Step 1: FullSubNet+ 降噪 ===="

for COND in $CONDITIONS; do
    INPUT_DIR="$SV_ECAPA/data/data_for_test/$COND/$DURATION"
    OUTPUT_DIR="$SV_ECAPA/data/data_for_test/denoised/$COND"

    if [ ! -d "$INPUT_DIR" ]; then
        echo "  [跳过] $COND: 输入目录不存在 $INPUT_DIR"
        continue
    fi

    echo ""
    echo "  --- $COND ---"
    echo "    输入: $INPUT_DIR"
    echo "    输出: $OUTPUT_DIR"

    WAV_LIST="$OUTPUT_DIR/wav_list.txt"
    EXISTING_COUNT=0
    if [ -f "$WAV_LIST" ]; then
        EXISTING_COUNT=$(wc -l < "$WAV_LIST")
    fi

    if $SKIP_DENOISE && [ "$EXISTING_COUNT" -gt 0 ]; then
        echo "    [跳过降噪] 已有 wav_list.txt，共 $EXISTING_COUNT 条"
        continue
    fi

    # 自动断点：如果已经有完整 16401 条，则跳过降噪
    if [ "$EXISTING_COUNT" -eq 16401 ]; then
        echo "    [跳过降噪] 已完成，共 16401 条"
        continue
    fi

    cd "$CODE_DIR"
    CUDA_VISIBLE_DEVICES=$DENOISE_GPU python -m speech_enhance.tools.inference \
        -C config/inference.toml \
        -M "$MODEL" \
        -I "$INPUT_DIR" \
        -O "$OUTPUT_DIR"
    cd "$SCRIPT_DIR"

    # 找到 enhanced_{epoch} 目录
    ENHANCED_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "enhanced_*" | head -1)
    if [ -z "$ENHANCED_DIR" ]; then
        echo "  [ERROR] 找不到 enhanced 输出目录: $OUTPUT_DIR"
        exit 1
    fi
    echo "    增强输出: $ENHANCED_DIR"

    # 生成 wav_list.txt
    find "$ENHANCED_DIR" -name "*.wav" | sort > "$WAV_LIST"
    echo "    降噪后 wav 数: $(wc -l < "$WAV_LIST")"
done

# ============================================================
# Step 2: 提 embedding + 打分
# ============================================================
echo ""
echo "==== Step 2: 提 embedding + 打分 ===="

cd "$SV_ECAPA"
source path.sh
for COND in $CONDITIONS; do
    WAV_LIST="data/data_for_test/denoised/$COND/wav_list.txt"
    if [ ! -f "$WAV_LIST" ]; then
        echo "  [跳过] $COND: wav_list.txt 不存在"
        continue
    fi

    EMB_DIR="exp/clean_bench/denoised_${COND}/eres2net/embeddings_${DURATION}"
    CLEAN_EMB="exp/clean_bench/eres2net/embeddings_${DURATION}"
    SCORE_DIR="data/data_for_test/clean/result/denoised_${COND}/eres2net/${DURATION}"
    TRIAL="data/data_for_test/clean/${DURATION}/trials_cross_device"

    echo ""
    echo "  --- $COND ---"

    # 提 embedding (多卡)
    # 若上次 OOM 留下了不完整输出，先删除后只重跑 embedding，不影响前面的降噪 wav
    if [ -d "$EMB_DIR" ]; then
        echo "    清理旧/不完整 embedding 目录: $EMB_DIR"
        rm -rf "$EMB_DIR"
    fi

    CUDA_VISIBLE_DEVICES=$EMB_GPU python ../../../speakerlab/bin/infer_sv_batch.py \
        --model_id iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k \
        --wavs "$WAV_LIST" \
        --feat_out_dir "$EMB_DIR" \
        --feat_out_format ark \
        --batch_size "$EMB_BATCH"

    # 打分
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
echo "  降噪评测完成！"
echo "============================================"
for COND in $CONDITIONS; do
    METRICS="data/data_for_test/clean/result/denoised_${COND}/eres2net/${DURATION}/result.metrics"
    if [ -f "$METRICS" ]; then
        echo "  --- $COND ---"
        grep -E "EER|minDCF" "$METRICS" | head -2
    fi
done
