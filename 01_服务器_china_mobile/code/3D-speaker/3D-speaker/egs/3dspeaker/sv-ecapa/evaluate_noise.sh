#!/bin/bash
# ============================================================
# Noise Benchmark 一键评测脚本
#
# 对 2 种 SNR × 3 时长 × 3 模型 进行评测（仅 cross_device）
# 0dB 和 -5dB 使用完全相同的噪声片段，仅 SNR 不同
#
# 前置条件：
#   1. bash evaluate_clean.sh         ← clean embedding 已提取
#   2. python gen_noise_benchmark.py  ← noise 音频已生成
#
# 用法：bash evaluate_noise.sh
# ============================================================
set -e

source path.sh

CONDITIONS="noise_0db noise_m5db"
DURATIONS="1 3 5"
BENCH_DIR="data/data_for_test"
RESULT_BASE="${BENCH_DIR}/clean/result"

echo "============================================"
echo "  Noise Benchmark 评测（仅 cross_device）"
echo "  SNR: ${CONDITIONS}"
echo "============================================"

# ============================================================
# 1/3: ECAPA-TDNN
# ============================================================
echo ""
echo "============================================"
echo "  1/3: ECAPA-TDNN"
echo "============================================"

for cond in $CONDITIONS; do
    for d in $DURATIONS; do
        WAV_LIST="${BENCH_DIR}/${cond}/${d}s/wav_list.txt"
        if [ ! -f "$WAV_LIST" ]; then
            echo "  [跳过] ${cond}/${d}s: wav_list.txt 不存在"
            continue
        fi

        EMB_DIR="exp/clean_bench/${cond}/ecapa/embeddings_${d}s"
        CLEAN_EMB="exp/clean_bench/ecapa/embeddings_${d}s"
        SCORE_DIR="${RESULT_BASE}/${cond}/ecapa/${d}s"
        TRIAL="${BENCH_DIR}/clean/${d}s/trials_cross_device"

        echo ""
        echo "--- ECAPA  ${cond}  ${d}s ---"

        python ../../../speakerlab/bin/infer_sv_batch.py \
            --model_id iic/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k \
            --wavs "$WAV_LIST" \
            --feat_out_dir "$EMB_DIR" \
            --feat_out_format ark

        mkdir -p "$SCORE_DIR"
        python ../../../speakerlab/bin/compute_score_metrics.py \
            --enrol_data "$CLEAN_EMB" \
            --test_data "$EMB_DIR" \
            --scores_dir "$SCORE_DIR" \
            --trials "$TRIAL"

        echo "  结果 → ${SCORE_DIR}/result.metrics"
    done
done

# ============================================================
# 2/3: ERes2Net-large
# ============================================================
echo ""
echo "============================================"
echo "  2/3: ERes2Net-large"
echo "============================================"

for cond in $CONDITIONS; do
    for d in $DURATIONS; do
        WAV_LIST="${BENCH_DIR}/${cond}/${d}s/wav_list.txt"
        if [ ! -f "$WAV_LIST" ]; then
            echo "  [跳过] ${cond}/${d}s: wav_list.txt 不存在"
            continue
        fi

        EMB_DIR="exp/clean_bench/${cond}/eres2net/embeddings_${d}s"
        CLEAN_EMB="exp/clean_bench/eres2net/embeddings_${d}s"
        SCORE_DIR="${RESULT_BASE}/${cond}/eres2net/${d}s"
        TRIAL="${BENCH_DIR}/clean/${d}s/trials_cross_device"

        echo ""
        echo "--- ERes2Net  ${cond}  ${d}s ---"

        python ../../../speakerlab/bin/infer_sv_batch.py \
            --model_id iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k \
            --wavs "$WAV_LIST" \
            --feat_out_dir "$EMB_DIR" \
            --feat_out_format ark

        mkdir -p "$SCORE_DIR"
        python ../../../speakerlab/bin/compute_score_metrics.py \
            --enrol_data "$CLEAN_EMB" \
            --test_data "$EMB_DIR" \
            --scores_dir "$SCORE_DIR" \
            --trials "$TRIAL"

        echo "  结果 → ${SCORE_DIR}/result.metrics"
    done
done

# ============================================================
# 3/3: ReDimNet2-B6
# ============================================================
echo ""
echo "============================================"
echo "  3/3: ReDimNet2-B6"
echo "============================================"

for cond in $CONDITIONS; do
    for d in $DURATIONS; do
        WAV_LIST="${BENCH_DIR}/${cond}/${d}s/wav_list.txt"
        if [ ! -f "$WAV_LIST" ]; then
            echo "  [跳过] ${cond}/${d}s: wav_list.txt 不存在"
            continue
        fi

        EMB_BASE="exp/clean_bench/${cond}/redimnet2_${d}s"
        EMB_DIR="${EMB_BASE}/redimnet2_b6_vb2+vox2+cnc2_v0_lm/embeddings"
        CLEAN_EMB="exp/clean_bench/redimnet2_${d}s/redimnet2_b6_vb2+vox2+cnc2_v0_lm/embeddings"
        SCORE_DIR="${RESULT_BASE}/${cond}/redimnet2/${d}s"
        TRIAL="${BENCH_DIR}/clean/${d}s/trials_cross_device"

        echo ""
        echo "--- ReDimNet2  ${cond}  ${d}s ---"

        python ../../../speakerlab/bin/infer_sv_new.py \
            --model_id redimnet2/b6/vb2+vox2+cnc2_v0/lm \
            --wavs "$WAV_LIST" \
            --local_model_dir "$EMB_BASE" \
            --feat_out_format ark

        mkdir -p "$SCORE_DIR"
        python ../../../speakerlab/bin/compute_score_metrics.py \
            --enrol_data "$CLEAN_EMB" \
            --test_data "$EMB_DIR" \
            --scores_dir "$SCORE_DIR" \
            --trials "$TRIAL"

        echo "  结果 → ${SCORE_DIR}/result.metrics"
    done
done

# ============================================================
# 汇总
# ============================================================
echo ""
echo "============================================"
echo "  评测完成！"
echo "============================================"
echo ""
for cond in $CONDITIONS; do
    for model in ecapa eres2net redimnet2; do
        echo "  ── ${cond} / ${model} ──"
        for d in $DURATIONS; do
            m="${RESULT_BASE}/${cond}/${model}/${d}s/result.metrics"
            if [ -f "$m" ]; then
                echo "    ${d}s:"
                grep -E "EER|minDCF" "$m" | sed 's/^/      /'
            fi
        done
    done
done
