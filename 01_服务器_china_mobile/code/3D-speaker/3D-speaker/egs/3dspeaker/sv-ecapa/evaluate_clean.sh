#!/bin/bash
# ============================================================
# Clean Benchmark 一键评测脚本
#
# 对 3 个时长 (1s/3s/5s) × 3 个模型 × 4 种 trial 进行完整评测
# 结果保存到 data/data_for_test/clean/result/
#
# 前置条件：
#   1. python gen_clean_benchmark.py    ← 已运行
#   2. python gen_clean_protocols.py    ← 已运行
#
# 用法：bash evaluate_clean.sh
# ============================================================
set -e

source path.sh

DURATIONS="1 3 5"
TRIAL_NAMES="trials_all trials_cross_device trials_cross_distance trials_cross_dialect"
BENCH_DIR="data/data_for_test/clean"
RESULT_BASE="${BENCH_DIR}/result"

echo "============================================"
echo "  Clean Benchmark 评测"
echo "  时长: $DURATIONS"
echo "  模型: ECAPA-TDNN / ERes2Net-large / ReDimNet2-B6"
echo "============================================"

# ────────────────────────────────────────
# Step 0: 生成各时长的 wav_list.txt（仅 A + D，因为 trial 只评测 A vs A/D）
# ────────────────────────────────────────
echo ""
echo "生成 wav_list.txt..."
for d in $DURATIONS; do
    WDIR="${BENCH_DIR}/${d}s"
    if [ ! -d "$WDIR/A" ]; then
        echo "  [ERROR] 目录不存在: $WDIR/A，请先运行 gen_clean_benchmark.py"
        exit 1
    fi
    > "${WDIR}/wav_list.txt"
    find "${WDIR}/A" -name "*.wav" >> "${WDIR}/wav_list.txt"
    find "${WDIR}/D" -name "*.wav" >> "${WDIR}/wav_list.txt"
    n=$(wc -l < "${WDIR}/wav_list.txt")
    echo "  ${d}s: ${n} 条 (A + D)"
done

# ────────────────────────────────────────
# Step 1: ECAPA-TDNN
# ────────────────────────────────────────
echo ""
echo "============================================"
echo "  1/3: ECAPA-TDNN"
echo "============================================"

for d in $DURATIONS; do
    WAV_LIST="${BENCH_DIR}/${d}s/wav_list.txt"
    EMB_DIR="exp/clean_bench/ecapa/embeddings_${d}s"
    SCORE_DIR="${RESULT_BASE}/ecapa/${d}s"

    echo ""
    echo "--- ECAPA-TDNN  ${d}s ---"

    # 提取 embedding
    python ../../../speakerlab/bin/infer_sv_batch.py \
        --model_id iic/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k \
        --wavs "$WAV_LIST" \
        --feat_out_dir "$EMB_DIR" \
        --feat_out_format ark

    # 评分（4 种 trial）
    mkdir -p "$SCORE_DIR"
    python ../../../speakerlab/bin/compute_score_metrics.py \
        --enrol_data "$EMB_DIR" \
        --test_data "$EMB_DIR" \
        --scores_dir "$SCORE_DIR" \
        --trials \
            "${BENCH_DIR}/${d}s/trials_all" \
            "${BENCH_DIR}/${d}s/trials_cross_device" \
            "${BENCH_DIR}/${d}s/trials_cross_distance" \
            "${BENCH_DIR}/${d}s/trials_cross_dialect"

    echo "  结果 → ${SCORE_DIR}/result.metrics"
done

# ────────────────────────────────────────
# Step 2: ERes2Net-large
# ────────────────────────────────────────
echo ""
echo "============================================"
echo "  2/3: ERes2Net-large"
echo "============================================"

for d in $DURATIONS; do
    WAV_LIST="${BENCH_DIR}/${d}s/wav_list.txt"
    EMB_DIR="exp/clean_bench/eres2net/embeddings_${d}s"
    SCORE_DIR="${RESULT_BASE}/eres2net/${d}s"

    echo ""
    echo "--- ERes2Net-large  ${d}s ---"

    python ../../../speakerlab/bin/infer_sv_batch.py \
        --model_id iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k \
        --wavs "$WAV_LIST" \
        --feat_out_dir "$EMB_DIR" \
        --feat_out_format ark

    mkdir -p "$SCORE_DIR"
    python ../../../speakerlab/bin/compute_score_metrics.py \
        --enrol_data "$EMB_DIR" \
        --test_data "$EMB_DIR" \
        --scores_dir "$SCORE_DIR" \
        --trials \
            "${BENCH_DIR}/${d}s/trials_all" \
            "${BENCH_DIR}/${d}s/trials_cross_device" \
            "${BENCH_DIR}/${d}s/trials_cross_distance" \
            "${BENCH_DIR}/${d}s/trials_cross_dialect"

    echo "  结果 → ${SCORE_DIR}/result.metrics"
done

# ────────────────────────────────────────
# Step 3: ReDimNet2-B6
# ────────────────────────────────────────
echo ""
echo "============================================"
echo "  3/3: ReDimNet2-B6"
echo "============================================"

for d in $DURATIONS; do
    WAV_LIST="${BENCH_DIR}/${d}s/wav_list.txt"
    # ReDimNet2 通过 --local_model_dir 控制 embedding 输出路径
    EMB_BASE="exp/clean_bench/redimnet2_${d}s"
    # infer_sv_new.py 会在此之下创建 redimnet2_b6_vb2+vox2+cnc2_v0_lm/embeddings/
    EMB_DIR="${EMB_BASE}/redimnet2_b6_vb2+vox2+cnc2_v0_lm/embeddings"
    SCORE_DIR="${RESULT_BASE}/redimnet2/${d}s"

    echo ""
    echo "--- ReDimNet2-B6  ${d}s ---"

    python ../../../speakerlab/bin/infer_sv_new.py \
        --model_id redimnet2/b6/vb2+vox2+cnc2_v0/lm \
        --wavs "$WAV_LIST" \
        --local_model_dir "$EMB_BASE" \
        --feat_out_format ark

    mkdir -p "$SCORE_DIR"
    python ../../../speakerlab/bin/compute_score_metrics.py \
        --enrol_data "$EMB_DIR" \
        --test_data "$EMB_DIR" \
        --scores_dir "$SCORE_DIR" \
        --trials \
            "${BENCH_DIR}/${d}s/trials_all" \
            "${BENCH_DIR}/${d}s/trials_cross_device" \
            "${BENCH_DIR}/${d}s/trials_cross_distance" \
            "${BENCH_DIR}/${d}s/trials_cross_dialect"

    echo "  结果 → ${SCORE_DIR}/result.metrics"
done

# ────────────────────────────────────────
# 汇总
# ────────────────────────────────────────
echo ""
echo "============================================"
echo "  评测完成！"
echo "============================================"
echo ""
echo "结果目录: ${RESULT_BASE}/"
echo ""
echo "快速查看所有 EER："
for model in ecapa eres2net redimnet2; do
    echo "  ── ${model} ──"
    for d in $DURATIONS; do
        metrics_file="${RESULT_BASE}/${model}/${d}s/result.metrics"
        if [ -f "$metrics_file" ]; then
            echo "    ${d}s:"
            grep -E "EER|minDCF" "$metrics_file" | sed 's/^/      /'
        fi
    done
done
