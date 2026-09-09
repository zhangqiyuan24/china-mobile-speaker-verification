#!/bin/bash
# ============================================================
# 微调 ERes2Net-large (eres2net_1) 全量评测
#
# 评测条件: clean / noise_0db / noise_m5db / reverb
#           overlap2_cn / overlap2_en / overlap3_cn / overlap3_en
# 固定 3s，仅 cross_device
# 结果保存到 ../sv-ecapa/data/result/eres2net_1/
# 不覆盖原有 ERes2Net 结果
# ============================================================
set -e
source ./path.sh

MODEL_NAME="eres2net_1"
FINETUNED_CKPT="../sv-eres2net/exp/eres2net_finetune/models/CKPT-EPOCH-15-00/embedding_model.ckpt"

if [ ! -f "$FINETUNED_CKPT" ]; then
    echo "[ERROR] 微调模型不存在: $FINETUNED_CKPT"
    echo "  请先完成训练"
    exit 1
fi

# ---- 配置 ----
DURATION="3s"
TRIALS="data/data_for_test/clean/${DURATION}/trials_cross_device"
RESULT_BASE="data/result/${MODEL_NAME}"
DATA_BASE="data/data_for_test"
EMB_BASE="../sv-eres2net/exp/eval_${MODEL_NAME}/embeddings"

# 切换到 sv-ecapa 目录，wav_list.txt 和 trial 路径都相对于此
cd ../sv-ecapa

# Clean enrollment embeddings（所有条件共用，和原始 ERes2Net 一样）
CLEAN_EMB="${EMB_BASE}/clean"

echo "============================================"
echo "  微调模型评测: ${MODEL_NAME}"
echo "  时长: ${DURATION}  协议: cross_device"
echo "  结果: ${RESULT_BASE}/"
echo "============================================"

# ============================================================
# Step 0: 提取 clean enrollment embeddings（只跑一次）
# ============================================================
if [ ! -f "${CLEAN_EMB}/xvector_0.scp" ]; then
    echo ""
    echo ">> 提取 clean enrollment embeddings..."
    rm -rf "${CLEAN_EMB}"
    python ../../../speakerlab/bin/infer_sv_batch.py \
        --model_id iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k \
        --pretrained "$FINETUNED_CKPT" \
        --wavs "${DATA_BASE}/clean/${DURATION}/wav_list.txt" \
        --feat_out_dir "${CLEAN_EMB}" \
        --feat_out_format ark \
        --batch_size 256
else
    echo ">> clean embeddings 已存在，跳过"
fi

echo ""
echo "============================================"
echo "  条件评测"
echo "============================================"

# ============================================================
# 评测函数
# ============================================================
evaluate_condition() {
    local COND=$1      # clean / noise_0db / overlap2_cn ...
    local WAV_LIST="${DATA_BASE}/${COND}/${DURATION}/wav_list.txt"

    if [ ! -f "$WAV_LIST" ]; then
        echo "  [跳过] ${COND}: wav_list.txt 不存在"
        return
    fi

    local TEST_EMB="${EMB_BASE}/${COND}"
    local SCORE_DIR="${RESULT_BASE}/${COND}/${DURATION}"

    echo ""
    echo "--- ${COND} ---"

    # 提取测试端 embedding
    rm -rf "$TEST_EMB"
    python ../../../speakerlab/bin/infer_sv_batch.py \
        --model_id iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k \
        --pretrained "$FINETUNED_CKPT" \
        --wavs "$WAV_LIST" \
        --feat_out_dir "$TEST_EMB" \
        --feat_out_format ark \
        --batch_size 256

    # 评分: enrolment=clean, test=条件
    mkdir -p "$SCORE_DIR"
    python ../../../speakerlab/bin/compute_score_metrics.py \
        --enrol_data "${CLEAN_EMB}" \
        --test_data "$TEST_EMB" \
        --scores_dir "$SCORE_DIR" \
        --trials "$TRIALS"

    grep -E "EER|minDCF" "${SCORE_DIR}/result.metrics"
}

# Clean（enrol=test=clean，用原来的 clean embedding 就行）
echo ""
echo "--- clean (enrol=test) ---"
SCORE_DIR="${RESULT_BASE}/clean/${DURATION}"
mkdir -p "$SCORE_DIR"
python ../../../speakerlab/bin/compute_score_metrics.py \
    --enrol_data "${CLEAN_EMB}" \
    --test_data "${CLEAN_EMB}" \
    --scores_dir "$SCORE_DIR" \
    --trials "$TRIALS"
grep -E "EER|minDCF" "${SCORE_DIR}/result.metrics"

# 其他条件
for COND in noise_0db noise_m5db reverb \
            overlap2_cn overlap2_en overlap3_cn overlap3_en; do
    evaluate_condition "$COND"
done

# ============================================================
# 汇总
# ============================================================
echo ""
echo "============================================"
echo "  评测完成！"
echo "============================================"
echo ""
echo "  结果汇总:"
echo "  ─────────────────────────────────────"
for COND in clean noise_0db noise_m5db reverb \
            overlap2_cn overlap2_en overlap3_cn overlap3_en; do
    METRICS="${RESULT_BASE}/${COND}/${DURATION}/result.metrics"
    if [ -f "$METRICS" ]; then
        EER=$(grep "EER" "$METRICS" | head -1 | awk '{print $NF}')
        DCF=$(grep "minDCF" "$METRICS" | head -1 | awk '{print $NF}')
        printf "  %-14s  EER=%7s  minDCF=%7s\n" "$COND" "$EER" "$DCF"
    fi
done
echo "  ─────────────────────────────────────"
