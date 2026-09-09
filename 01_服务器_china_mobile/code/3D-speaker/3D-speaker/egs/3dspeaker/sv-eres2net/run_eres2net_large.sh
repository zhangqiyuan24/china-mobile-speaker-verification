#!/bin/bash
# ============================================================
# ERes2Net-large 鲁棒微调 — 一键脚本
#
# 4×P100 12GB: batch_size=128 (32/gpu)
# 训练量: ~48K utterances × 15 epochs ≈ 2-3 小时
#
# debug 模式 (不冻结+少量数据):
#   bash run_eres2net_large.sh --debug
#
# 用法: bash run_eres2net_large.sh
# ============================================================
set -e
. ./path.sh || exit 1

# ---- 配置 ----
gpus="0 1 2 3"
num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
seed=2026

# ---- 数据路径 (相对于 sv-eres2net/) ----
SV_ECAPA="../sv-ecapa"
# 训练数据目录
TRAIN_DIR="${SV_ECAPA}/data/data_for_train/train"
# 测试集 ori 目录 (提取 speaker 列表用)
ORI_DIR="${SV_ECAPA}/data/data_for_test/ori"
# 噪声文件
MUSAN_DIR="${SV_ECAPA}/data/musan"
# RIR 文件
RIRS_DIR="${SV_ECAPA}/data/rirs_noises/RIRS_NOISES/simulated_rirs/mediumroom"
# 预训练权重
PRETRAINED="../../../pretrained/speech_eres2net_large_sv_zh-cn_3dspeaker_16k/eres2net_large_model.ckpt"
# 输出目录
DATA_DIR="data"
EXP_DIR="exp/eres2net_finetune_2"

# ---- debug 模式 ----
DEBUG=false
if [ "$1" == "--debug" ]; then
    DEBUG=true
    echo "[DEBUG MODE] 少量数据 + 1 epoch"
fi

# ============================================================
# Step 0: 生成辅助文件
# ============================================================
echo "============================================"
echo "  Step 0: 生成辅助文件"
echo "============================================"

# test_speakers.txt
mkdir -p "$DATA_DIR"
TEST_SPKS="$DATA_DIR/test_speakers.txt"
if [ ! -f "$TEST_SPKS" ]; then
    echo "  生成 test_speakers.txt ..."
    for r in A B C D E F; do
        find "$ORI_DIR/$r" -maxdepth 1 -name "3D_SPK_*" -exec basename {} \; 2>/dev/null
    done | sort -u > "$TEST_SPKS"
    n=$(wc -l < "$TEST_SPKS")
    echo "    $n test speakers → $TEST_SPKS"
else
    echo "  test_speakers.txt 已存在 ($(wc -l < $TEST_SPKS) speakers)"
fi

# musan_wav.scp — 仅使用训练噪声池（排除测试噪声）
NOISE_SCP="$DATA_DIR/musan_wav.scp"
NOISE_TRAIN_LIST="$SV_ECAPA/data/divide/noise_train_freesound.txt"
NOISE_TEST_LIST="$SV_ECAPA/data/divide/noise_test_freesound.txt"
if [ ! -f "$NOISE_SCP" ]; then
    echo "  生成 musan_wav.scp (仅训练噪声池，排除测试噪声)..."
    if [ -f "$NOISE_TRAIN_LIST" ]; then
        awk -v base="${SV_ECAPA}/" '{print $1, base $1}' "$NOISE_TRAIN_LIST" > "$NOISE_SCP"
    else
        echo "    [WARN] $NOISE_TRAIN_LIST 不存在，回退到全量 MUSAN"
        find "$MUSAN_DIR" -name "*.wav" | \
            awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF, $0}' > "$NOISE_SCP"
    fi
    echo "    $(wc -l < $NOISE_SCP) 条 → $NOISE_SCP"
else
    echo "  musan_wav.scp 已存在 ($(wc -l < $NOISE_SCP) 条)"
fi

# 噪声泄漏检查
if [ -f "$NOISE_TEST_LIST" ] && [ -f "$NOISE_TRAIN_LIST" ]; then
    echo "  噪声泄漏检查:"
    python -c "
train = set(open('${NOISE_TRAIN_LIST}').read().splitlines())
test = set(open('${NOISE_TEST_LIST}').read().splitlines())
overlap = train & test
print(f'    Train noise: {len(train)}, Test noise: {len(test)}, Overlap: {len(overlap)}')
if overlap:
    print(f'    [ERROR] 噪声泄漏! {list(overlap)[:5]}')
    import sys; sys.exit(1)
else:
    print('    [OK] 训练/测试噪声完全隔离')
"
fi

# rirs_wav.scp (仅 mediumroom)
REVERB_SCP="$DATA_DIR/rirs_mediumroom.scp"
RIR_LIST="$RIRS_DIR/rir_list"
if [ ! -f "$REVERB_SCP" ]; then
    echo "  生成 rirs_mediumroom.scp ..."
    if [ -f "$RIR_LIST" ]; then
        awk -v base="${SV_ECAPA}/data/rirs_noises" '{print $5, base"/"$5}' "$RIR_LIST" > "$REVERB_SCP"
        echo "    $(wc -l < $REVERB_SCP) 条 → $REVERB_SCP"
    else
        echo "    [WARN] $RIR_LIST 不存在，RIR 增强将不可用"
        echo "" > "$REVERB_SCP"
    fi
else
    echo "  rirs_mediumroom.scp 已存在 ($(wc -l < $REVERB_SCP) 条)"
fi

# ============================================================
# Step 1: 数据准备
# ============================================================
echo ""
echo "============================================"
echo "  Step 1/3: 准备训练数据"
echo "============================================"

NUM_UTTS=120000
if $DEBUG; then
    NUM_UTTS=2000
fi

python local/prepare_finetune_data.py \
    --train_dir "$TRAIN_DIR" \
    --test_speakers_file "$TEST_SPKS" \
    --output_dir "$DATA_DIR" \
    --num_utts $NUM_UTTS \
    --seed $seed

# ============================================================
# Step 2: 验证 overlap 池
# ============================================================
echo ""
echo "============================================"
echo "  Step 2/3: 验证 overlap 干扰池"
echo "============================================"
python -c "
import json
with open('${DATA_DIR}/train_spk2utt.json') as f:
    spk2utt = json.load(f)
speakers = list(spk2utt.keys())
n_utts = sum(len(v) for v in spk2utt.values())
print(f'  Speakers: {len(speakers)}')
print(f'  Utterances: {n_utts}')
print(f'  Avg utt/speaker: {n_utts/len(speakers):.1f}')

# 检查与测试集重叠
with open('${TEST_SPKS}') as f:
    test_spks = set(l.strip() for l in f if l.strip())
overlap = set(speakers) & test_spks
if overlap:
    print(f'  [ERROR] Train={len(speakers)} Test={len(test_spks)} Overlap={len(overlap)}')
    import sys; sys.exit(1)
print(f'  [OK] No overlap with test ({len(test_spks)} test speakers)')
"

# ============================================================
# Step 3: 训练
# ============================================================256128
echo ""
echo "============================================"
echo "  Step 3/3: 启动训练 (${num_gpu}×P100)"
echo "============================================"
echo "  Model:    ERes2Net-large (22.46M, dim=512)"
echo "  Frozen:   conv1/bn1/layer1/layer2"
echo "  Train:    layer3/layer4/fusion/pooling/embedding/classifier"
echo "  Data:     $NUM_UTTS utterances"
echo "  Epochs:   $EPOCHS (debug: 1)"
echo "  Batch:    256 (64/GPU)"
echo ""

mkdir -p "$EXP_DIR"

EPOCHS=30
SAVE_FREQ=1
if $DEBUG; then
    EPOCHS=1
    SAVE_FREQ=1
fi

torchrun --nproc_per_node=$num_gpu \
    ../../../speakerlab/bin/train.py \
    --config conf/eres2net_finetune.yaml \
    --gpu $gpus \
    --pretrained "$PRETRAINED" \
    --seed $seed \
    --data "${DATA_DIR}/train.csv" \
    --noise "$NOISE_SCP" \
    --reverb "$REVERB_SCP" \
    --overlap_spk2utt "${DATA_DIR}/train_spk2utt.json" \
    --valid_data "${DATA_DIR}/valid.csv" \
    --exp_dir "$EXP_DIR" \
    --num_epoch $EPOCHS \
    --save_epoch_freq $SAVE_FREQ

echo ""
echo "============================================"
echo "  训练完成！"
echo "============================================"
echo "  模型: ${EXP_DIR}/models/"
echo "  日志: ${EXP_DIR}/train.log"
echo "  下一步:"
echo "    cp ${EXP_DIR}/models/CKPT-EPOCH-${EPOCHS}-00/embedding_model.ckpt 到测试服务器"
echo "    用 evaluate_clean.sh / evaluate_noise.sh / evaluate_overlap.sh 评测"
