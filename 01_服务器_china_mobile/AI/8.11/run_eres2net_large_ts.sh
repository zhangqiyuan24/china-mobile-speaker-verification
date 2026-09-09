#!/bin/bash
# ============================================================
# ERes2Net-large Teacher-Student 鲁棒微调 — 一键脚本
#
# 4×P100 12GB: batch_size=128 (32/gpu, Teacher 模型额外占用 ~90MB)
# 训练量: ~120K utterances × 15 epochs
#
# debug 模式 (少量数据):
#   bash run_eres2net_large_ts.sh --debug
#
# 用法: bash run_eres2net_large_ts.sh
# ============================================================
set -e
. ./path.sh || exit 1

# ---- 配置 ----
gpus="0 1 2 3"
num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
seed=2026

# ---- 数据路径 (相对于 sv-eres2net/) ----
SV_ECAPA="../sv-ecapa"
TRAIN_DIR="${SV_ECAPA}/data/data_for_train/train"
ORI_DIR="${SV_ECAPA}/data/data_for_test/ori"
MUSAN_DIR="${SV_ECAPA}/data/musan"
RIRS_DIR="${SV_ECAPA}/data/rirs_noises/RIRS_NOISES/simulated_rirs/mediumroom"
PRETRAINED="../../../pretrained/speech_eres2net_large_sv_zh-cn_3dspeaker_16k/eres2net_large_model.ckpt"
DATA_DIR="data"
EXP_DIR="exp/eres2net_finetune_ts"

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

mkdir -p "$DATA_DIR"
TEST_SPKS="$DATA_DIR/test_speakers.txt"
if [ ! -f "$TEST_SPKS" ]; then
    echo "  生成 test_speakers.txt ..."
    for r in A B C D E F; do
        find "$ORI_DIR/$r" -maxdepth 1 -name "3D_SPK_*" -exec basename {} \; 2>/dev/null
    done | sort -u > "$TEST_SPKS"
    n=$(wc -l < "$TEST_SPKS")
    echo "    $n test speakers -> $TEST_SPKS"
else
    echo "  test_speakers.txt 已存在 ($(wc -l < $TEST_SPKS) speakers)"
fi

# ============================================================
# musan_wav.scp + noise train/valid split
# ============================================================
NOISE_SCP="$DATA_DIR/musan_wav.scp"
NOISE_VALID_SCP="$DATA_DIR/musan_wav_valid.scp"
NOISE_TRAIN_LIST="$SV_ECAPA/data/divide/noise_train_freesound.txt"
NOISE_TEST_LIST="$SV_ECAPA/data/divide/noise_test_freesound.txt"

if [ ! -f "$NOISE_TRAIN_LIST" ]; then
    echo "[ERROR] $NOISE_TRAIN_LIST 不存在！拒绝回退到全量 MUSAN（防止噪声泄漏）。"
    echo "  请确认 sv-ecapa/data/divide/noise_train_freesound.txt 存在。"
    exit 1
fi

if [ ! -f "$NOISE_SCP" ]; then
    echo "  噪声池划分: 90% train / 10% valid (seed=2026)..."
    python -c "
import random
random.seed(2026)
lines = open('${NOISE_TRAIN_LIST}').read().splitlines()
random.shuffle(lines)
n_valid = max(1, int(len(lines) * 0.1))
valid_lines = lines[:n_valid]
train_lines = lines[n_valid:]
open('${DATA_DIR}/noise_train.txt', 'w').write('\n'.join(train_lines))
open('${DATA_DIR}/noise_valid.txt', 'w').write('\n'.join(valid_lines))
print(f'  Train noise: {len(train_lines)} 条')
print(f'  Valid noise: {len(valid_lines)} 条')
"
    echo "  生成 musan_wav.scp (train noise only)..."
    awk -v base="${SV_ECAPA}/" '{print $1, base $1}' "${DATA_DIR}/noise_train.txt" > "$NOISE_SCP"
    echo "  生成 musan_wav_valid.scp (valid noise only)..."
    awk -v base="${SV_ECAPA}/" '{print $1, base $1}' "${DATA_DIR}/noise_valid.txt" > "$NOISE_VALID_SCP"
    echo "  $(wc -l < $NOISE_SCP) 条 -> $NOISE_SCP"
    echo "  $(wc -l < $NOISE_VALID_SCP) 条 -> $NOISE_VALID_SCP"
else
    echo "  musan_wav.scp 已存在 ($(wc -l < $NOISE_SCP) 条)"
    if [ ! -f "$NOISE_VALID_SCP" ]; then
        echo "  [WARN] musan_wav_valid.scp 不存在，请删除 musan_wav.scp 重新生成"
    fi
fi

# 噪声泄漏检查 (train / valid / test 三者隔离)
if [ -f "$NOISE_TEST_LIST" ]; then
    echo "  噪声泄漏检查 (train/valid/test):"
    python -c "
train = set(open('${DATA_DIR}/noise_train.txt').read().splitlines())
valid = set(open('${DATA_DIR}/noise_valid.txt').read().splitlines())
test  = set(open('${NOISE_TEST_LIST}').read().splitlines())
print(f'  Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}')
for a_name, a_set in [('train&valid', train & valid), ('train&test', train & test), ('valid&test', valid & test)]:
    if a_set:
        print(f'  [ERROR] 噪声泄漏: {a_name} 重叠 {len(a_set)} 条!')
        import sys; sys.exit(1)
print('  [OK] Train/Valid/Test 噪声完全隔离')
"
fi

# rirs_wav.scp (仅 mediumroom)
REVERB_SCP="$DATA_DIR/rirs_mediumroom.scp"
RIR_LIST="$RIRS_DIR/rir_list"
if [ ! -f "$REVERB_SCP" ]; then
    echo "  生成 rirs_mediumroom.scp ..."
    if [ -f "$RIR_LIST" ]; then
        awk -v base="${SV_ECAPA}/data/rirs_noises" '{print $5, base"/"$5}' "$RIR_LIST" > "$REVERB_SCP"
        echo "    $(wc -l < $REVERB_SCP) 条 -> $REVERB_SCP"
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
# Step 2: 验证数据隔离 + 生成 valid_spk2utt
# ============================================================
echo ""
echo "============================================"
echo "  Step 2/4: 验证数据隔离 + valid_spk2utt"
echo "============================================"

# 生成 valid_spk2utt.json (用于 overlap validation 干扰池)
VALID_SPK2UTT="$DATA_DIR/valid_spk2utt.json"
echo "  生成 valid_spk2utt.json ..."
python -c "
import json, csv
from collections import defaultdict
spk2utt = defaultdict(list)
with open('${DATA_DIR}/valid.csv') as f:
    for row in csv.DictReader(f):
        spk2utt[row['spk']].append(row['path'])
with open('${VALID_SPK2UTT}', 'w') as f:
    json.dump(dict(spk2utt), f)
print(f'  {len(spk2utt)} valid speakers -> ${VALID_SPK2UTT}')
"

echo ""
echo "  数据隔离检查 (train/valid/test speaker):"
python -c "
import json
with open('${DATA_DIR}/train_spk2utt.json') as f:
    train_spks = set(json.load(f).keys())
with open('${VALID_SPK2UTT}') as f:
    valid_spks = set(json.load(f).keys())
with open('${TEST_SPKS}') as f:
    test_spks = set(l.strip() for l in f if l.strip())

print(f'  Train: {len(train_spks)}, Valid: {len(valid_spks)}, Test: {len(test_spks)}')
for a_name, a_set in [('train&valid', train_spks & valid_spks),
                       ('train&test', train_spks & test_spks),
                       ('valid&test', valid_spks & test_spks)]:
    if a_set:
        print(f'  [ERROR] Speaker 泄漏: {a_name} 重叠 {len(a_set)}!')
        import sys; sys.exit(1)
print('  [OK] Train/Valid/Test speaker 完全隔离')
"

# ============================================================
# Step 3: 训练
# ============================================================
EPOCHS=15
LAMBDA_CONS=0.25
SAVE_FREQ=1
if $DEBUG; then
    EPOCHS=1
    SAVE_FREQ=1
fi

echo ""
echo "============================================"
echo "  Step 3/4: Teacher-Student 训练 (${num_gpu}×P100)"
echo "============================================"
echo "  Model:    ERes2Net-large (22.46M, dim=512)"
echo "  Teacher:  全冻结 (原始 pretrained)"
echo "  Student:  冻结 conv1/bn1/layer1/layer2/layer3"
echo "  Train:    layer4/fusion/pooling/embedding/classifier"
echo "  Data:     $NUM_UTTS utterances"
echo "  Epochs:   $EPOCHS (debug: 1)"
echo "  Batch:    128 (32/GPU)"
echo "  Loss:     L_arcface + ${LAMBDA_CONS} * L_cons"
echo ""

mkdir -p "$EXP_DIR"

torchrun --nproc_per_node=$num_gpu \
    ../../../speakerlab/bin/train_ts.py \
    --config conf/eres2net_finetune_ts.yaml \
    --gpu $gpus \
    --pretrained "$PRETRAINED" \
    --seed $seed \
    --data "${DATA_DIR}/train.csv" \
    --noise "$NOISE_SCP" \
    --reverb "$REVERB_SCP" \
    --overlap_spk2utt "${DATA_DIR}/train_spk2utt.json" \
    --valid_data "${DATA_DIR}/valid.csv" \
    --valid_noise "${DATA_DIR}/musan_wav_valid.scp" \
    --valid_spk2utt "${DATA_DIR}/valid_spk2utt.json" \
    --exp_dir "$EXP_DIR" \
    --num_epoch $EPOCHS \
    --save_epoch_freq $SAVE_FREQ

echo ""
echo "============================================"
echo "  训练完成！"
echo "============================================"
echo "  模型:"
echo "    best_clean:  ${EXP_DIR}/models/best_clean/embedding_model.ckpt"
echo "    best_robust: ${EXP_DIR}/models/best_robust/embedding_model.ckpt"
echo "  日志: ${EXP_DIR}/train.log"
echo "  评测:"
echo "    bash evaluate_all_finetune.sh (修改 FINETUNED_CKPT 指向需要的模型)"
