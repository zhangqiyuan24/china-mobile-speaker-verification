#!/bin/bash
# Fine-tune ECAPA-TDNN from a ModelScope pretrained checkpoint.
#
# Usage:
#   1. First, download & prepare data:
#      bash run.sh --stage 1 --stop_stage 2
#   2. Then fine-tune:
#      bash run_finetune.sh --pretrained /path/to/ecapa-tdnn.ckpt
#   3. Extract embeddings + evaluate:
#      bash run_finetune.sh --stage 4 --stop_stage 5 --pretrained /path/to/ecapa-tdnn.ckpt
#
# The pretrained checkpoint path defaults to the ModelScope download location:
#   ../../../pretrained/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k/ecapa-tdnn.ckpt

set -e
. ./path.sh || exit 1

stage=3
stop_stage=5

data=data
exp=exp
exp_name=ecapa_tdnn_finetune
gpus="0"

# ---- pretrained checkpoint path ----
PRETRAINED_DEFAULT="../../../pretrained/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k/ecapa-tdnn.ckpt"
pretrained=${pretrained:-$PRETRAINED_DEFAULT}

. utils/parse_options.sh || exit 1

exp_dir=$exp/$exp_name

if [ ! -f "$pretrained" ]; then
  echo "ERROR: Pretrained checkpoint not found at $pretrained"
  echo "Make sure you have run 'python speakerlab/bin/infer_sv.py --model_id iic/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k --wavs some.wav' at least once to download the model."
  exit 1
fi

echo "Using pretrained checkpoint: $pretrained"
echo "Experiment directory: $exp_dir"

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage3: Fine-tuning ECAPA-TDNN from pretrained weights..."
  num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
  torchrun --nproc_per_node=$num_gpu speakerlab/bin/train.py \
    --config conf/ecapa_tdnn.yaml \
    --gpu $gpus \
    --pretrained "$pretrained" \
    --data $data/3dspeaker/train/train.csv \
    --noise $data/musan/wav.scp \
    --reverb $data/rirs/wav.scp \
    --exp_dir $exp_dir
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage4: Extracting speaker embeddings..."
  torchrun --nproc_per_node=8 speakerlab/bin/extract.py \
    --exp_dir $exp_dir \
    --data $data/3dspeaker/test/wav.scp \
    --use_gpu --gpu $gpus
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage5: Computing score metrics..."
  trials="$data/3dspeaker/trials/trials_cross_device $data/3dspeaker/trials/trials_cross_distance $data/3dspeaker/trials/trials_cross_dialect"
  python speakerlab/bin/compute_score_metrics.py \
    --enrol_data $exp_dir/embeddings \
    --test_data $exp_dir/embeddings \
    --scores_dir $exp_dir/scores \
    --trials $trials
fi
