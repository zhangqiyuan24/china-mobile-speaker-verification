#!/bin/bash
# eval_1s5s_pipeline.sh — #19 1s/5s 时长泛化（官方管线，8.28 晨备好待布防）
# 协议：enrol=官方 clean_{dur}s 嵌入（exp/clean_bench 现成），test={cond}_{dur}s，
#       trials=clean/{dur}s/trials_cross_device，scorer=官方 compute_score_metrics。
# 模式：raw（基线对照）+ FE（enhance_tse_v2 --dur 产物）。后端=ERes2Net 官方 ckpt。
# 用法：CKPT=<前端ckpt绝对路径> GPU=<0|1|3> bash eval_1s5s_pipeline.sh
set -e
SV=/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
CKPT=${CKPT:?需指定前端 ckpt}
FE_TAG=$(basename $(dirname $CKPT))          # 如 tse_v3_ns05ext3
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU

OUT=$AI/8.27/exp/official_1s5s_$FE_TAG
FE_ROOT=$AI/8.27/exp/$FE_TAG/enhanced_official_1s5s
MID=iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k

cd $SV
source path.sh
mkdir -p $OUT

# ---- 第一步：增强 1s/5s 测试集（断点续跑，两时长四批）----
for dur in 1 5; do
  echo "--- enhance ${dur}s ---"
  $PY $AI/8.27/enhance_tse_v2.py \
      --config train_tse.toml --checkpoint $CKPT \
      --gpu $GPU --batch-size 8 --dur $dur \
      --out-root $FE_ROOT/${dur}s \
      >> $AI/8.27/enhance_1s5s_$FE_TAG.log 2>&1
done

# ---- 第二步：官方管线评测 raw + FE ----
for dur in 1 5; do
  CLEAN_EMB=$SV/exp/clean_bench/eres2net/embeddings_${dur}s
  TRIALS=$SV/data/data_for_test/clean/${dur}s/trials_cross_device
  for cond in noise_0db noise_m5db; do
    # raw
    echo "--- eres2net raw $cond ${dur}s ---"
    $PY ../../../speakerlab/bin/infer_sv_batch.py \
        --model_id $MID --wavs $SV/data/data_for_test/$cond/${dur}s/wav_list.txt \
        --feat_out_dir $OUT/eres2net/raw_${cond}_${dur}s --feat_out_format ark
    mkdir -p $OUT/eres2net/raw_${cond}_${dur}s
    $PY ../../../speakerlab/bin/compute_score_metrics.py \
        --enrol_data $CLEAN_EMB --test_data $OUT/eres2net/raw_${cond}_${dur}s \
        --scores_dir $OUT/eres2net/raw_${cond}_${dur}s --trials $TRIALS
    # fe
    echo "--- eres2net fe $cond ${dur}s ---"
    find $FE_ROOT/${dur}s/$cond -name '*.wav' | sort > $OUT/fe_${cond}_${dur}s_wavlist.txt
    $PY ../../../speakerlab/bin/infer_sv_batch.py \
        --model_id $MID --wavs $OUT/fe_${cond}_${dur}s_wavlist.txt \
        --feat_out_dir $OUT/eres2net/fe_${cond}_${dur}s --feat_out_format ark
    mkdir -p $OUT/eres2net/fe_${cond}_${dur}s
    $PY ../../../speakerlab/bin/compute_score_metrics.py \
        --enrol_data $CLEAN_EMB --test_data $OUT/eres2net/fe_${cond}_${dur}s \
        --scores_dir $OUT/eres2net/fe_${cond}_${dur}s --trials $TRIALS
  done
done

echo ""
echo "===== #19 1s/5s 官方管线汇总 ====="
for dur in 1 5; do
  for cond in noise_0db noise_m5db; do
    for mode in raw fe; do
      M=$OUT/eres2net/${mode}_${cond}_${dur}s/result.metrics
      [ -f "$M" ] && echo "eres2net $mode $cond ${dur}s: $(grep 'EER' $M | head -1)"
    done
  done
done
echo "[done] $OUT"
