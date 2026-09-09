#!/bin/bash
# launch_noise_domains_raw.sh — #26 噪声域泛化·前置半场（8.28）
# 阶段1(CPU): 生成 music/speech/soundbible × {0,-5}dB × 3s 换域测试集（镜像官方协议）
# 阶段2(GPU0): raw 侧基线（官方 ERes2Net 管线，与 ckpt 无关）
# FE 侧等终评 ckpt 定了另跑。不碰 data_for_test，只写 AI/8.28。
# 8.28 修复：原 set -u 撞 path.sh 未定义 PYTHONPATH 致 stage2 秒挂；stage1 加跳过守卫（生成完不重跑）
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
SV=/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
ROOT=$AI/8.28/noise_domains
MID=iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k
ENROL=$SV/exp/clean_bench/eres2net/embeddings_3s
TRIALS=$SV/data/data_for_test/clean/3s/trials_cross_device

if [ -s $ROOT/soundbible_m5db/3s/wav_list.txt ]; then
  echo "[stage1 skip] 6 份清单已存在，直接进 stage2"
else
  echo "[stage1] 生成换域测试集 $(date)"
  $PY $AI/8.28/gen_noise_domains.py --domains music,speech,soundbible --out $ROOT
  echo "[stage1 done] $(date)"
fi

echo "[stage2] raw 基线 GPU0 $(date)"
cd $SV
source path.sh
export CUDA_VISIBLE_DEVICES=0

for COND in music_0db music_m5db speech_0db speech_m5db soundbible_0db soundbible_m5db; do
  WAVS=$ROOT/$COND/3s/wav_list.txt
  if [ ! -s "$WAVS" ]; then echo "[skip] $WAVS 不存在"; continue; fi
  OUT=$ROOT/eval_raw/$COND
  mkdir -p $OUT
  echo "--- raw $COND ---"
  $PY ../../../speakerlab/bin/infer_sv_batch.py --model_id $MID --wavs $WAVS \
      --feat_out_dir $OUT --feat_out_format ark >> $ROOT/eval_raw/infer.log 2>&1
  $PY ../../../speakerlab/bin/compute_score_metrics.py \
      --enrol_data $ENROL --test_data $OUT --scores_dir $OUT --trials $TRIALS \
      >> $ROOT/eval_raw/score.log 2>&1
  echo "[$COND] $(grep EER $OUT/result.metrics 2>/dev/null | head -1)"
done
echo "[all done] $(date)  raw 结果: $ROOT/eval_raw"
