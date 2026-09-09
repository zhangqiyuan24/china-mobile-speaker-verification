#!/bin/bash
# launch_noise_domains_fe.sh — #26 噪声域泛化·FE 侧（8.28 晚，ext3 ckpt，GPU0）
# 对 6 个换域测试集跑 TSE 前端增强 → 官方 ERes2Net 管线评测，与 eval_raw 同口径对照。
# 增强产物落 exp/tse_v3_ns05ext3/enhanced_noise_domains/<cond>/（断点续跑）。
# 注：这是"当前官方最优 ckpt"的换域答案；dur-aug 收敛后终版大表另跑。
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
SV=/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
ROOT=$AI/8.28/noise_domains
CKPT=$AI/8.27/exp/tse_v3_ns05ext3/checkpoint_best.tar
FE_ROOT=$AI/8.27/exp/tse_v3_ns05ext3/enhanced_noise_domains
OUT=$ROOT/eval_fe
MID=iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k
ENROL=$SV/exp/clean_bench/eres2net/embeddings_3s
TRIALS=$SV/data/data_for_test/clean/3s/trials_cross_device
mkdir -p $OUT

echo "[stage-fe1] 增强六条件 GPU0 $(date)"
$PY $AI/8.27/enhance_tse_v2.py --config train_tse.toml --checkpoint $CKPT \
    --gpu 0 --batch-size 8 --dur 3 --test-root $ROOT \
    --conds music_0db,music_m5db,speech_0db,speech_m5db,soundbible_0db,soundbible_m5db \
    --out-root $FE_ROOT >> $ROOT/fe_enhance.log 2>&1
echo "[stage-fe1 done] $(date)"

echo "[stage-fe2] FE 评测 GPU0 $(date)"
cd $SV
source path.sh
export CUDA_VISIBLE_DEVICES=0

for COND in music_0db music_m5db speech_0db speech_m5db soundbible_0db soundbible_m5db; do
  WAVS=$FE_ROOT/$COND
  if [ ! -d "$WAVS" ] || [ -z "$(find $WAVS -name '*.wav' 2>/dev/null | head -1)" ]; then
    echo "[skip] $WAVS 无增强产物"; continue
  fi
  find $WAVS -name '*.wav' | sort > $OUT/fe_${COND}_wavlist.txt
  echo "--- fe $COND ---"
  $PY ../../../speakerlab/bin/infer_sv_batch.py --model_id $MID --wavs $OUT/fe_${COND}_wavlist.txt \
      --feat_out_dir $OUT/$COND --feat_out_format ark >> $OUT/infer.log 2>&1
  mkdir -p $OUT/$COND
  $PY ../../../speakerlab/bin/compute_score_metrics.py \
      --enrol_data $ENROL --test_data $OUT/$COND --scores_dir $OUT/$COND --trials $TRIALS \
      >> $OUT/score.log 2>&1
  echo "[$COND fe] $(grep EER $OUT/$COND/result.metrics 2>/dev/null | head -1)"
done
echo "[all done] $(date)  FE 结果: $OUT"
