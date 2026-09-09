#!/bin/bash
# launch_scenes_r5.sh — #21 场景分项 r5 版（GPU0，8.30）
# 前端 = duraug_r5（交付 ckpt）；raw 基线沿用 ext2 场景轮官方口径（前端无关，不重跑）。
# 守卫：FE 格非空且含 EER 才跳过；raw 基线缺格的场景整格跳过。
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
SV=/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
R5=$AI/8.27/exp/tse_v3_ns05duraug_r5
FE_ROOT=$R5/enhanced_official_scenes
OUT=$AI/8.28/scenes_r5
MID=iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k
RAW_REF=$AI/8.27/exp/official_scenes_tse_v3_ns05ext2/eres2net
mkdir -p $OUT

cd $SV
source path.sh
export CUDA_VISIBLE_DEVICES=0

for SCENE in reverb overlap2_cn overlap2_en overlap3_cn overlap3_en; do
  if ! { [ -s "$RAW_REF/raw_${SCENE}_3s/result.metrics" ] && grep -q EER "$RAW_REF/raw_${SCENE}_3s/result.metrics"; }; then
    echo "[skip-no-raw] $SCENE"; continue
  fi
  D=$OUT/eres2net/fe_${SCENE}_3s
  if [ -s "$D/result.metrics" ] && grep -q EER "$D/result.metrics"; then
    echo "[skip] $SCENE"; continue
  fi
  echo "===== scene $SCENE ($(date)) ====="
  $PY $AI/8.27/enhance_tse_v2.py --config train_tse.toml --checkpoint $R5/checkpoint_best.tar \
      --gpu 0 --batch-size 8 --dur 3 --conds $SCENE \
      --out-root $FE_ROOT/$SCENE >> $OUT/enhance_scenes_r5.log 2>&1
  find $FE_ROOT/$SCENE/$SCENE -name '*.wav' | sort > $OUT/fe_${SCENE}_wavlist.txt
  $PY ../../../speakerlab/bin/infer_sv_batch.py --model_id $MID --wavs $OUT/fe_${SCENE}_wavlist.txt \
      --feat_out_dir $D --feat_out_format ark >> $OUT/infer_scenes_r5.log 2>&1
  mkdir -p $D
  $PY ../../../speakerlab/bin/compute_score_metrics.py \
      --enrol_data $SV/exp/clean_bench/eres2net/embeddings_3s --test_data $D \
      --scores_dir $D --trials $SV/data/data_for_test/clean/3s/trials_cross_device >> $OUT/score_scenes_r5.log 2>&1
  echo "[$SCENE] raw: $(grep -m1 EER $RAW_REF/raw_${SCENE}_3s/result.metrics)"
  echo "[$SCENE] fe : $(grep -m1 EER $D/result.metrics 2>/dev/null)"
done
echo "[scenes-r5 done] $(date)"
