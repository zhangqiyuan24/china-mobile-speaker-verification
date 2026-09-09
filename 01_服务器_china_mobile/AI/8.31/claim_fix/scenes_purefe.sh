#!/bin/bash
# scenes_purefe.sh — #43 通用前端（无条件 FullSubNet+ e10）场景补测（GPU $1，默认 0）
# 主张：overlap 场景纯前端≈raw（无判别信息压不掉人声）→ 与 r5 FE 形成原理性差距
#      = "TSE 应用价值"三柱论证第二柱的空白格。
# 协议与 launch_scenes_r5.sh / fix_scenes_overlap_r5.sh 完全一致：
#   ENROL=clean_bench/eres2net/embeddings_3s, TRIALS=clean/3s/trials_cross_device,
#   MID=eres2net 官方模型；raw/r5 参照行末尾回显。
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
SV=/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
OUT=$AI/8.28/scenes_purefe
FE_ROOT=$OUT/enhanced
MID=iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k
ENROL=$SV/exp/clean_bench/eres2net/embeddings_3s
TRIALS=$SV/data/data_for_test/clean/3s/trials_cross_device
RAW_REF=$AI/8.27/exp/official_scenes_tse_v3_ns05ext2/eres2net
R5_REF=$AI/8.28/scenes_r5/eres2net
GPU=${1:-0}
mkdir -p $OUT
cd $SV
source path.sh
export CUDA_VISIBLE_DEVICES=$GPU

for SCENE in overlap2_cn overlap2_en overlap3_cn overlap3_en reverb; do
  D=$OUT/eres2net/purefe_${SCENE}_3s
  if [ -s "$D/result.metrics" ] && grep -q EER "$D/result.metrics"; then
    echo "[skip] $SCENE"; continue
  fi
  echo "===== purefe $SCENE ($(date)) ====="
  $PY $AI/8.31/claim_fix/enhance_purefe.py \
      --wav-list $SV/data/data_for_test/$SCENE/3s/wav_list.txt \
      --out-root $FE_ROOT/$SCENE --gpu 0 --batch-size 8 >> $OUT/enhance_purefe.log 2>&1
  find $FE_ROOT/$SCENE -name '*.wav' | sort > $OUT/purefe_${SCENE}_wavlist.txt
  $PY ../../../speakerlab/bin/infer_sv_batch.py --model_id $MID \
      --wavs $OUT/purefe_${SCENE}_wavlist.txt \
      --feat_out_dir $D --feat_out_format ark >> $OUT/infer_purefe.log 2>&1
  mkdir -p $D
  $PY ../../../speakerlab/bin/compute_score_metrics.py \
      --enrol_data $ENROL --test_data $D --scores_dir $D \
      --trials $TRIALS >> $OUT/score_purefe.log 2>&1
  RAWV=$(grep -m1 EER $R5_REF/raw_${SCENE}_3s/result.metrics 2>/dev/null)
  [ -z "$RAWV" ] && RAWV=$(grep -m1 EER $RAW_REF/raw_${SCENE}_3s/result.metrics 2>/dev/null)
  echo "[$SCENE] raw   : $RAWV"
  echo "[$SCENE] purefe: $(grep -m1 EER $D/result.metrics 2>/dev/null)"
  echo "[$SCENE] r5    : $(grep -m1 EER $R5_REF/fe_${SCENE}_3s/result.metrics 2>/dev/null)"
done
echo "[purefe-scenes done] $(date)"
