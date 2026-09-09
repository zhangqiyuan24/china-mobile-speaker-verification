#!/bin/bash
# fix_scenes_overlap_r5.sh — 补齐 overlap 三场景 raw 基线 + r5 FE（GPU0，8.30 深夜）
# 起因：ext2 场景轮 infer 未设 CUDA_VISIBLE_DEVICES，多卡 worker 落到被占用的 GPU3 OOM，
#       raw overlap 三格留空壳 result.metrics。本脚本锁 CUDA_VISIBLE_DEVICES=0 重跑 raw + r5 FE。
# 守卫：非空且含 EER 才跳过；重跑前 rm 空壳。
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
SV=/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
R5=$AI/8.27/exp/tse_v3_ns05duraug_r5
FE_ROOT=$R5/enhanced_official_scenes
OUT=$AI/8.28/scenes_r5
MID=iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k
ENROL=$SV/exp/clean_bench/eres2net/embeddings_3s
TRIALS=$SV/data/data_for_test/clean/3s/trials_cross_device
mkdir -p $OUT
cd $SV
source path.sh
export CUDA_VISIBLE_DEVICES=0

ok () { [ -s "$1" ] && grep -q EER "$1"; }

for SCENE in overlap2_cn overlap2_en overlap3_cn; do
  for MODE in raw fe; do
    D=$OUT/eres2net/${MODE}_${SCENE}_3s
    if ok $D/result.metrics; then echo "[skip] $MODE $SCENE"; continue; fi
    rm -f $D/result.metrics
    if [ "$MODE" = "raw" ]; then
      WAVS=$SV/data/data_for_test/$SCENE/3s/wav_list.txt
    else
      $PY $AI/8.27/enhance_tse_v2.py --config train_tse.toml --checkpoint $R5/checkpoint_best.tar \
          --gpu 0 --batch-size 8 --dur 3 --conds $SCENE \
          --out-root $FE_ROOT/$SCENE >> $OUT/enhance_scenes_r5.log 2>&1
      find $FE_ROOT/$SCENE/$SCENE -name '*.wav' | sort > $OUT/fe_${SCENE}_wavlist.txt
      WAVS=$OUT/fe_${SCENE}_wavlist.txt
    fi
    $PY ../../../speakerlab/bin/infer_sv_batch.py --model_id $MID --wavs $WAVS \
        --feat_out_dir $D --feat_out_format ark >> $OUT/infer_scenes_r5.log 2>&1
    mkdir -p $D
    $PY ../../../speakerlab/bin/compute_score_metrics.py \
        --enrol_data $ENROL --test_data $D --scores_dir $D --trials $TRIALS >> $OUT/score_scenes_r5.log 2>&1
    echo "[$SCENE/$MODE] $(grep -m1 EER $D/result.metrics 2>/dev/null)"
  done
done
echo "[overlap-fix done] $(date)"
