#!/bin/bash
# scenes_v8d_reverb_gpu3.sh — v8d 场景表 reverb 单格提前解尾（GPU3，9.2 晚）
# GPU1 主循环（scenes_v8d.sh）顺序到 reverb 要 ~22:40；本脚本先跑出 result.metrics
# 后主循环自动 skip。OUT/FE_ROOT 与主循环同根，零协议漂移。
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
SV=/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
V8D=$AI/8.27/exp/tse_v8_v8d_sil
FE_ROOT=$V8D/enhanced_official_scenes
OUT=$AI/8.28/scenes_v8d
MID=iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k
ENROL=$SV/exp/clean_bench/eres2net/embeddings_3s
TRIALS=$SV/data/data_for_test/clean/3s/trials_cross_device
PF_REF=$AI/8.28/scenes_purefe/eres2net
R5_REF=$AI/8.28/scenes_r5/eres2net
SCENE=reverb
D=$OUT/eres2net/fe_${SCENE}_3s

if [ -s "$D/result.metrics" ] && grep -q EER "$D/result.metrics"; then
  echo "[skip] $SCENE"; exit 0
fi
echo "===== v8d $SCENE gpu3 ($(date)) ====="
cd $SV
source path.sh
export CUDA_VISIBLE_DEVICES=3

$PY $AI/8.27/enhance_tse_v2.py --config train_tse.toml --checkpoint $V8D/checkpoint_epoch1.tar \
    --gpu 0 --batch-size 8 --dur 3 --conds $SCENE \
    --out-root $FE_ROOT/$SCENE >> $OUT/enhance_v8d.log 2>&1
find $FE_ROOT/$SCENE/$SCENE -name '*.wav' | sort > $OUT/fe_${SCENE}_wavlist.txt
$PY ../../../speakerlab/bin/infer_sv_batch.py --model_id $MID \
    --wavs $OUT/fe_${SCENE}_wavlist.txt \
    --feat_out_dir $D --feat_out_format ark >> $OUT/infer_v8d.log 2>&1
mkdir -p $D
$PY ../../../speakerlab/bin/compute_score_metrics.py \
    --enrol_data $ENROL --test_data $D --scores_dir $D \
    --trials $TRIALS >> $OUT/score_v8d.log 2>&1
echo "[$SCENE] v8d  : $(grep -m1 EER $D/result.metrics 2>/dev/null)"
echo "[$SCENE] puref: $(grep -m1 EER $PF_REF/purefe_${SCENE}_3s/result.metrics 2>/dev/null)"
echo "[$SCENE] r5   : $(grep -m1 EER $R5_REF/fe_${SCENE}_3s/result.metrics 2>/dev/null)"
echo "[reverb-gpu3 done] $(date)"
