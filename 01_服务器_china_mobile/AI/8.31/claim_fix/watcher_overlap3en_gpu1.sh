#!/bin/bash
# watcher_overlap3en_gpu1.sh — 等 overlap3_en 增强(1912590)完成后接 infer+score（GPU1）
# 主循环已杀（防 reverb 重跑竞态），本脚本补它没做完的 overlap3_en 收尾。
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
SCENE=overlap3_en
D=$OUT/eres2net/fe_${SCENE}_3s

echo "[watcher start] 等 1912590 退出 $(date)"
while kill -0 1912590 2>/dev/null; do sleep 30; done
echo "[watcher] enhance 已退出 $(date)"

find $FE_ROOT/$SCENE/$SCENE -name '*.wav' | sort > $OUT/fe_${SCENE}_wavlist.txt
N=$(wc -l < $OUT/fe_${SCENE}_wavlist.txt)
echo "[$SCENE] wavlist=$N（应 16401）"
if [ "$N" -lt 16000 ]; then
  echo "[FATAL] $SCENE 产物不全（enhance 可能 OOM），需人工重跑"
  exit 1
fi

cd $SV
source path.sh
export CUDA_VISIBLE_DEVICES=1
rm -f $D/result.metrics
$PY ../../../speakerlab/bin/infer_sv_batch.py --model_id $MID \
    --wavs $OUT/fe_${SCENE}_wavlist.txt \
    --feat_out_dir $D --feat_out_format ark >> $OUT/infer_v8d_fix.log 2>&1
mkdir -p $D
$PY ../../../speakerlab/bin/compute_score_metrics.py \
    --enrol_data $ENROL --test_data $D --scores_dir $D \
    --trials $TRIALS >> $OUT/score_v8d_fix.log 2>&1
echo "[$SCENE] v8d  : $(grep -m1 EER $D/result.metrics 2>/dev/null)"
echo "[$SCENE] puref: $(grep -m1 EER $PF_REF/purefe_${SCENE}_3s/result.metrics 2>/dev/null)"
echo "[$SCENE] r5   : $(grep -m1 EER $R5_REF/fe_${SCENE}_3s/result.metrics 2>/dev/null)"
echo "[watcher done] $(date)"
