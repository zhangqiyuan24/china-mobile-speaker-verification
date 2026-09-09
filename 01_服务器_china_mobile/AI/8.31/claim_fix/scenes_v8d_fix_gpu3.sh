#!/bin/bash
# scenes_v8d_fix_gpu3.sh — 修复 OOM 双格补跑（GPU3，9.2 晚）
# 事故根因：enhance_tse_v2.py 的 --gpu 是【物理卡号】语义（内部覆盖外层
# CUDA_VISIBLE_DEVICES）。scenes_v8d.sh 用 CVD=1+--gpu 0 实际落在物理 0，
# 与 GPU0 大表 enhance 叠罗汉 → overlap3_cn/reverb OOM（产物 2737/16401）。
# 本脚本：--gpu 3 直指物理 3（空卡），串行补 overlap3_cn → reverb 两格。
# reverb 抢在主循环到达前出 result.metrics → 主循环自动 skip（主循环 reverb
# 若自己跑会再次 --gpu 0 落物理 0 与 5s enhance 挤爆）。
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
mkdir -p $OUT

# 先清 OOM 半成品（enhance 断点续跑会基于已有文件计数，残缺 2737 条会污染）
for S in overlap3_cn reverb; do
  D=$OUT/eres2net/fe_${S}_3s
  rm -f $D/result.metrics
done

cd $SV
source path.sh
export CUDA_VISIBLE_DEVICES=3

for SCENE in overlap3_cn reverb; do
  D=$OUT/eres2net/fe_${SCENE}_3s
  if [ -s "$D/result.metrics" ] && grep -q EER "$D/result.metrics"; then
    echo "[skip] $SCENE"; continue
  fi
  echo "===== fix $SCENE gpu3 ($(date)) ====="
  # 清该场景残缺增强产物，强制全量重做（避免 2737/16401 半成品）
  rm -rf $FE_ROOT/$SCENE/$SCENE
  $PY $AI/8.27/enhance_tse_v2.py --config train_tse.toml --checkpoint $V8D/checkpoint_epoch1.tar \
      --gpu 3 --batch-size 8 --dur 3 --conds $SCENE \
      --out-root $FE_ROOT/$SCENE >> $OUT/enhance_v8d_fix.log 2>&1
  find $FE_ROOT/$SCENE/$SCENE -name '*.wav' | sort > $OUT/fe_${SCENE}_wavlist.txt
  N=$(wc -l < $OUT/fe_${SCENE}_wavlist.txt)
  echo "[$SCENE] wavlist=$N（应 16401）"
  if [ "$N" -lt 16000 ]; then echo "[FATAL] $SCENE 产物不全，中止本格"; continue; fi
  rm -rf $D; mkdir -p $D
  $PY ../../../speakerlab/bin/infer_sv_batch.py --model_id $MID \
      --wavs $OUT/fe_${SCENE}_wavlist.txt \
      --feat_out_dir $D --feat_out_format ark >> $OUT/infer_v8d_fix.log 2>&1
  $PY ../../../speakerlab/bin/compute_score_metrics.py \
      --enrol_data $ENROL --test_data $D --scores_dir $D \
      --trials $TRIALS >> $OUT/score_v8d_fix.log 2>&1
  echo "[$SCENE] v8d  : $(grep -m1 EER $D/result.metrics 2>/dev/null)"
  echo "[$SCENE] puref: $(grep -m1 EER $PF_REF/purefe_${SCENE}_3s/result.metrics 2>/dev/null)"
  echo "[$SCENE] r5   : $(grep -m1 EER $R5_REF/fe_${SCENE}_3s/result.metrics 2>/dev/null)"
done
echo "[fix-gpu3 done] $(date)"
