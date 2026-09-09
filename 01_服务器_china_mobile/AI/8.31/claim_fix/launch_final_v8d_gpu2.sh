#!/bin/bash
# launch_final_v8d_gpu2.sh — v8d 终评大表·3s 轴三后端 + #26 换域 v8d 版（GPU2，9.2 晚）
# 3s 轴不重增强：复用官方链 enhanced_fe wav 清单（16401 条，与 r5 大表同口径）。
# 换域：music/speech/soundbible × 0db/m5db 六条件 eres（人声盲区 story 的 v8d 复测）。
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
SV=/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
GPU=2
V8D=$AI/8.27/exp/tse_v8_v8d_sil
OFF=$AI/8.31/claim_fix/official_v8d_sil
ND=$AI/8.28/noise_domains
OUT=$AI/8.28/final_eval_v8d
MID_ERES=iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k
MID_ECAPA=iic/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k
RD_ID='redimnet2/b6/vb2+vox2+cnc2_v0/lm'
RD_SUB=redimnet2_b6_vb2+vox2+cnc2_v0_lm/embeddings
RD_ENROL=$SV/exp/clean_bench/redimnet2_3s/$RD_SUB
EC_ENROL_3=$SV/exp/clean_bench/ecapa/embeddings_3s
TRIALS_3=$SV/data/data_for_test/clean/3s/trials_cross_device
mkdir -p $OUT

echo "[v8d-3s start] $(date)"
cd $SV
source path.sh
export CUDA_VISIBLE_DEVICES=$GPU

eval_std () {  # $1=name $2=mid $3=wavlist $4=enrol $5=trials
  local D=$OUT/$1
  [ -f "$D/result.metrics" ] && { echo "[skip] $1"; return; }
  $PY ../../../speakerlab/bin/infer_sv_batch.py --model_id "$2" --wavs "$3" \
      --feat_out_dir $D --feat_out_format ark >> $OUT/infer.log 2>&1
  mkdir -p $D
  $PY ../../../speakerlab/bin/compute_score_metrics.py \
      --enrol_data "$4" --test_data $D --scores_dir $D --trials "$5" >> $OUT/score.log 2>&1
  echo "[$1] $(grep EER $D/result.metrics 2>/dev/null | head -1)"
}
eval_b6 () {  # $1=name $2=wavlist $3=trials
  local D=$OUT/$1
  [ -f "$D/result.metrics" ] && { echo "[skip] $1"; return; }
  $PY ../../../speakerlab/bin/infer_sv_new.py --model_id "$RD_ID" \
      --wavs "$2" --local_model_dir $D --feat_out_format ark >> $OUT/infer.log 2>&1
  mkdir -p $D
  $PY ../../../speakerlab/bin/compute_score_metrics.py \
      --enrol_data $RD_ENROL --test_data $D/$RD_SUB --scores_dir $D --trials "$3" >> $OUT/score.log 2>&1
  echo "[$1] $(grep EER $D/result.metrics 2>/dev/null | head -1)"
}

# ---- ① 3s 轴三后端（复用官方链 wav 清单）----
for COND in noise_0db noise_m5db; do
  WL=$OFF/wav_fe_${COND}.txt
  eval_std v8d_ecapa_${COND}_3s $MID_ECAPA $WL $EC_ENROL_3 $TRIALS_3
  eval_std v8d_eres_${COND}_3s  $MID_ERES  $WL $SV/exp/clean_bench/eres2net/embeddings_3s $TRIALS_3
  eval_b6   v8d_b6_${COND}_3s  $WL $TRIALS_3
done
echo "[① 3s 三后端完] $(date)"

# ---- ② #26 换域 v8d（六条件 eres）----
echo "[② 换域增强 start] $(date)"
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY $AI/8.27/enhance_tse_v2.py --config train_tse.toml \
    --checkpoint $V8D/checkpoint_epoch1.tar --gpu $GPU --batch-size 8 --dur 3 \
    --test-root $ND \
    --conds music_0db,music_m5db,speech_0db,speech_m5db,soundbible_0db,soundbible_m5db \
    --out-root $V8D/enhanced_noise_domains >> $OUT/enhance_v8d_nd.log 2>&1 || { echo "[FATAL] enhance nd"; exit 1; }
for COND in music_0db music_m5db speech_0db speech_m5db soundbible_0db soundbible_m5db; do
  find $V8D/enhanced_noise_domains/$COND -name '*.wav' | sort > $OUT/wav_v8d_nd_${COND}.txt
  eval_std nd_v8d_eres_${COND} $MID_ERES $OUT/wav_v8d_nd_${COND}.txt \
      $SV/exp/clean_bench/eres2net/embeddings_3s $TRIALS_3
done

echo ""
echo "===== v8d 3s 轴 + 换域汇总 ====="
for D in $OUT/v8d_*_3s $OUT/nd_v8d_*; do
  M=$D/result.metrics
  [ -f "$M" ] && echo "$(basename $D): $(grep EER $M | head -1)"
done
echo "[v8d-3s+nd done] $(date)"
