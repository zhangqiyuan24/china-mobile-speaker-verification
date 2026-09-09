#!/bin/bash
# launch_final_v8d_fix.sh — 9.3 接力补链：GPU0 m5db 5s 三后端 + GPU2 换域剩三条件
# 背景：昨夜主控脚本 00:12 中断（eres_m5db_5s 嵌入 2120/16401 断、ecapa/b6 未跑、
#       soundbible 两条件清单未建）。本脚本幂等（result.metrics 存在即 skip）。
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
SV=/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
V8D=$AI/8.27/exp/tse_v8_v8d_sil
ND=$V8D/enhanced_noise_domains
FIN=$AI/8.28/final_eval_v8d
MID_ERES=iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k
MID_ECAPA=iic/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k
RD_ID=redimnet2/b6/vb2+vox2+cnc2_v0/lm
RD_SUB=redimnet2_b6_vb2+vox2+cnc2_v0_lm/embeddings
TRIALS_3=$SV/data/data_for_test/clean/3s/trials_cross_device
TRIALS_5=$SV/data/data_for_test/clean/5s/trials_cross_device

score () {  # $1=enrol $2=test $3=scores_dir $4=trials
  $PY ../../../speakerlab/bin/compute_score_metrics.py --enrol_data "$1" --test_data "$2" \
      --scores_dir "$3" --trials "$4" >> $FIN/score_fix.log 2>&1
}

echo "[fix start] $(date)"
cd $SV
source path.sh

# ---- leg1 GPU0：m5db 5s 三后端（wav 清单 16401 已齐）----
export CUDA_VISIBLE_DEVICES=0
WL=$FIN/wav_v8d_noise_m5db_5s.txt

D=$FIN/v8d_eres_noise_m5db_5s
if [ ! -f "$D/result.metrics" ]; then
  rm -f $D/embedding_0.ark $D/embedding_0.scp   # 清 00:12 断链残缺嵌入
  $PY ../../../speakerlab/bin/infer_sv_batch.py --model_id "$MID_ERES" --wavs "$WL" \
      --feat_out_dir $D --feat_out_format ark >> $FIN/infer_fix.log 2>&1
  score $SV/exp/clean_bench/eres2net/embeddings_5s $D $D "$TRIALS_5"
fi
echo "[eres_m5db_5s] $(grep EER $D/result.metrics 2>/dev/null | head -1)"

D=$FIN/v8d_ecapa_noise_m5db_5s
if [ ! -f "$D/result.metrics" ]; then
  $PY ../../../speakerlab/bin/infer_sv_batch.py --model_id "$MID_ECAPA" --wavs "$WL" \
      --feat_out_dir $D --feat_out_format ark >> $FIN/infer_fix.log 2>&1
  score $SV/exp/clean_bench/ecapa/embeddings_5s $D $D "$TRIALS_5"
fi
echo "[ecapa_m5db_5s] $(grep EER $D/result.metrics 2>/dev/null | head -1)"

D=$FIN/v8d_b6_noise_m5db_5s
if [ ! -f "$D/result.metrics" ]; then
  $PY ../../../speakerlab/bin/infer_sv_new.py --model_id "$RD_ID" --wavs "$WL" \
      --local_model_dir $D --feat_out_format ark >> $FIN/infer_fix.log 2>&1
  score $SV/exp/clean_bench/redimnet2_5s/$RD_SUB $D/$RD_SUB $D "$TRIALS_5"
fi
echo "[b6_m5db_5s] $(grep EER $D/result.metrics 2>/dev/null | head -1)"
echo "[leg1 done] $(date)"

# ---- leg2 GPU2：换域剩三条件（eres 3s 口径；增强产物六条件已全）----
export CUDA_VISIBLE_DEVICES=2
for COND in speech_m5db soundbible_0db soundbible_m5db; do
  WL=$FIN/wav_v8d_nd_${COND}.txt
  [ -f "$WL" ] || find $ND/${COND} -name "*.wav" | sort > "$WL"
  D=$FIN/nd_v8d_eres_${COND}
  [ -f "$D/result.metrics" ] && { echo "[skip] ${COND}"; continue; }
  $PY ../../../speakerlab/bin/infer_sv_batch.py --model_id "$MID_ERES" --wavs "$WL" \
      --feat_out_dir $D --feat_out_format ark >> $FIN/infer_fix.log 2>&1
  score $SV/exp/clean_bench/eres2net/embeddings_3s $D $D "$TRIALS_3"
  echo "[nd_${COND}] $(grep EER $D/result.metrics 2>/dev/null | head -1)"
done
echo "[leg2 done] $(date)"

echo "===== fix summary ====="
for D in $FIN/v8d_eres_noise_m5db_5s $FIN/v8d_ecapa_noise_m5db_5s $FIN/v8d_b6_noise_m5db_5s \
         $FIN/nd_v8d_eres_speech_m5db $FIN/nd_v8d_eres_soundbible_0db $FIN/nd_v8d_eres_soundbible_m5db; do
  M=$D/result.metrics
  [ -f "$M" ] && echo "$(basename $D): $(grep EER $M | head -1)"
done
echo "[fix all done] $(date)"
