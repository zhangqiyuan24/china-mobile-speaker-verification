#!/bin/bash
# launch_final_v8d_gpu0.sh — v8d 终评大表·1s/5s 轴（GPU0，9.2 晚）
# 交付换 v8d 决策的必跑件：ckpt=官方链同源 ep1（oracle 4.7275/5.3855 纪录）。
# 结构照抄 launch_final_r5_gpu0.sh ②段：增强 1s/5s → eres/ecapa/B6 三后端 12 格。
# raw 参照行复用 final_eval_r5 已有 raw_*（不重跑）。
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
SV=/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
GPU=0
V8D=$AI/8.27/exp/tse_v8_v8d_sil
OUT=$AI/8.28/final_eval_v8d
MID_ERES=iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k
MID_ECAPA=iic/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k
RD_ID='redimnet2/b6/vb2+vox2+cnc2_v0/lm'
RD_SUB=redimnet2_b6_vb2+vox2+cnc2_v0_lm/embeddings
TRIALS_1=$SV/data/data_for_test/clean/1s/trials_cross_device
TRIALS_5=$SV/data/data_for_test/clean/5s/trials_cross_device
mkdir -p $OUT

echo "[v8d-1s5s start] $(date)"
# ---- 增强 v8d 1s/5s（enhance 在 export 前=全卡可见，--gpu 物理号）----
for DUR in 1 5; do
  echo "[enhance v8d ${DUR}s] $(date)"
  $PY $AI/8.27/enhance_tse_v2.py --config train_tse.toml --checkpoint $V8D/checkpoint_epoch1.tar \
      --gpu $GPU --batch-size 8 --dur $DUR \
      --out-root $V8D/enhanced_official_1s5s/${DUR}s >> $OUT/enhance_v8d_${DUR}s.log 2>&1 || { echo "[FATAL] enhance ${DUR}s"; exit 1; }
done

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
eval_b6 () {  # $1=name $2=wavlist $3=trials $4=enrol
  local D=$OUT/$1
  [ -f "$D/result.metrics" ] && { echo "[skip] $1"; return; }
  $PY ../../../speakerlab/bin/infer_sv_new.py --model_id "$RD_ID" \
      --wavs "$2" --local_model_dir $D --feat_out_format ark >> $OUT/infer.log 2>&1
  mkdir -p $D
  $PY ../../../speakerlab/bin/compute_score_metrics.py \
      --enrol_data "$4" --test_data $D/$RD_SUB --scores_dir $D --trials "$3" >> $OUT/score.log 2>&1
  echo "[$1] $(grep EER $D/result.metrics 2>/dev/null | head -1)"
}

# ---- 注册嵌入兜底：ecapa / redimnet2 1s/5s（r5 大表建过，命中即跳）----
for DUR in 1 5; do
  E=$SV/exp/clean_bench/ecapa/embeddings_${DUR}s
  if [ ! -d "$E" ] || [ -z "$(find $E -name '*.ark' 2>/dev/null | head -1)" ]; then
    echo "[build] ecapa clean enrol ${DUR}s"
    $PY ../../../speakerlab/bin/infer_sv_batch.py --model_id $MID_ECAPA \
        --wavs $SV/data/data_for_test/clean/${DUR}s/wav_list.txt \
        --feat_out_dir $E --feat_out_format ark >> $OUT/infer.log 2>&1
  fi
  E=$SV/exp/clean_bench/redimnet2_${DUR}s/$RD_SUB
  if [ ! -d "$E" ] || [ -z "$(find $E -name '*.ark' 2>/dev/null | head -1)" ]; then
    echo "[build] b6 clean enrol ${DUR}s"
    $PY ../../../speakerlab/bin/infer_sv_new.py --model_id "$RD_ID" \
        --wavs $SV/data/data_for_test/clean/${DUR}s/wav_list.txt \
        --local_model_dir $SV/exp/clean_bench/redimnet2_${DUR}s --feat_out_format ark >> $OUT/infer.log 2>&1
  fi
done

for COND in noise_0db noise_m5db; do
  for DUR in 1 5; do
    find $V8D/enhanced_official_1s5s/${DUR}s/$COND -name '*.wav' | sort > $OUT/wav_v8d_${COND}_${DUR}s.txt
    WL=$OUT/wav_v8d_${COND}_${DUR}s.txt
    if [ $DUR = 5 ]; then T=$TRIALS_5; else T=$TRIALS_1; fi
    eval_std v8d_eres_${COND}_${DUR}s  $MID_ERES  $WL $SV/exp/clean_bench/eres2net/embeddings_${DUR}s $T
    eval_std v8d_ecapa_${COND}_${DUR}s $MID_ECAPA $WL $SV/exp/clean_bench/ecapa/embeddings_${DUR}s $T
    eval_b6   v8d_b6_${COND}_${DUR}s  $WL $T $SV/exp/clean_bench/redimnet2_${DUR}s/$RD_SUB
  done
done

echo ""
echo "===== v8d 1s/5s 轴汇总 ====="
for D in $OUT/v8d_*_1s $OUT/v8d_*_5s; do
  M=$D/result.metrics
  [ -f "$M" ] && echo "$(basename $D): $(grep EER $M | head -1)"
done
echo "[v8d-1s5s done] $(date)"
