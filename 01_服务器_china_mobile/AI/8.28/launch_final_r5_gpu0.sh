#!/bin/bash
# launch_final_r5_gpu0.sh — GPU0：①B6 哨兵补测 ②r5 终评 1s/5s 轴 ③#26 换域重跑（8.29）
# B6 哨兵修复：infer_sv_new.py --local_model_dir（哨兵误用 infer_sv_batch 被拒）
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
SV=/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
GPU=0
R5=$AI/8.27/exp/tse_v3_ns05duraug_r5
FIN=$AI/8.28/final_eval_r5
PB=$AI/8.28/gen_probe            # ①哨兵 B6 结果回填到这里
ND=$AI/8.28/noise_domains        # ③#26 换域根
MID_ERES=iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k
MID_ECAPA=iic/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k
RD_ID='redimnet2/b6/vb2+vox2+cnc2_v0/lm'
RD_SUB=redimnet2_b6_vb2+vox2+cnc2_v0_lm/embeddings
TRIALS_1=$SV/data/data_for_test/clean/1s/trials_cross_device
TRIALS_3=$SV/data/data_for_test/clean/3s/trials_cross_device
TRIALS_5=$SV/data/data_for_test/clean/5s/trials_cross_device
mkdir -p $FIN

echo "[gpu0 start] $(date)"
cd $SV
source path.sh
export CUDA_VISIBLE_DEVICES=$GPU

eval_std () {  # $1=name $2=mid $3=wavlist $4=enrol $5=trials $6=outroot
  local O=${6:-$FIN}
  local D=$O/$1
  [ -f "$D/result.metrics" ] && { echo "[skip] $1"; return; }
  $PY ../../../speakerlab/bin/infer_sv_batch.py --model_id "$2" --wavs "$3" \
      --feat_out_dir $D --feat_out_format ark >> $O/infer.log 2>&1
  mkdir -p $D
  $PY ../../../speakerlab/bin/compute_score_metrics.py \
      --enrol_data "$4" --test_data $D --scores_dir $D --trials "$5" >> $O/score.log 2>&1
  echo "[$1] $(grep EER $D/result.metrics 2>/dev/null | head -1)"
}
eval_b6 () {  # $1=name $2=wavlist $3=trials $4=enrol $5=outroot
  local O=${5:-$FIN}
  local D=$O/$1
  [ -f "$D/result.metrics" ] && { echo "[skip] $1"; return; }
  $PY ../../../speakerlab/bin/infer_sv_new.py --model_id "$RD_ID" \
      --wavs "$2" --local_model_dir $D --feat_out_format ark >> $O/infer.log 2>&1
  mkdir -p $D
  $PY ../../../speakerlab/bin/compute_score_metrics.py \
      --enrol_data "$4" --test_data $D/$RD_SUB --scores_dir $D --trials "$3" >> $O/score.log 2>&1
  echo "[$1] $(grep EER $D/result.metrics 2>/dev/null | head -1)"
}

# ---- ① B6 哨兵补测（r2/ext3 3s，结果回填 gen_probe）----
RD_ENROL_3=$SV/exp/clean_bench/redimnet2_3s/$RD_SUB
for COND in noise_0db noise_m5db; do
  eval_b6 r2_b6_${COND}_3s   $PB/wav_r2_eres_${COND}_3s.txt   $TRIALS_3 $RD_ENROL_3 $PB
  eval_b6 ext3_b6_${COND}_3s $PB/wav_ext3_b6_${COND}_3s.txt   $TRIALS_3 $RD_ENROL_3 $PB
done
echo "[① B6 哨兵补测完] $(date)"

# ---- B6 注册嵌入兜底：redimnet2 1s/5s ----
for DUR in 1 5; do
  E=$SV/exp/clean_bench/redimnet2_${DUR}s/$RD_SUB
  if [ ! -d "$E" ] || [ -z "$(find $E -name '*.ark' 2>/dev/null | head -1)" ]; then
    echo "[build] b6 clean enrol ${DUR}s"
    $PY ../../../speakerlab/bin/infer_sv_new.py --model_id "$RD_ID" \
        --wavs $SV/data/data_for_test/clean/${DUR}s/wav_list.txt \
        --local_model_dir $SV/exp/clean_bench/redimnet2_${DUR}s --feat_out_format ark >> $FIN/infer.log 2>&1
  fi
done

# ---- ② r5 增强 1s/5s → 三后端评测 ----
for DUR in 1 5; do
  echo "[enhance r5 ${DUR}s] $(date)"
  $PY $AI/8.27/enhance_tse_v2.py --config train_tse.toml --checkpoint $R5/checkpoint_best.tar \
      --gpu $GPU --batch-size 8 --dur $DUR \
      --out-root $R5/enhanced_official_1s5s/${DUR}s >> $FIN/enhance_r5_${DUR}s.log 2>&1 || { echo "[FATAL] enhance ${DUR}s"; exit 1; }
done
for COND in noise_0db noise_m5db; do
  if [ "$COND" = noise_0db ]; then C=0db; else C=m5db; fi
  for DUR in 1 5; do
    find $R5/enhanced_official_1s5s/${DUR}s/$COND -name '*.wav' | sort > $FIN/wav_r5_${COND}_${DUR}s.txt
    WL=$FIN/wav_r5_${COND}_${DUR}s.txt
    if [ $DUR = 5 ]; then T=$TRIALS_5; else T=$TRIALS_1; fi
    eval_std r5_eres_${COND}_${DUR}s  $MID_ERES  $WL $SV/exp/clean_bench/eres2net/embeddings_${DUR}s $T
    eval_std r5_ecapa_${COND}_${DUR}s $MID_ECAPA $WL $SV/exp/clean_bench/ecapa/embeddings_${DUR}s $T
    eval_b6   r5_b6_${COND}_${DUR}s  $WL $T $SV/exp/clean_bench/redimnet2_${DUR}s/$RD_SUB
  done
done
echo "[② r5 1s/5s 轴完] $(date)"

# ---- ③ #26 换域重跑（r5 ckpt，六条件，eres 口径）----
echo "[③ #26 r5 增强] $(date)"
$PY $AI/8.27/enhance_tse_v2.py --config train_tse.toml --checkpoint $R5/checkpoint_best.tar \
    --gpu $GPU --batch-size 8 --dur 3 --test-root $ND \
    --conds music_0db,music_m5db,speech_0db,speech_m5db,soundbible_0db,soundbible_m5db \
    --out-root $R5/enhanced_noise_domains >> $FIN/enhance_r5_nd.log 2>&1 || { echo "[FATAL] enhance nd"; exit 1; }
for COND in music_0db music_m5db speech_0db speech_m5db soundbible_0db soundbible_m5db; do
  find $R5/enhanced_noise_domains/$COND -name '*.wav' | sort > $FIN/wav_r5_nd_${COND}.txt
  eval_std nd_r5_eres_${COND} $MID_ERES $FIN/wav_r5_nd_${COND}.txt \
      $SV/exp/clean_bench/eres2net/embeddings_3s $TRIALS_3
done

echo ""
echo "===== GPU0 汇总（B6哨兵 + r5 1s/5s + #26 r5）====="
for D in $PB/r2_b6* $PB/ext3_b6* $FIN/r5_* $FIN/nd_*; do
  M=$D/result.metrics
  [ -f "$M" ] && echo "$(basename $D): $(grep EER $M | head -1)"
done
echo "[gpu0 all done] $(date)"
