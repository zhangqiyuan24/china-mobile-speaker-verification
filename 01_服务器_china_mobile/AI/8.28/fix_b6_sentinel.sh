#!/bin/bash
# fix_b6_sentinel.sh — 补跑 #27 哨兵 4 个 B6 格（GPU1，8.29）
# 起因：probe 的 compute_score_metrics 崩溃留下 0 字节 result.metrics，
#       弱守卫 [ -f ] 误判"已完成"。本脚本守卫 = 非空且含 EER。
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
SV=/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
PB=/data/home/ligangrui/china_mobile/AI/8.28/gen_probe
GPU=1
RD_ID='redimnet2/b6/vb2+vox2+cnc2_v0/lm'
RD_SUB=redimnet2_b6_vb2+vox2+cnc2_v0_lm/embeddings
RD_ENROL_3=$SV/exp/clean_bench/redimnet2_3s/$RD_SUB
TRIALS_3=$SV/data/data_for_test/clean/3s/trials_cross_device

cd $SV
source path.sh
export CUDA_VISIBLE_DEVICES=$GPU

eval_b6 () {  # $1=name $2=wavlist
  local D=$PB/$1
  if [ -s "$D/result.metrics" ] && grep -q EER "$D/result.metrics"; then echo "[skip] $1"; return; fi
  rm -f "$D/result.metrics"
  $PY ../../../speakerlab/bin/infer_sv_new.py --model_id "$RD_ID" \
      --wavs "$2" --local_model_dir $D --feat_out_format ark >> $PB/infer_fix.log 2>&1
  mkdir -p $D
  $PY ../../../speakerlab/bin/compute_score_metrics.py \
      --enrol_data $RD_ENROL_3 --test_data $D/$RD_SUB --scores_dir $D --trials $TRIALS_3 >> $PB/score_fix.log 2>&1
  echo "[$1] $(grep EER $D/result.metrics 2>/dev/null | head -1)"
}

echo "[fix-b6 start] $(date)"
eval_b6 r2_b6_noise_0db_3s   $PB/wav_r2_eres_noise_0db_3s.txt
eval_b6 ext3_b6_noise_0db_3s $PB/wav_ext3_b6_noise_0db_3s.txt
eval_b6 r2_b6_noise_m5db_3s   $PB/wav_r2_eres_noise_m5db_3s.txt
eval_b6 ext3_b6_noise_m5db_3s $PB/wav_ext3_b6_noise_m5db_3s.txt
echo "[fix-b6 done] $(date)"
