#!/bin/bash
# fix_raw_b6.sh — 修终评 raw_b6 1s/5s 四格（8.29）
# 起因：launch_final_r5_gpu3.sh 的 eval_b6 把 B6 注册嵌入写死为 3s（redimnet2_3s），
#       1s/5s trials 的注册键在 3s ark 中不存在 → score 异常、result.metrics 空。
# 策略：等 gpu3 主脚本（pid 1121682）退出避免同目录写冲突；
#       测试侧 ark 已算好（infer 与 enrol 无关），仅用正确时长注册嵌入重跑 score（纯 CPU）。
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
SV=/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
FIN=/data/home/ligangrui/china_mobile/AI/8.28/final_eval_r5
RD_SUB=redimnet2_b6_vb2+vox2+cnc2_v0_lm/embeddings
TRIALS_1=$SV/data/data_for_test/clean/1s/trials_cross_device
TRIALS_5=$SV/data/data_for_test/clean/5s/trials_cross_device

echo "[fix-raw-b6 start] $(date)（等待 gpu3 主脚本退出）"
while kill -0 1121682 2>/dev/null; do sleep 120; done
sleep 60
cd $SV
source path.sh

fix_cell () {  # $1=name $2=dur
  local D=$FIN/$1
  if [ -s "$D/result.metrics" ] && grep -q EER "$D/result.metrics"; then echo "[skip] $1"; return; fi
  if [ ! -d "$D/$RD_SUB" ] || [ -z "$(find $D/$RD_SUB -name '*.ark' 2>/dev/null | head -1)" ]; then
    echo "[miss-ark] $1 无测试嵌入，跳过（需重跑 infer）"; return
  fi
  local ENROL=$SV/exp/clean_bench/redimnet2_${2}s/$RD_SUB
  local T; if [ "$2" = 5 ]; then T=$TRIALS_5; else T=$TRIALS_1; fi
  rm -f "$D/result.metrics"
  $PY ../../../speakerlab/bin/compute_score_metrics.py \
      --enrol_data $ENROL --test_data $D/$RD_SUB --scores_dir $D --trials $T >> $FIN/score_fix_raw_b6.log 2>&1
  echo "[$1] $(grep EER $D/result.metrics 2>/dev/null | head -1)"
}

fix_cell raw_b6_noise_0db_1s  1
fix_cell raw_b6_noise_0db_5s  5
fix_cell raw_b6_noise_m5db_1s 1
fix_cell raw_b6_noise_m5db_5s 5
echo "[fix-raw-b6 done] $(date)"
