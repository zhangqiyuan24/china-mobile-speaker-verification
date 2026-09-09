#!/bin/bash
# 9.4 补全 clean 基准打分(纯 CPU;嵌入全为既有缓存,不提取/不训练/不改任何现有文件)
# 背景:sv-ecapa/evaluate_clean.sh 三后端x三时长此前仅 ecapa/1s 完成打分,其余 8 格补齐
set -u
SV=/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
OUT=/data/home/ligangrui/china_mobile/AI/9.4/clean_bench
mkdir -p "$OUT"
cd "$SV"
export PYTHONPATH=../../../:${PYTHONPATH:-}
export OMP_NUM_THREADS=4
run_one() {
  m=$1; d=$2
  EMB=exp/clean_bench/$m/embeddings_$d
  [ "$m" = redimnet2 ] && EMB=exp/clean_bench/redimnet2_${d}/redimnet2_b6_vb2+vox2+cnc2_v0_lm/embeddings
  SC=$OUT/$m/$d
  mkdir -p "$SC"
  echo "[$(date +%H:%M:%S)] start $m $d"
  "$PY" ../../../speakerlab/bin/compute_score_metrics.py     --enrol_data "$EMB" --test_data "$EMB" --scores_dir "$SC"     --trials "data/data_for_test/clean/$d/trials_all"              "data/data_for_test/clean/$d/trials_cross_device"              "data/data_for_test/clean/$d/trials_cross_distance"              "data/data_for_test/clean/$d/trials_cross_dialect"     > "$SC/run.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] done  $m $d rc=$rc"
}
run_one ecapa 3s & run_one ecapa 5s &
run_one eres2net 1s & run_one eres2net 3s & run_one eres2net 5s &
run_one redimnet2 1s & run_one redimnet2 3s & run_one redimnet2 5s &
wait
echo ALL_DONE
