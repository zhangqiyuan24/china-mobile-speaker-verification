#!/bin/bash
# run_claim_retest.sh — #37 claim-conditioned 重测全链（8.31，GPU0）
# ① 抽样(幂等) ② claim 增强(断点续跑)+摊平 ③ 官方提嵌入 ④ 四口径分析
set -e
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
SV=/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
SL=$SV/../../../speakerlab/bin
W=$AI/8.31/claim_retest
R5=$AI/8.27/exp/tse_v3_ns05duraug_r5
MID_ERES=iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k
GPU=0

cd $SV && source path.sh
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p $W/emb

echo "[① 抽样] $(date)"
if [ ! -f $W/claim_groups.json ]; then
  $PY $W/sample_claim_trials.py | tee $W/sample_stats.txt
else
  echo "[skip] 已有 claim_groups.json"
fi

echo "[② claim 增强] $(date)"
$PY $W/enhance_claim.py --checkpoint $R5/checkpoint_best.tar \
    --gpu $GPU --batch-size 8 --out-root $W/enhanced \
    >> $W/enhance_claim.log 2>&1
echo "[增强完成] $(date)"

echo "[③ 官方提嵌入] $(date)"
for COND in noise_0db noise_m5db; do
  find $W/enhanced/flat_$COND -name '*.wav' | sort > $W/wav_claim_${COND}.txt
  echo "  $COND: $(wc -l < $W/wav_claim_${COND}.txt) 条"
  if [ ! -f $W/emb/$COND/embedding_0.scp ]; then
    $PY $SL/infer_sv_batch.py --model_id $MID_ERES \
        --wavs $W/wav_claim_${COND}.txt \
        --feat_out_dir $W/emb/$COND --feat_out_format ark >> $W/infer.log 2>&1
  else
    echo "  [skip] emb/$COND 已有"
  fi
done

echo "[④ 分析] $(date)"
$PY $W/analyze_claim.py 2>&1 | tee $W/claim_summary.txt
echo "[all done] $(date)"
