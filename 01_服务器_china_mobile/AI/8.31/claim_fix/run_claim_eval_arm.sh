#!/bin/bash
# run_claim_eval_arm.sh — #38 单臂官方 claim 评测全链（claim 优先口径，用户 8.31 批准）
# 用法: run_claim_eval_arm.sh TAG GPU
#   TAG 如 asym30 → EXP=8.27/exp/tse_v5_asym30，训练日志 $FIX/asym30.log
# 链：① 选 ckpt（proxy claim eer_avg 最低 epoch，平局取晚；缺则 checkpoint_best）
#     ② 全官方测试集 oracle 条件增强（3s 两档，断点续跑）
#     ③ 冒认抽样集声称者条件增强（对集与 #37 逐位一致）
#     ④ 官方 ERes2Net 提嵌入 ×2（幂等）
#     ⑤ analyze_claim_v2 三口径（raw / oracleFE / claimFE）+ bootstrap CI
set -e
TAG=$1
GPU=$2
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
SV=/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
SL=$SV/../../../speakerlab/bin
FIX=$AI/8.31/claim_fix
CR=$AI/8.31/claim_retest
EXP=$AI/8.27/exp/${EXP_PREFIX:-tse_v5}_$TAG  # v6 臂: EXP_PREFIX=tse_v6 bash 本脚本
MID_ERES=iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k
OUT=$FIX/official_$TAG
mkdir -p $OUT
LOG=$OUT/chain.log
cd $SV && source path.sh
export CUDA_VISIBLE_DEVICES=$GPU

echo "[① 选ckpt] $(date)" | tee -a $LOG
BEST_EP=$(grep -a 'val-hard' $FIX/$TAG.log 2>/dev/null \
  | sed -n 's/.*ep\([0-9]\+\) .*eer_avg=\([0-9.]\+\).*/\1 \2/p' \
  | sort -k2,2n -k1,1nr | head -1 | awk '{print $1}')
if [ -n "$BEST_EP" ] && [ -f "$EXP/checkpoint_epoch${BEST_EP}.tar" ]; then
  CKPT=$EXP/checkpoint_epoch${BEST_EP}.tar
else
  CKPT=$EXP/checkpoint_best.tar
fi
echo "  ckpt=$CKPT (proxy-claim 最优 ep=${BEST_EP:-best})" | tee -a $LOG

echo "[② 全集oracle增强] $(date)" | tee -a $LOG
$PY $AI/8.27/enhance_tse_v2.py --checkpoint $CKPT --gpu $GPU --batch-size 8 \
    --dur 3 --out-root $OUT/enhanced_fe >> $LOG 2>&1

echo "[③ 冒认claim增强] $(date)" | tee -a $LOG
$PY $CR/enhance_claim.py --checkpoint $CKPT --gpu $GPU --batch-size 8 \
    --out-root $OUT/enhanced_claim >> $OUT/enhance_claim.log 2>&1

echo "[④ 提嵌入] $(date)" | tee -a $LOG
for COND in noise_0db noise_m5db; do
  find $OUT/enhanced_fe/$COND -name '*.wav' | sort > $OUT/wav_fe_${COND}.txt
  if [ ! -f $OUT/fe_emb/$COND/embedding_0.scp ]; then
    $PY $SL/infer_sv_batch.py --model_id $MID_ERES --wavs $OUT/wav_fe_${COND}.txt \
        --feat_out_dir $OUT/fe_emb/$COND --feat_out_format ark >> $OUT/infer.log 2>&1
  else
    echo "  [skip] fe_emb/$COND 已有"
  fi
  find $OUT/enhanced_claim/flat_$COND -name '*.wav' | sort > $OUT/wav_claim_${COND}.txt
  if [ ! -f $OUT/claim_emb/$COND/embedding_0.scp ]; then
    $PY $SL/infer_sv_batch.py --model_id $MID_ERES --wavs $OUT/wav_claim_${COND}.txt \
        --feat_out_dir $OUT/claim_emb/$COND --feat_out_format ark >> $OUT/infer.log 2>&1
  else
    echo "  [skip] claim_emb/$COND 已有"
  fi
done

echo "[⑤ 分析] $(date)" | tee -a $LOG
$PY $FIX/analyze_claim_v2.py --fe-root $OUT/fe_emb --claim-root $OUT/claim_emb \
    --out $OUT/summary.json 2>&1 | tee $OUT/summary.txt
echo "[done $TAG] $(date)" | tee -a $LOG
