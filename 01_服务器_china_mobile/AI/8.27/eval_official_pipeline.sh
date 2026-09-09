#!/bin/bash
# eval_official_pipeline.sh — #18 三后端官方管线口径闭合（8.28 凌晨布防，GPU0）
# 后端：ECAPA-TDNN / ERes2Net-large（infer_sv_batch.py）+ ReDimNet2-B6（infer_sv_new.py）
# 模式：raw 自检（复现官方基线）+ FE（ns05ext2 ep9 新版前端增强产物，−34.8/−41.6% 版）
# 协议：enrol=官方 clean 3s 嵌入（复用官方基线原产物），test=各条件嵌入，
#       trials=clean/3s/trials_cross_device 全量，scorer=官方 compute_score_metrics。
# 判定：raw 复现基线 → FE 数字即完美口径闭合（官方链、官方 ckpt、零我们链成分）。
set -e
SV=/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
FE_ROOT=/data/home/ligangrui/china_mobile/AI/8.27/exp/tse_v3_ns05ext2/enhanced_official
OUT=/data/home/ligangrui/china_mobile/AI/8.27/exp/official_pipeline_3backends
TRIALS=$SV/data/data_for_test/clean/3s/trials_cross_device

cd $SV
source path.sh
export CUDA_VISIBLE_DEVICES=0
mkdir -p $OUT

# ---- FE wav_list（增强产物，stem 与官方 trials key 一致）----
for cond in noise_0db noise_m5db; do
  find $FE_ROOT/$cond -name '*.wav' | sort > $OUT/fe_${cond}_wavlist.txt
done

infer_batch () {  # $1=model_id $2=wav_list $3=emb_dir
  $PY ../../../speakerlab/bin/infer_sv_batch.py \
      --model_id "$1" --wavs "$2" --feat_out_dir "$3" --feat_out_format ark
}
infer_redim () {  # $1=wav_list $2=emb_base
  $PY ../../../speakerlab/bin/infer_sv_new.py \
      --model_id redimnet2/b6/vb2+vox2+cnc2_v0/lm \
      --wavs "$1" --local_model_dir "$2" --feat_out_format ark
}
score () {  # $1=enrol $2=test $3=out_dir
  mkdir -p "$3"
  $PY ../../../speakerlab/bin/compute_score_metrics.py \
      --enrol_data "$1" --test_data "$2" --scores_dir "$3" --trials $TRIALS
}

# ============ 1/3 ECAPA-TDNN ============
B=ecapa; MID=iic/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k
CLEAN_EMB=$SV/exp/clean_bench/ecapa/embeddings_3s
for cond in noise_0db noise_m5db; do
  echo "--- $B raw $cond ---"
  infer_batch $MID $SV/data/data_for_test/$cond/3s/wav_list.txt $OUT/$B/raw_$cond
  score $CLEAN_EMB $OUT/$B/raw_$cond $OUT/$B/raw_$cond
  echo "--- $B fe $cond ---"
  infer_batch $MID $OUT/fe_${cond}_wavlist.txt $OUT/$B/fe_$cond
  score $CLEAN_EMB $OUT/$B/fe_$cond $OUT/$B/fe_$cond
done

# ============ 2/3 ERes2Net-large ============
B=eres2net; MID=iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k
CLEAN_EMB=$SV/exp/clean_bench/eres2net/embeddings_3s
for cond in noise_0db noise_m5db; do
  echo "--- $B raw $cond ---"
  infer_batch $MID $SV/data/data_for_test/$cond/3s/wav_list.txt $OUT/$B/raw_$cond
  score $CLEAN_EMB $OUT/$B/raw_$cond $OUT/$B/raw_$cond
  echo "--- $B fe $cond ---"
  infer_batch $MID $OUT/fe_${cond}_wavlist.txt $OUT/$B/fe_$cond
  score $CLEAN_EMB $OUT/$B/fe_$cond $OUT/$B/fe_$cond
done

# ============ 3/3 ReDimNet2-B6 ============
B=redimnet2
RD_SUB=redimnet2_b6_vb2+vox2+cnc2_v0_lm/embeddings
CLEAN_EMB=$SV/exp/clean_bench/redimnet2_3s/$RD_SUB
for cond in noise_0db noise_m5db; do
  echo "--- $B raw $cond ---"
  infer_redim $SV/data/data_for_test/$cond/3s/wav_list.txt $OUT/${B}_raw_${cond}
  score $CLEAN_EMB $OUT/${B}_raw_${cond}/$RD_SUB $OUT/$B/raw_$cond
  echo "--- $B fe $cond ---"
  infer_redim $OUT/fe_${cond}_wavlist.txt $OUT/${B}_fe_${cond}
  score $CLEAN_EMB $OUT/${B}_fe_${cond}/$RD_SUB $OUT/$B/fe_$cond
done

# ============ 汇总 ============
echo ""
echo "===== #18 三后端官方管线汇总（raw 自检 vs FE）====="
for B in ecapa eres2net redimnet2; do
  for cond in noise_0db noise_m5db; do
    for mode in raw fe; do
      M=$OUT/$B/${mode}_${cond}/result.metrics
      [ -f "$M" ] && echo "$B $mode $cond: $(grep 'EER' $M | head -1)"
    done
  done
done
echo "[done] 全部结果在 $OUT"
