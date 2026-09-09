#!/bin/bash
# launch_fe_clean.sh — 缺口2:clean 测试集直测格(9.3)
# r5 前端增强 clean 测试语音(oracle 条件,与交付口径一致) -> ERes2Net-large 嵌入 -> 官方协议打分
# 对照锚点: AI/8.28/clean_baseline_eres2net_3s (raw clean EER 5.29)
set -e
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
SV=/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
GPU=0
R5=$AI/8.27/exp/tse_v3_ns05duraug_r5
OUT=$AI/9.3/fe_clean
MID_ERES=iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k
TRIALS_3=$SV/data/data_for_test/clean/3s/trials_cross_device
mkdir -p $OUT

echo "[fe_clean start] $(date)"
cd $SV
source path.sh
export CUDA_VISIBLE_DEVICES=$GPU

# ① clean 测试集过 r5 前端
if [ -z "$(find $OUT/enhanced_clean -name "*.wav" 2>/dev/null | head -1)" ]; then
  $PY $AI/8.27/enhance_tse_v2.py --config train_tse.toml --checkpoint $R5/checkpoint_best.tar \
      --gpu $GPU --batch-size 8 --dur 3 --conds clean \
      --out-root $OUT/enhanced_clean > $OUT/enhance_clean.log 2>&1
fi
find $OUT/enhanced_clean -name "*.wav" | sort > $OUT/wav_fe_clean_3s.txt
echo "enhanced wavs: $(wc -l < $OUT/wav_fe_clean_3s.txt)"

# ② ERes2Net 嵌入 + 官方试验清单打分(注册嵌入与交付大表同一套 clean_bench embeddings_3s)
D=$OUT/fe_clean_eres_3s
mkdir -p $D
$PY ../../../speakerlab/bin/infer_sv_batch.py --model_id "$MID_ERES" --wavs $OUT/wav_fe_clean_3s.txt \
    --feat_out_dir $D --feat_out_format ark > $OUT/infer_fe_clean.log 2>&1
$PY ../../../speakerlab/bin/compute_score_metrics.py \
    --enrol_data $SV/exp/clean_bench/eres2net/embeddings_3s --test_data $D \
    --scores_dir $D --trials "$TRIALS_3" > $OUT/score_fe_clean.log 2>&1
echo "[fe_clean result] $(grep EER $D/result.metrics | head -1)"
echo "[raw anchor]    $(grep EER $AI/8.28/clean_baseline_eres2net_3s/result.metrics | head -1)"
echo "[fe_clean done] $(date)"
