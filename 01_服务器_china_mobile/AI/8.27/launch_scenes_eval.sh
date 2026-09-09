#!/bin/bash
# launch_scenes_eval.sh — 守望器 9（#21 场景分项，8.28 布防）
# 等 #19 1s/5s 管线退出 -> GPU0 依次 5 场景 × (enhance + raw/fe 嵌入 + score)。
# 前端 = ext2 ep9（与主线同 ckpt 可比）；trials = clean/3s/trials_cross_device（stem 对齐）。
# 预期：overlap 场景是 TSE 强项（注册条件化定向提取）；reverb 是分布外（诚实报）。
while pgrep -u ligangrui -f 'eval_1s5s_pipeline' > /dev/null; do sleep 180; done
sleep 120

PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
AI=/data/home/ligangrui/china_mobile/AI
SV=/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
CKPT=$AI/8.27/exp/tse_v3_ns05ext2/checkpoint_best.tar
FE_ROOT=$AI/8.27/exp/tse_v3_ns05ext2/enhanced_official_scenes
OUT=$AI/8.27/exp/official_scenes_tse_v3_ns05ext2
MID=iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k
mkdir -p $OUT

cd $SV
source path.sh

for SCENE in reverb overlap2_cn overlap2_en overlap3_cn overlap3_en; do
  echo "===== scene $SCENE ====="
  # 1) FE 增强（断点续跑）
  $PY $AI/8.27/enhance_tse_v2.py --config train_tse.toml --checkpoint $CKPT \
      --gpu 0 --batch-size 8 --dur 3 --conds $SCENE \
      --out-root $FE_ROOT/$SCENE \
      >> $AI/8.27/enhance_scenes.log 2>&1
  # 2) raw + fe 官方管线评测
  CLEAN_EMB=$SV/exp/clean_bench/eres2net/embeddings_3s
  TRIALS=$SV/data/data_for_test/clean/3s/trials_cross_device
  for MODE in raw fe; do
    echo "--- eres2net $MODE $SCENE ---"
    if [ "$MODE" = "raw" ]; then
      WAVS=$SV/data/data_for_test/$SCENE/3s/wav_list.txt
    else
      find $FE_ROOT/$SCENE/$SCENE -name '*.wav' | sort > $OUT/fe_${SCENE}_wavlist.txt
      WAVS=$OUT/fe_${SCENE}_wavlist.txt
    fi
    $PY ../../../speakerlab/bin/infer_sv_batch.py --model_id $MID --wavs $WAVS \
        --feat_out_dir $OUT/eres2net/${MODE}_${SCENE}_3s --feat_out_format ark
    mkdir -p $OUT/eres2net/${MODE}_${SCENE}_3s
    $PY ../../../speakerlab/bin/compute_score_metrics.py \
        --enrol_data $CLEAN_EMB --test_data $OUT/eres2net/${MODE}_${SCENE}_3s \
        --scores_dir $OUT/eres2net/${MODE}_${SCENE}_3s --trials $TRIALS
  done
  echo "[$SCENE] raw: $(grep EER $OUT/eres2net/raw_${SCENE}_3s/result.metrics 2>/dev/null | head -1)"
  echo "[$SCENE] fe : $(grep EER $OUT/eres2net/fe_${SCENE}_3s/result.metrics 2>/dev/null | head -1)"
done
echo "[all done] $OUT"
