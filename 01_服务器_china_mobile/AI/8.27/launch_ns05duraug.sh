#!/bin/bash
# launch_ns05duraug.sh — 守望器 8（#25 时长适配臂，8.28 晨布防）
# 等守望器 7（ext3 官方链）退出 → 读 ext3 FE 数字自动定 base → GPU3 起 dur-aug 续训。
# 预声明：ext3 官方降幅 ≥ ext2（−34.8%/−41.6%）→ base=ext3 best；否则退 ext2 ep9。
# 训练侧双门槛：①3s hard 不降 ②1s 官方降幅 ≥28%（训后官方重评定，此脚本只管起训）。
while pgrep -u ligangrui -f 'launch_ns05ext3_official.sh' > /dev/null; do sleep 120; done
sleep 300

AI=/data/home/ligangrui/china_mobile/AI
FE0=$(grep EER $AI/8.27/exp/official_ns05ext3/eval/combo/scores/frontend_noise_0db/result.metrics 2>/dev/null | awk '{print $3}')
FE5=$(grep EER $AI/8.27/exp/official_ns05ext3/eval/combo/scores/frontend_noise_m5db/result.metrics 2>/dev/null | awk '{print $3}')

BASE=""
if [ -n "$FE0" ] && [ -n "$FE5" ]; then
  # 绝对值直比（8.28 修正原降幅口径混用 bug；ext2 终值 5.4669/6.4615）
  USE3=$(awk "BEGIN{print ($FE0<5.4669 && $FE5<6.4615)?1:0}")
  echo "[base] ext3 官方 $FE0/$FE5 vs ext2 5.4669/6.4615, 用ext3=$USE3"
  if [ "$USE3" = "1" ]; then
    BASE=$AI/8.27/exp/tse_v3_ns05ext3/checkpoint_best.tar
  fi
fi
[ -z "$BASE" ] && BASE=$AI/8.27/exp/tse_v3_ns05ext2/checkpoint_best.tar
echo "[base] 采用 $BASE"

PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
cd $AI/8.27
nohup $PY train_tse_v4.py --lambda-ns 0.5 --base-ckpt $BASE \
  --exp-tag ns05duraug --epochs 3 --gpu 3 --dur-aug --dur-seg 200 \
  > $AI/8.27/tse_v3_ns05duraug.log 2>&1 &
echo "[launched] dur-aug 训练 pid $! 日志 tse_v3_ns05duraug.log"
