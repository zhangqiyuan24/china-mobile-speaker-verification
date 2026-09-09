#!/bin/bash
# launch_duraug_cont.sh — dur-aug 收敛续训守望器（8.28，用户授权"降就续，收益小自动停"）
# 预声明规则：
#   度量 = proxy hard val avg（官方数字永不回流训练决策）
#   每轮 3 epoch（~7.5h，GPU3，配方不变：--lambda-ns 0.5 --dur-aug --dur-seg 200）
#   轮末决策：本轮最优 avg 相对历史最优的增益 GAIN
#     GAIN >= 0.08 → 续下一轮（base = 本轮 checkpoint_best.tar）
#     GAIN <  0.08 → 判收敛，停；终评 ckpt = 历史最优轮的 checkpoint_best.tar
#   安全上限：续 4 轮（+12 ep）；崩溃/无 val → 停，留给巡检收尸
AI=/data/home/ligangrui/china_mobile/AI
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
GAIN_MIN=0.08
BEST=${BEST:-3.869}            # 历史最优（ext3 base；可由环境变量恢复现场）
BEST_TAG=${BEST_TAG:-ns05ext3} # 该最优所在轮
CUR_TAG=${CUR_TAG:-ns05duraug} # 当前在跑的轮次
RSTART=${RSTART:-2}            # 下一轮编号（现场恢复用）
STATE=$AI/8.27/duraug_cont_state.txt
echo "[$(date '+%m-%d %H:%M')] 收敛守望器启动 BEST=$BEST($BEST_TAG) CUR=$CUR_TAG RSTART=$RSTART" >> $STATE

for ROUND in $(seq $RSTART 5); do
  while pgrep -u ligangrui -f "train_tse_v4.py.*${CUR_TAG}" > /dev/null; do sleep 120; done
  sleep 60
  LOG=$AI/8.27/tse_v3_${CUR_TAG}.log
  NEW=$(grep -o 'avg=[0-9.]\+' $LOG 2>/dev/null | cut -d= -f2 | sort -n | head -1)
  if [ -z "$NEW" ]; then
    echo "[$(date '+%m-%d %H:%M')] $CUR_TAG 无 val，停（可能崩溃，留巡检收尸）" >> $STATE; break
  fi
  GAIN=$(awk "BEGIN{print $BEST-$NEW}")
  echo "[$(date '+%m-%d %H:%M')] $CUR_TAG 轮最优=$NEW 历史最优=$BEST($BEST_TAG) 增益=$GAIN" >> $STATE
  GO=$(awk "BEGIN{print ($GAIN>=$GAIN_MIN)?1:0}")
  if [ "$GO" != "1" ]; then
    echo "[$(date '+%m-%d %H:%M')] 判收敛停（增益 $GAIN < $GAIN_MIN）。终评 ckpt = exp/tse_v3_${BEST_TAG}/checkpoint_best.tar" >> $STATE; break
  fi
  NEW_TAG=ns05duraug_r${ROUND}
  cd $AI/8.27
  nohup $PY train_tse_v4.py --lambda-ns 0.5 \
    --base-ckpt $AI/8.27/exp/tse_v3_${CUR_TAG}/checkpoint_best.tar \
    --exp-tag $NEW_TAG --epochs 3 --gpu 3 --dur-aug --dur-seg 200 \
    > $AI/8.27/tse_v3_${NEW_TAG}.log 2>&1 &
  echo "[$(date '+%m-%d %H:%M')] 续训 $NEW_TAG（base=$CUR_TAG best）pid=$!" >> $STATE
  BEST=$NEW; BEST_TAG=$CUR_TAG; CUR_TAG=$NEW_TAG   # 8.28 修复：BEST 属于产出它的 CUR 轮
done
echo "[$(date '+%m-%d %H:%M')] 收敛守望器退出" >> $STATE
