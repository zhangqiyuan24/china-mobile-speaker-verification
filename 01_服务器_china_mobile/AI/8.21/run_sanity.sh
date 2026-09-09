#!/bin/bash
# ============================================================
# E2-proto sanity：prototype 提取 + 3 epoch 训练（与 8.20 sanity2 同参对照）
#
# 用法：
#   bash run_sanity.sh                 # 默认 GPU 3（共享卡）
#   GPU=1 bash run_sanity.sh
#   ABLATION=1 bash run_sanity.sh      # 追加 --no-proto 随机初始化对照组
#                                     # （输出到 exp/proto_robust_random，不覆盖）
#
# 判读重点（见 README「epoch1 对照」）：
#   1) epoch1 train_acc 明显高于 random 的 0.017
#   2) epoch1 clean 不再从 1.25 跳到 4%+
#   3) 0dB/-5dB 初始退化明显减轻（random ep1 = 11.00 / 16.50）
# ============================================================
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU="${GPU:-3}"
CFG="${CFG:-config_proto.toml}"
NPZ="exp/prototypes/prototypes.npz"
MANIFEST_820="../8.20/exp/robust/data/manifest.json"

if [ ! -f "$MANIFEST_820" ]; then
    echo "[ERROR] 缺 $MANIFEST_820 —— 8.21 复用 8.20 的数据，请先在 AI/8.20 跑："
    echo "        python prepare_multicond_data.py --config config_robust.toml"
    exit 1
fi

if [ ! -f "$NPZ" ]; then
    echo "==== Step 1: 生成 speaker prototypes（pretrained 模型，一次性，缓存） ===="
    python prepare_prototypes.py --config "$CFG" --gpu "$GPU"
else
    echo "==== Step 1: 跳过（$NPZ 已存在；重提请删除或 --force） ===="
fi

echo "==== Step 2: sanity 训练（3 epoch，prototype init；与 8.20 sanity2 同参） ===="
# 与 8.20 sanity2 完全一致：batch 12 × accum 8（有效 96）、AMP、共享 GPU 3
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python train_eres2net_robust_proto.py --config "$CFG" --sanity --gpu "$GPU" \
    --amp --batch-size 12 --accum 8 --num-workers 4 --val-batch-size 8

if [ "${ABLATION:-0}" = "1" ]; then
    echo ""
    echo "==== Step 3（可选 ablation）: 随机初始化对照组（同一代码路径） ===="
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    python train_eres2net_robust_proto.py --config "$CFG" --sanity --gpu "$GPU" \
        --amp --batch-size 12 --accum 8 --num-workers 4 --val-batch-size 8 \
        --no-proto --exp-name proto_robust_random
    echo "对照：exp/proto_robust/val_history.json  vs  exp/proto_robust_random/val_history.json"
    echo "     （另见 8.20 随机初始化历史：../8.20/exp/robust/val_history.json）"
fi

echo ""
echo "==== sanity 完成；检查清单 ===="
echo "  1) [proto-init] 诊断：shape 对齐、行范数 ≈1、逐位回读校验通过"
echo "  2) [prototype init] clean top1 明显高于随机（≈0.01%），aam loss 明显低于 15.6"
echo "  3) epoch1 对照表：clean/0dB/-5dB/loss/acc 全面优于 random（4.00/11.00/16.50/15.05/0.017）"
echo "  4) [guard] 冻结参数未变 / 可训练参数已更新"
echo "  5) exp/proto_robust/checkpoints/ 有 epoch*.tar 与 best.tar，val_history.json 完整"
echo "全部正常后：bash run_train.sh（8 epoch 正式训练）"
