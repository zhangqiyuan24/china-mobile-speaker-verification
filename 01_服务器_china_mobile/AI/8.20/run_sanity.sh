#!/bin/bash
# ============================================================
# E2 sanity：prepare（如未跑过）+ 3 epoch 训练 + 逐 epoch 验证
# 目的：验证数据/冻结/AAM/验证/保存整条管线，看趋势不看绝对值。
# 预期：0dB/-5dB 验证 EER 相对 epoch0 明显下降，clean 基本不掉。
# 用法：  bash run_sanity.sh            # 单卡，默认 GPU 1
#         GPU=3 bash run_sanity.sh
# ============================================================
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU="${GPU:-1}"
CFG="${CFG:-config_robust.toml}"
MANIFEST="exp/robust/data/manifest.json"

if [ ! -f "$MANIFEST" ]; then
    echo "==== Step 1: 数据准备（首次运行，含验证 bench 落盘） ===="
    python prepare_multicond_data.py --config "$CFG"
else
    echo "==== Step 1: 跳过（$MANIFEST 已存在；如需重建请删除 exp/robust/data） ===="
fi

echo "==== Step 2: sanity 训练（3 epoch） ===="
python train_eres2net_robust.py --config "$CFG" --sanity --gpu "$GPU"

echo ""
echo "==== sanity 完成 ===="
echo "检查点："
echo "  1) prepare 输出的隔离断言是否全部通过（说话人三方 / 噪声三方零交集）"
echo "  2) [freeze_report] 可训练参数量是否约 20M+（embedding 侧）"
echo "  3) [guard] 冻结参数未变 / 可训练参数已更新"
echo "  4) 每个 epoch 的验证 EER：0dB/-5dB 应逐 epoch 下降，clean 下降不超过门限"
echo "  5) exp/robust/checkpoints/ 下有 checkpoint_epoch*.tar 与 checkpoint_best.tar"
echo "  6) val_history.json 记录完整"
echo "全部正常后运行： bash run_train.sh"
