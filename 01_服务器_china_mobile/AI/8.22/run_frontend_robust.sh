#!/bin/bash
# ============================================================
# 8.22 第一步：补齐消融矩阵的最后一格
#   微调前端(8.18 spk-preserve) × robust 后端(8.21)
#
# 一条命令做完五件事：
#   ① 自动确认 7.7902/9.9186 出自哪个 eval 目录（eval_e10 / eval_results）
#      —— 也就确定了该用哪个前端 ckpt（epoch10 / epoch3_fixed）
#   ② 该目录 denoised/ 下的增强 wav 若还在（每档 16401 条）→ 直接复用，
#      与 7.7902 那次是逐位相同的前端输出，省 1~2h 重降噪
#      若被清了 → Plan B：用确认的 ckpt 重降噪（调 8.21/enhance_testset.sh，
#      输出 8.21/exp/enhanced_spk，不覆盖原版 exp/enhanced）
#   ③ 把上一轮"原版前端(8.17)"的 frontend 缓存改名保底（*_vanilla*），不销毁记录
#   ④ 跑 8.21 的 SKIP_TRAIN=1 run_train.sh（raw 侧全缓存，只跑 frontend 侧 ≈40min）
#   ⑤ 结束打印 2×3 完整矩阵 + 判读标准
#
# 注意：这不是 8.22 的正式训练（混合域 matched 训练），只是它的前置测量——
#       训练方案（用哪个前端生成语料 / 要不要换 teacher）由本格结果决定。
#
# 用法：
#   cd ~/china_mobile/AI/8.22
#   bash run_frontend_robust.sh
#   GPU=1 bash run_frontend_robust.sh   # 仅 Plan B 重降噪时用到（默认 3）
# ============================================================
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V818="$HERE/../8.18"
V821="$HERE/../8.21"
SPK="$V818/exp/spk_preserve"
COMBO="$V821/exp/proto_robust/eval/combo"
SV_ECAPA="$(cd "$HERE/../../code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa" && pwd)"

# ---------- ① 确认 7.7902 的出处 → 定前端 ckpt ----------
echo "==== ① 确认 7.7902/9.9186 出处（前端 ckpt 跟着定） ===="
EVAL_DIR=""
CKPT_NAME=""
for pair in "eval_e10 checkpoint_epoch10.tar" "eval_results checkpoint_epoch3_fixed.tar"; do
    set -- $pair
    d="$SPK/$1"; name="$2"
    m="$d/scores/denoised_noise_0db/eres2net/3s/result.metrics"
    if [ -f "$m" ] && grep -q "7.7902" "$m"; then
        EVAL_DIR="$d"; CKPT_NAME="$name"; break
    fi
done
if [ -z "$EVAL_DIR" ]; then
    echo "[ERROR] eval_e10 / eval_results 的 0dB result.metrics 都不含 7.7902，不敢猜前端。现有全部结果："
    grep -RH EER --include=result.metrics "$SPK" || true
    exit 1
fi
echo "  出处     : $EVAL_DIR"
echo "  前端 ckpt: $SPK/$CKPT_NAME"
[ -f "$SPK/$CKPT_NAME" ] || { echo "[ERROR] 前端 ckpt 不存在: $SPK/$CKPT_NAME"; exit 1; }

# ---------- ② 增强wav：能复用就复用 ----------
count_wavs() { find "$1" -name "*.wav" 2>/dev/null | wc -l; }

echo ""
echo "==== ② 检查现成增强 wav ===="
N0=$(count_wavs "$EVAL_DIR/denoised/noise_0db")
N5=$(count_wavs "$EVAL_DIR/denoised/noise_m5db")
echo "  noise_0db: $N0 / 16401    noise_m5db: $N5 / 16401"
if [ "$N0" -eq 16401 ] && [ "$N5" -eq 16401 ]; then
    ENH0="$EVAL_DIR/denoised/noise_0db"
    ENH5="$EVAL_DIR/denoised/noise_m5db"
    echo "  → 复用现成 wav（与 7.7902 那次前端输出逐位一致）"
else
    echo "  → wav 不全，Plan B：用 $CKPT_NAME 重降噪（约 1~2h）"
    ( cd "$V821" && FULLSUBNET_CKPT="$SPK/$CKPT_NAME" \
        OUT_ROOT=exp/enhanced_spk GPU="${GPU:-3}" bash enhance_testset.sh )
    ENH0="$V821/exp/enhanced_spk/noise_0db"
    ENH5="$V821/exp/enhanced_spk/noise_m5db"
    for d in "$ENH0" "$ENH5"; do
        n=$(count_wavs "$d")
        [ "$n" -eq 16401 ] || { echo "[ERROR] 重降噪不完整: $d 只有 $n 条"; exit 1; }
    done
fi

# ---------- ③ 原版前端旧缓存改名保底 ----------
echo ""
echo "==== ③ 原版前端(8.17) 旧缓存改名 ===="
if [ -d "$COMBO/frontend" ]; then
    if [ -d "$COMBO/frontend_vanilla817" ]; then rm -rf "$COMBO/frontend"; else mv "$COMBO/frontend" "$COMBO/frontend_vanilla817"; fi
    echo "  frontend -> frontend_vanilla817"
fi
for c in noise_0db noise_m5db; do
    s="$COMBO/scores/frontend_$c"
    if [ -d "$s" ]; then
        t="$COMBO/scores/frontend_vanilla_$c"
        if [ -d "$t" ]; then rm -rf "$s"; else mv "$s" "$t"; fi
        echo "  scores/frontend_$c -> frontend_vanilla_$c"
    fi
done

# ---------- ④ 跑组合评测 ----------
echo ""
echo "==== ④ 组合评测（raw 侧全缓存，只跑 frontend 侧 ≈40min） ===="
cd "$V821"
ENH_0DB="$ENH0" ENH_M5DB="$ENH5" SKIP_TRAIN=1 bash run_train.sh

# ---------- ⑤ 打印完整矩阵 ----------
eer() { grep -oE "EER = [0-9.]+" "$1" 2>/dev/null | head -1 | grep -oE "[0-9.]+" || true; }
FRESH0=$(eer "$COMBO/scores/frontend_noise_0db/result.metrics")
FRESH5=$(eer "$COMBO/scores/frontend_noise_m5db/result.metrics")
SPK0=$(eer "$EVAL_DIR/scores/denoised_noise_0db/eres2net/3s/result.metrics")
SPK5=$(eer "$EVAL_DIR/scores/denoised_noise_m5db/eres2net/3s/result.metrics")
VANP0=$(eer "$SV_ECAPA/data/data_for_test/clean/result/denoised_noise_0db/eres2net/3s/result.metrics")
VANP5=$(eer "$SV_ECAPA/data/data_for_test/clean/result/denoised_noise_m5db/eres2net/3s/result.metrics")
VANR0=$(eer "$COMBO/scores/frontend_vanilla_noise_0db/result.metrics")
VANR5=$(eer "$COMBO/scores/frontend_vanilla_noise_m5db/result.metrics")
RAWR0=$(eer "$COMBO/scores/raw_noise_0db/result.metrics")
RAWR5=$(eer "$COMBO/scores/raw_noise_m5db/result.metrics")

fmt() { if [ -n "$1" ]; then printf "%8s" "$1"; else printf "%8s" "-"; fi; }
echo ""
echo "==== ⑤ 完整矩阵（0dB / -5dB，EER%） ===="
echo "测试侧 \\ 后端      pretrained           robust(8.21)"
echo "原始带噪           $(fmt 8.3911)/$(fmt 11.0557)      $(fmt "$RAWR0")/$(fmt "$RAWR5")"
echo "原版前端(8.17)     $(fmt "$VANP0")/$(fmt "$VANP5")      $(fmt "$VANR0")/$(fmt "$VANR5")"
echo "微调前端($CKPT_NAME) $(fmt "$SPK0")/$(fmt "$SPK5")      $(fmt "$FRESH0")/$(fmt "$FRESH5")"
echo ""
echo "判读（看右下角新格）："
echo "  ≈7.0~7.6      级联成立，8.22 混合域 matched 训练直接叠加"
echo "  ≈7.8~8.2      两路增益互斥，matched 训练是唯一路线"
echo "  明显差于 7.79  前端收益绑死 pretrained 后端，8.22 需用 robust 当 teacher 重调前端"
echo ""
echo "记住前端 ckpt：$SPK/$CKPT_NAME —— 8.22 生成 enh 训练语料必须用同一个。"
