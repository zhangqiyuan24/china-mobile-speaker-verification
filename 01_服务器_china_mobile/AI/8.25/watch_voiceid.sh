#!/bin/bash
# watch_voiceid.sh — 等 VoiceID 前端训练完成后自动跑 FE 评测链
# 流程: 1) 等 voiceid_train.log 出现 [done]
#       2) 用 epoch3(缺失则回退 epoch2/1) ckpt 降噪两条件(GPU2, 8.17 工具)
#       3) evaluate_combo 官方全表(GPU2, soup3 ckpt + FE 模式)
# 约束: GPU0 始终留空; 只写 AI/8.25
set -u
HERE=/data/home/ligangrui/china_mobile/AI/8.25
cd "$HERE"
PY=/home/ligangrui/china_mobile/zhangqiyuan/bin/python
SV=/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
FSN=/data/home/ligangrui/china_mobile/AI/8.17/code

echo "[watch] $(date '+%F %T') 等 voiceid 完成..."
while ! grep -q '\[done\]' "$HERE/voiceid_train.log"; do sleep 300; done
sleep 60

CKPT="$HERE/exp/spk_voiceid/checkpoint_epoch3.tar"
[ -f "$CKPT" ] || CKPT="$HERE/exp/spk_voiceid/checkpoint_epoch2.tar"
[ -f "$CKPT" ] || CKPT="$HERE/exp/spk_voiceid/checkpoint_epoch1.tar"
echo "[watch] $(date '+%F %T') 用 $CKPT 开始降噪"

OUT="$HERE/exp/spk_voiceid/eval_auto"
mkdir -p "$OUT"
cd "$FSN"
export PYTHONPATH="$FSN:$FSN/speech_enhance:${PYTHONPATH:-}"
for COND in noise_0db noise_m5db; do
    CUDA_VISIBLE_DEVICES=2 $PY -m speech_enhance.tools.inference \
        -C config/inference.toml -M "$CKPT" \
        -I "$SV/data/data_for_test/$COND/3s" -O "$OUT/denoised/$COND" \
        >> "$HERE/voiceid_eval.log" 2>&1
    echo "[watch] $(date '+%F %T') $COND 降噪退出码 $?"
done

E0=$(ls -d "$OUT"/denoised/noise_0db/enhanced_* 2>/dev/null | head -1)
E5=$(ls -d "$OUT"/denoised/noise_m5db/enhanced_* 2>/dev/null | head -1)
echo "[watch] 增强目录: $E0 / $E5"

cd /data/home/ligangrui/china_mobile/AI/8.24
"$PY" ../8.21/evaluate_combo.py --config config_soup_ac.toml \
    --checkpoint exp/soup3_abc/checkpoints/checkpoint_soup3.tar --gpu 2 \
    --enh-0db "$E0" --enh-m5db "$E5" \
    > "$HERE/voiceid_combo.log" 2>&1
echo "[watch] $(date '+%F %T') 评测退出码 $? — 结果: voiceid_combo.log + combo_summary.json"
