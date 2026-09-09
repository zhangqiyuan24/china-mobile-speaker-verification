#!/bin/bash
# 一键把 sv-ecapa/data 链到 SpeakerProject，并设置相对路径索引。
# 用法: bash link_data.sh
# 前提: SpeakerProject 下已运行过 prepare_index.sh

set -e
cd "$(dirname "$0")"

SPEAKER_PROJ="../../../../../../SpeakerProject"

echo "========================================"
echo "  设置数据软链接"
echo "  目标: $SPEAKER_PROJ"
echo "========================================"

# 验证 SpeakerProject 存在
if [ ! -d "$SPEAKER_PROJ" ]; then
  echo "[ERROR] SpeakerProject 未找到: $SPEAKER_PROJ"
  echo "请确认 code/ 和 SpeakerProject/ 在同一级目录下"
  exit 1
fi
SPEAKER_PROJ=$(realpath "$SPEAKER_PROJ" 2>/dev/null || echo "$SPEAKER_PROJ")
echo "SpeakerProject 真实路径: $SPEAKER_PROJ"

# 如果 SpeakerProject 还没生成索引，先跑 prepare_index.sh
if [ ! -f "$SPEAKER_PROJ/data/train/wav.scp" ]; then
  echo ""
  echo "SpeakerProject 索引文件尚未生成，正在运行 prepare_index.sh..."
  bash "$SPEAKER_PROJ/prepare_index.sh"
fi

# 删除旧链接（如果有），建立新的
rm -f data
ln -sf "$SPEAKER_PROJ" data

echo ""
echo "软链接已建立: data → $SPEAKER_PROJ"

# 验证
echo ""
echo "验证链接:"
echo "  data/train/wav.scp    → $(head -1 data/data/train/wav.scp 2>/dev/null)"
echo "  data/test/wav.scp     → $(head -1 data/data/test/wav.scp 2>/dev/null)"
echo "  data/trials/               → $(ls data/data/trials/ 2>/dev/null | tr '\n' ' ')"

echo ""
echo "========================================"
echo "  设置完成"
echo "========================================"
echo ""
echo "下一步："
echo "  # 生成训练 CSV"
echo "  python local/prepare_data_csv.py --data_dir data/data/train"
echo ""
echo "  # 跑基线评测（以 sv-ecapa 为工作目录）"
echo "  python ../../../speakerlab/bin/infer_sv_batch.py \\"
echo "      --model_id iic/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k \\"
echo "      --wavs data/data/test/wav.scp \\"
echo "      --feat_out_dir exp/ecapa/embeddings \\"
echo "      --feat_out_format ark"
