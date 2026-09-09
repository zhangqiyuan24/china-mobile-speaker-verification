#!/bin/bash
# ============================================================
# 本地（有网络）运行 — 下载 3 个声纹模型的 checkpoint
#
#   1. ECAPA-TDNN     (ModelScope)
#   2. ERes2Net-large (ModelScope)
#   3. ReDimNet2-B6   (GitHub Release)
#
# 用法 (本地 Windows/WSL, Git Bash):
#   bash download_models.sh
#
# 完成后, 把 AI/8.15/ 传到服务器, 运行 install.sh
# ============================================================
set -e

cd "$(dirname "$0")"
mkdir -p models

# 模型下载地址
declare -A URLS=(
  ["ecapa-tdnn.ckpt"]="https://www.modelscope.cn/models/iic/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k/resolve/v1.0.0/ecapa-tdnn.ckpt"
  ["eres2net_large_model.ckpt"]="https://www.modelscope.cn/models/iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k/resolve/v1.0.0/eres2net_large_model.ckpt"
  ["b6-vb2+vox2+cnc2_v0-lm.pt"]="https://github.com/PalabraAI/redimnet2/releases/download/v1.0.0/b6-vb2+vox2+cnc2_v0-lm.pt"
)

echo "============================================"
echo "  下载 3 个声纹模型 checkpoint"
echo "============================================"

for fname in "${!URLS[@]}"; do
  url="${URLS[$fname]}"
  dest="models/$fname"
  if [ -s "$dest" ]; then
    echo "  [跳过] $fname 已存在 ($(du -h "$dest" | cut -f1))"
    continue
  fi
  echo "  下载 $fname ..."
  echo "    $url"
  python -c "
import urllib.request
url = '$url'
dest = '$dest'
print(f'下载中...')
urllib.request.urlretrieve(url, dest)
print(f'完成 → {dest}')
"
done

echo ""
echo "============================================"
echo "  下载完成！"
echo "============================================"
ls -la models/
echo ""
echo "下一步: 把 AI/8.15/ 传到服务器, 运行 bash install.sh"
