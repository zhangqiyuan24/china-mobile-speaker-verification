#!/bin/bash
# ============================================================
# 数据与预训练模型一键获取脚本(子课题一:多条件中文声纹识别)
#
# 用法:  bash scripts/download_data.sh <项目根目录>
#   <项目根目录> = 本仓库克隆路径,例如:
#     bash scripts/download_data.sh /path/to/china-mobile-speaker-verification
#
# 下载目标目录(与服务器实验时布局一致):
#   data/musan/                  MUSAN 噪声库(自动下载,OpenSLR 17)
#   data/rirs_noises/            RIR 混响库(自动下载,OpenSLR 28)
#   data/3dspeaker/              3D-Speaker 原始数据集(需账号,见下方引导)
#   pretrained/                  三个后端预训练模型(ModelScope,自动)
#
# 说明:MUSAN 与 RIR 有直链自动下载;3D-Speaker 数据集需在 OpenDataLab
#   注册后手动下载(约 190GB,含训练集/测试集),脚本会给出引导与目录放置位置。
# ============================================================
set -e

ROOT="${1:-.}"
mkdir -p "$ROOT/data" "$ROOT/pretrained"
cd "$ROOT/data"

DL() {  # DL <url> <输出文件>
  if command -v wget >/dev/null 2>&1; then wget -c "$1" -O "$2"
  else curl -L -C - "$1" -o "$2"; fi
}

# ---------- 1. MUSAN 噪声库(约 11 GB) ----------
if [ -d musan/librispeech ] || [ -f musan.tar.gz.done ]; then
  echo "[1/4] MUSAN 已存在,跳过"
else
  echo "[1/4] 下载 MUSAN(OpenSLR 17,约 11 GB)..."
  DL https://www.openslr.org/resources/17/musan.tar.gz musan.tar.gz
  tar xzf musan.tar.gz && touch musan.tar.gz.done
fi

# ---------- 2. RIRS_NOISES 混响库(约 1.3 GB) ----------
if [ -d rirs_noises/RIRS ] || [ -f rirs_noises.zip.done ]; then
  echo "[2/4] RIRS 已存在,跳过"
else
  echo "[2/4] 下载 RIRS_NOISES(OpenSLR 28,约 1.3 GB)..."
  DL https://www.openslr.org/resources/28/rirs_noises.zip rirs_noises.zip
  unzip -q rirs_noises.zip && touch rirs_noises.zip.done
fi

# ---------- 3. 3D-Speaker 原始数据集(约 190 GB,需手动) ----------
if [ -d 3dspeaker/test ]; then
  echo "[3/4] 3D-Speaker 已存在,跳过"
else
  cat <<'EOF'
[3/4] 3D-Speaker 数据集需要手动获取(约 190 GB,需注册免费账号):
  1. 访问 https://opendatalab.com/3DSpeaker (或 3D-Speaker GitHub README 的数据集链接)
  2. 注册/登录后下载:
       - 训练集(train,约 190GB、579,003 条、9,900 说话人)
       - 官方测试集(test,240 人、18,782 条)
  3. 解压放置到 data/3dspeaker/ 下,使目录形如:
       data/3dspeaker/train/...        (训练 wav 与说话人标注)
       data/3dspeaker/test/...         (测试 wav,含 Device/Distance/Dialect 元信息)
  4. 重新运行本脚本继续。
EOF
  echo "已给出 3D-Speaker 引导;放置完成后可重跑本脚本,或先用 pip 方式继续第 4 步。"
fi

# ---------- 4. 三个后端预训练模型(ModelScope,自动) ----------
cd "$ROOT/pretrained"
python3 - <<'EOF'
import os, sys
try:
    from modelscope.hub.snapshot_download import snapshot_download
except ImportError:
    print("未安装 modelscope,请先: pip install modelscope")
    sys.exit(0)

# 三个基线后端(3D-Speaker 官方中文预训练模型)
models = {
    "ERes2Net-large(主力后端)": "damo/speech_eres2net_large_sv_zh-cn_3dspeaker_16k",
    "ECAPA-TDNN(经典后端)":    "damo/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k",
    "ERes2Net-base(对照)":     "damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k",
}
for name, mid in models.items():
    print(f"下载 {name}: {mid}")
    try:
        snapshot_download(mid)
    except Exception as e:
        print(f"  跳过({e});可在 https://modelscope.cn 搜索该模型手动下载")

# 可选:WavLM-base-sv(第 8 章大模型对比用;不需复现主线可不装)
print("可选:WavLM-base-sv 需要 huggingface_hub —— pip install huggingface_hub 后执行")
print("  huggingface-cli download microsoft/wavlm-base-sv")
EOF

echo ""
echo "========== 数据准备完成检查 =========="
for d in musan rirs_noises 3dspeaker; do
  [ -d "$ROOT/data/$d" ] && echo "  ✔ data/$d" || echo "  ✘ data/$d 缺失"
done
echo "下一步:进入 code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa/,运行 gen_all_test_data.sh 生成多条件测试集(详见仓库 README.md)"
