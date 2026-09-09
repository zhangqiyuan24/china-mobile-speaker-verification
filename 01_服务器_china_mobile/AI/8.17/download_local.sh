#!/bin/bash
# ============================================================
# 本地（有网络）运行 — 下载 FullSubNet+ 降噪模型 + 依赖
#
# 下载内容:
#   1. FullSubNet+ 预训练 checkpoint (best_model.tar, Google Drive)
#   2. FullSubNet+ 代码 (从本地 code/FullSubNet-plus-master 拷贝)
#   3. pip 依赖 (librosa / torch_complex / toml, Linux wheel)
#
# 用法 (本地 Windows/WSL, Git Bash):
#   bash download_local.sh
#
# 完成后, 把 AI/8.17/ 传到服务器, 运行 bash setup_offline.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 服务器 Python 版本 (3.8 → 38)
PY_VER="${PY_VER:-38}"

# FullSubNet+ checkpoint 的 Google Drive file id
CHECKPOINT_ID="1UJSt1G0P_aXry-u79LLU_l9tCnNa2u7C"
CHECKPOINT_URL="https://drive.google.com/uc?id=${CHECKPOINT_ID}"

mkdir -p models wheels

# ============================================================
# 1/3: 下载 checkpoint
# ============================================================
echo "==== 1/3: 下载 FullSubNet+ checkpoint ===="
if [ -s "models/best_model.tar" ]; then
    echo "  已存在，跳过 ($(du -h models/best_model.tar | cut -f1))"
else
    echo "  用 gdown 下载 (Google Drive) ..."
    echo "  注意: Google Drive 国内访问不稳定, 若卡住请见 README 备选方案"
    pip install -q gdown 2>&1 | tail -2 || true
    # 用完整 URL + --fuzzy, 失败重试 3 次
    for i in 1 2 3; do
        echo "  第 $i 次尝试 ..."
        if gdown --fuzzy "$CHECKPOINT_URL" -O models/best_model.tar; then
            break
        fi
        echo "  下载失败, 重试 ..."
        sleep 3
    done
    if [ -s "models/best_model.tar" ]; then
        echo "  完成: $(du -h models/best_model.tar | cut -f1)"
    else
        echo "  [ERROR] 下载失败, 请参考 README 的备选方案"
    fi
fi

# ============================================================
# 2/3: 拷贝 FullSubNet+ 代码
# ============================================================
echo ""
echo "==== 2/3: 拷贝 FullSubNet+ 代码 ===="
if [ -d "code/speech_enhance" ]; then
    echo "  已存在，跳过"
else
    SRC="../../code/FullSubNet-plus-master/FullSubNet-plus-master"
    if [ -d "$SRC" ]; then
        mkdir -p code
        cp -r "$SRC"/* code/
        echo "  代码 → code/"
    else
        echo "  [WARN] 找不到本地代码: $SRC"
        echo "  请确认 code/FullSubNet-plus-master/FullSubNet-plus-master 存在"
    fi
fi

# ============================================================
# 3/3: 下载 pip 依赖 (Linux wheel)
# ============================================================
echo ""
echo "==== 3/3: 下载 pip 依赖 (Linux manylinux wheel) ===="
pip download -d wheels \
    --platform manylinux2014_x86_64 \
    --python-version "${PY_VER}" \
    --implementation cp \
    --abi "cp${PY_VER}" \
    --only-binary=:all: \
    librosa torch_complex toml importlib-metadata 2>&1 | tail -20
echo "  wheel 已下载到 wheels/"

echo ""
echo "============================================"
echo "  本地下载完成！"
echo "============================================"
ls -la models/ wheels/
echo ""
echo "下一步: 把整个 AI/8.17/ 传到服务器, 运行 bash setup_offline.sh"
