#!/bin/bash
# ============================================================
# 本地（有网络）运行 — 下载所有离线依赖到 AI/8.13/
#
# 下载内容:
#   1. WeSep pretrained 模型 (bsrnn_ecapa_vox1, 249MB)
#   2. wespeaker 代码 (WeSep 的 ECAPA speaker model 依赖)
#   3. pip 依赖的 Linux wheel (供服务器离线安装)
#   4. wesep 代码 (从本地 wesep-master 拷贝)
#
# 用法 (本地 Windows, Git Bash):
#   bash download_local.sh
#
# 完成后, 把整个 AI/8.13/ 目录传到服务器, 在服务器运行 setup_offline.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 服务器 Python 版本 (3.8 → 38, 3.9 → 39), 按你服务器实际改
PY_VER="${PY_VER:-38}"

# WeSep 模型下载地址
WESEP_MODEL_URL="https://www.modelscope.cn/datasets/wenet/wesep_pretrained_models/resolve/master/bsrnn_ecapa_vox1.tar.gz"
# wespeaker 仓库
WESPEAKER_REPO="https://github.com/wenet-e2e/wespeaker.git"

mkdir -p third_party/models wheels

# ============================================================
# 1/4: 下载 WeSep 模型
# ============================================================
echo "==== 1/4: 下载 WeSep 模型 (249MB) ===="
if [ -f "third_party/models/bsrnn_ecapa_vox1.tar.gz" ]; then
    echo "  已存在，跳过"
else
    python -c "
import urllib.request, sys
url = '${WESEP_MODEL_URL}'
dest = 'third_party/models/bsrnn_ecapa_vox1.tar.gz'
print(f'下载 {url}')
urllib.request.urlretrieve(url, dest)
print(f'完成 → {dest}')
"
fi

# ============================================================
# 2/4: 下载 wespeaker 代码
# ============================================================
echo "==== 2/4: 下载 wespeaker 代码 ===="
if [ -d "third_party/wespeaker" ]; then
    echo "  已存在，跳过"
else
    git clone --depth 1 "$WESPEAKER_REPO" third_party/_wespeaker_tmp
    mv third_party/_wespeaker_tmp/wespeaker third_party/wespeaker
    rm -rf third_party/_wespeaker_tmp
    echo "  wespeaker → third_party/wespeaker"
fi

# ============================================================
# 3/4: 下载 pip 依赖 (Linux wheel)
# ============================================================
echo "==== 3/4: 下载 pip 依赖 ===="

# 3a. 纯 Python 包 (py3-none-any, 跨平台, 无需 --platform)
#     --no-deps 关键: 否则 pip 会解析 silero-vad 的 torch 依赖, 卡住
#     silero-vad 固定 5.1.2 (WeSep 官方版本, 兼容 torch 1.10)
echo "  下载纯 Python 包: tqdm / packaging / silero-vad==5.1.2 ..."
pip download --no-deps -d wheels tqdm packaging "silero-vad==5.1.2" 2>&1 | tail -8

# 3b. C 扩展包 (需要 Linux manylinux wheel, 指定 manylinux2014 = GLIBC 2.17 最保守)
echo "  下载 C 扩展包: PyYAML / soundfile (Linux manylinux wheel) ..."
pip download --no-deps -d wheels \
    --platform manylinux2014_x86_64 \
    --python-version "${PY_VER}" \
    --implementation cp \
    --abi "cp${PY_VER}" \
    --only-binary=:all: \
    PyYAML soundfile 2>&1 | tail -8

echo "  wheel 已下载到 wheels/"

# ============================================================
# 4/4: 拷贝 wesep 代码 (从本地 wesep-master)
# ============================================================
echo "==== 4/4: 拷贝 wesep 代码 ===="
WESEP_SRC="../../code/wesep-master/wesep-master/wesep"
if [ -d "third_party/wesep" ]; then
    echo "  third_party/wesep 已存在，跳过"
elif [ -d "$WESEP_SRC" ]; then
    cp -r "$WESEP_SRC" third_party/wesep
    echo "  wesep → third_party/wesep"
else
    echo "  [WARN] 找不到本地 wesep 代码: $WESEP_SRC"
    echo "  请确认 code/wesep-master/wesep-master/wesep 存在，"
    echo "  或手动把 wesep 包拷到 third_party/wesep"
fi

echo ""
echo "============================================"
echo "  本地下载完成！"
echo "============================================"
echo "  third_party/wesep/      ← wesep 代码"
echo "  third_party/wespeaker/  ← wespeaker 代码"
echo "  third_party/models/     ← WeSep 模型 (tar.gz)"
echo "  wheels/                 ← pip 离线 wheel"
echo ""
echo "下一步: 把整个 AI/8.13/ 目录传到服务器，运行 bash setup_offline.sh"
