#!/bin/bash
# ============================================================
# 本地（有网络）运行 — 下载兼容的 cryptography wheel
#
# 问题: 服务器 cryptography 47.0.0 需要 GLIBC 2.33, 但服务器 GLIBC 2.31
# 解决: 下载 cryptography 40.0.2 (manylinux_2_28 = GLIBC 2.28, 兼容 2.31)
#
# 用法 (本地 Windows, Git Bash):
#   bash download_crypto.sh
#
# 完成后, 把 AI/8.14/crypto_wheels/ 传到服务器, 运行 install_crypto.sh
# ============================================================
set -e

cd "$(dirname "$0")"

# 服务器 Python 版本 (3.8 → 38, 3.9 → 39)
PY_VER="${PY_VER:-38}"

mkdir -p crypto_wheels

echo "============================================"
echo "  下载兼容的 cryptography (40.0.2)"
echo "  GLIBC 2.28 兼容, 服务器 GLIBC 2.31 可用"
echo "============================================"

# cryptography 40.0.2 是 abi3 wheel (cp36-abi3-manylinux_2_28_x86_64)
# --no-deps 只下载 cryptography 本身 (cffi 服务器已随 47.0.0 装好, 不用动)
pip download "cryptography==40.0.2" \
    -d crypto_wheels \
    --platform manylinux_2_28_x86_64 \
    --implementation cp \
    --python-version 36 \
    --abi abi3 \
    --only-binary=:all: \
    --no-deps \
    2>&1 | tail -15

echo ""
echo "============================================"
echo "  下载完成!"
echo "============================================"
echo "  crypto_wheels/ 内容:"
ls -la crypto_wheels/
echo ""
echo "下一步: 把 AI/8.14/crypto_wheels/ 传到服务器, 运行 bash install_crypto.sh"
