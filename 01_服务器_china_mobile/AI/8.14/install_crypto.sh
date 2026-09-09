#!/bin/bash
# ============================================================
# 服务器（无网络）运行 — 离线降级 cryptography
#
# 前提: 已把 AI/8.14/crypto_wheels/ 传到服务器
#
# 用法: bash install_crypto.sh
# ============================================================
set -e

cd "$(dirname "$0")"

echo "============================================"
echo "  离线降级 cryptography (47.0.0 → 40.0.2)"
echo "============================================"

if [ ! -d crypto_wheels ] || [ "$(ls crypto_wheels/*.whl 2>/dev/null | wc -l)" -eq 0 ]; then
    echo "[ERROR] crypto_wheels/ 为空或不存在"
    echo "  请先在本地运行 download_crypto.sh，再传 crypto_wheels/ 到服务器"
    exit 1
fi

echo "  当前版本: $(pip show cryptography 2>/dev/null | grep Version || echo '未安装')"

echo ""
echo "  离线安装 cryptography==40.0.2 ..."
pip install --no-index --find-links crypto_wheels "cryptography==40.0.2" 2>&1 | tail -15

echo ""
echo "  安装后版本: $(pip show cryptography | grep Version)"

echo ""
echo "  验证 import ..."
python -c "from cryptography.hazmat.primitives import hashes; print('  [OK] cryptography import 成功')" 2>&1 | tail -3

echo ""
echo "============================================"
echo "  完成! 现在可以重新运行评测:"
echo "    bash evaluate_clean.sh"
echo "============================================"
