#!/bin/bash
# ============================================================
# 批量上传文件到 GPU 服务器（一次输密码）
# 用法：bash upload_to_server.sh
# ============================================================
set -e

PORT=30188
HOST="root@sc01-ssh.gpuhome.cc"
REMOTE_DIR="/root/rivermind-data/"

# ── 本地文件路径（按实际情况修改）──
CODE_TAR="/mnt/e/china_mobile/code.tar.gz"
ZHANG_TAR="/mnt/e/china_mobile/zhangqiyuan.tar.gz"
ORI_TAR="/mnt/e/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa/data/data_for_test/ori.tar"
EVAL_SH="/mnt/e/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa/evaluate_clean.sh"

echo "============================================"
echo "  上传到 ${HOST}:${REMOTE_DIR}"
echo "============================================"
echo ""
echo "  1. code.tar.gz"
echo "  2. zhangqiyuan.tar.gz"
echo "  3. ori.tar"
echo "  4. evaluate_clean.sh"
echo ""

# 四个文件一条命令，只输一次密码
scp -P ${PORT} \
    "${CODE_TAR}" \
    "${ZHANG_TAR}" \
    "${ORI_TAR}" \
    "${EVAL_SH}" \
    ${HOST}:${REMOTE_DIR}

echo ""
echo "============================================"
echo "  全部上传完成"
echo "============================================"
