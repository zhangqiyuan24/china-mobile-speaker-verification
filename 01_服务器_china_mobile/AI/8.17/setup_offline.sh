#!/bin/bash
# ============================================================
# 服务器（无网络）运行 — 离线部署 FullSubNet+ 降噪模型
#
# 前提: 已把 AI/8.17/ 整个目录传到服务器
#
# 作用:
#   1. 离线安装 pip 依赖 (librosa / torch_complex / toml)
#   2. 验证 import 和 checkpoint
#
# 用法: bash setup_offline.sh
# ============================================================
set -e

cd "$(dirname "$0")"

echo "============================================"
echo "  FullSubNet+ 降噪模型离线部署"
echo "============================================"

# ============================================================
# 1/3: 离线安装 pip 依赖
# ============================================================
echo ""
echo "==== 1/3: 离线安装 pip 依赖 ===="
if [ -d wheels ] && [ "$(ls wheels/*.whl 2>/dev/null | wc -l)" -gt 0 ]; then
    pip install --no-index --find-links wheels librosa torch_complex toml 2>&1 | tail -20
    echo "  依赖安装完成"
else
    echo "  [WARN] wheels/ 为空，请先本地运行 download_local.sh"
fi

# ============================================================
# 2/3: 验证 import
# ============================================================
echo ""
echo "==== 2/3: 验证 import ===="
python -c "
import librosa
import torch_complex
import toml
import torch
print('  [OK] librosa / torch_complex / toml / torch 全部可用')
print(f'  librosa={librosa.__version__}')
" 2>&1 | tail -5

# ============================================================
# 3/3: 检查 checkpoint + 代码
# ============================================================
echo ""
echo "==== 3/3: 检查 checkpoint + 代码 ===="
ok=1
if [ -s models/best_model.tar ]; then
    echo "  [OK] models/best_model.tar ($(du -h models/best_model.tar | cut -f1))"
else
    echo "  [ERROR] models/best_model.tar 缺失"
    ok=0
fi
if [ -d code/speech_enhance ]; then
    echo "  [OK] code/speech_enhance"
else
    echo "  [ERROR] code/speech_enhance 缺失"
    ok=0
fi

if [ $ok -eq 0 ]; then
    echo ""
    echo "  [ERROR] 有缺失，请先本地运行 download_local.sh 再传"
    exit 1
fi

echo ""
echo "============================================"
echo "  部署完成！"
echo "============================================"
echo ""
echo "  单条降噪测试 (对一个 noisy wav 目录做降噪):"
echo "    cd code"
echo "    python -m speech_enhance.tools.inference \\"
echo "        -C config/inference.toml \\"
echo "        -M ../models/best_model.tar \\"
echo "        -I <noisy_wav目录> \\"
echo "        -O <输出目录>"
echo ""
echo "  注意: 推理前需在 config/inference.toml 里把 dataset_dir_list 改成实际输入目录"
