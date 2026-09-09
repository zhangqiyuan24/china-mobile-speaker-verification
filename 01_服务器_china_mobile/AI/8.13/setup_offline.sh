#!/bin/bash
# ============================================================
# 服务器（无网络）运行 — 离线准备评测环境
#
# 前提: 已把整个 AI/8.13/ 目录 (含 third_party/ + wheels/) 传到服务器
#
# 作用:
#   1. 解压 WeSep 模型到 third_party/models/
#   2. 离线安装 pip 依赖 (从 wheels/)
#   3. 检查 wesep / wespeaker 代码是否就位
#
# 用法: bash setup_offline.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  WeSep TSE + ERes2Net 离线准备"
echo "============================================"

# ============================================================
# 1/3: 解压 WeSep 模型
# ============================================================
echo ""
echo "==== 1/3: 解压 WeSep 模型 ===="
MODEL_DIR="third_party/models"
if [ -f "$MODEL_DIR/avg_model.pt" ] && [ -f "$MODEL_DIR/config.yaml" ]; then
    echo "  模型已解压，跳过"
else
    if [ -f "$MODEL_DIR/bsrnn_ecapa_vox1.tar.gz" ]; then
        tar -xzf "$MODEL_DIR/bsrnn_ecapa_vox1.tar.gz" -C "$MODEL_DIR" --strip-components=1
        rm -f "$MODEL_DIR/bsrnn_ecapa_vox1.tar.gz"
        echo "  解压完成: $(ls "$MODEL_DIR")"
    else
        echo "  [ERROR] 找不到 $MODEL_DIR/bsrnn_ecapa_vox1.tar.gz"
        echo "  请先在本地运行 download_local.sh，再传整个 AI/8.13 到服务器"
        exit 1
    fi
fi

# ============================================================
# 2/3: 离线安装 pip 依赖
# ============================================================
echo ""
echo "==== 2/3: 离线安装 pip 依赖 ===="
if [ -d "wheels" ] && [ "$(ls wheels/*.whl 2>/dev/null | wc -l)" -gt 0 ]; then
    # 普通包正常装
    pip install --no-index --find-links wheels \
        tqdm PyYAML soundfile packaging 2>&1 | tail -10

    # silero-vad 用 --no-deps 强制装: 绕过 torch>=1.12 版本检查
    # (服务器 torch 1.10.1, silero-vad 实际兼容, 只是 metadata 声明 >=1.12)
    pip install --no-index --find-links wheels --no-deps silero-vad 2>&1 | tail -10

    echo "  依赖安装完成"
else
    echo "  [WARN] wheels/ 目录为空或不存在"
    echo "  请先在本地运行 download_local.sh 下载 wheel"
fi

# ============================================================
# 3/3: 检查代码就位
# ============================================================
echo ""
echo "==== 3/3: 检查代码就位 ===="
ok=1
if [ ! -d "third_party/wesep" ]; then
    echo "  [ERROR] third_party/wesep 缺失 (wesep 代码)"
    ok=0
else
    echo "  [OK] third_party/wesep"
fi
if [ ! -d "third_party/wespeaker" ]; then
    echo "  [ERROR] third_party/wespeaker 缺失 (wespeaker 代码)"
    ok=0
else
    echo "  [OK] third_party/wespeaker"
fi
if [ ! -f "$MODEL_DIR/avg_model.pt" ]; then
    echo "  [ERROR] third_party/models/avg_model.pt 缺失 (WeSep 模型)"
    ok=0
else
    echo "  [OK] third_party/models/avg_model.pt"
fi

# 修复 wespeaker 的 s3prl 依赖 (我们只用 ECAPA speaker model, 不需要 frontend)
echo ""
echo "==== 2.5/3: 清理 wespeaker 的 s3prl 依赖 ===="
python << 'PYEOF'
# 1) __init__.py 去掉 cli.speaker import (触发 frontend → s3prl)
init_path = 'third_party/wespeaker/__init__.py'
content = open(init_path).read()
content = content.replace('from wespeaker.cli.speaker import load_model  # noqa', '')
content = content.replace('from wespeaker.cli.speaker import load_model_pt  # noqa', '')
open(init_path, 'w').write(content)
print('  [OK] 清理 wespeaker/__init__.py')

# 2) speaker_model.py 去掉 redimnet2 / w2vbert_adapter_mfa (它们 import frontend → s3prl)
sm_path = 'third_party/wespeaker/models/speaker_model.py'
content = open(sm_path).read()
content = content.replace('import wespeaker.models.redimnet2 as redimnet2\n', '')
content = content.replace('import wespeaker.models.w2vbert_adapter_mfa as w2vbert_adapter_mfa\n', '')
open(sm_path, 'w').write(content)
print('  [OK] 清理 speaker_model.py 的 s3prl 相关 import')

# 3) extractor.py 跳过 VAD 模型加载 (torch 1.10.1 加载 silero-vad JIT 会崩, 且 apply_vad=False 用不到)
ext_path = 'third_party/wesep/cli/extractor.py'
content = open(ext_path).read()
old_vad = '        self.vad = load_silero_vad()'
new_vad = '        self.vad = None  # skip VAD load: torch 1.10.1 incompatible with silero-vad JIT'
content = content.replace(old_vad, new_vad)
open(ext_path, 'w').write(content)
print('  [OK] 跳过 extractor.py 的 VAD 加载')
PYEOF

# 验证 import
echo ""
echo "  验证 import ..."
python -c "
import sys
sys.path.insert(0, 'third_party')
import wesep
import wespeaker
print('  [OK] import wesep / wespeaker 成功')
" 2>&1 | tail -5

if [ $ok -eq 0 ]; then
    echo ""
    echo "  [ERROR] 有缺失，请先在本地完整运行 download_local.sh 再传"
    exit 1
fi

echo ""
echo "============================================"
echo "  离线准备完成！"
echo "============================================"
echo ""
echo "  下一步评测:"
echo "    python tse_consistency_test.py --enroll_wav <A> --target_wav <A> \\"
echo "        --interferer_wav <B> --interferer2_wav <C> --out_dir ./out"
echo "    python tse_eres2net_eval.py --condition overlap2_en --balanced --out_dir ./eval_out"
echo "    bash run_tse_eval.sh"
