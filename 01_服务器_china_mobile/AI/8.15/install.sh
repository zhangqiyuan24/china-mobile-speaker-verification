#!/bin/bash
# ============================================================
# 服务器（无网络）运行 — 部署模型 + patch 推理脚本，实现断网评测
#
# 前提: 已把 AI/8.15/ 整个目录传到服务器
#
# 作用:
#   1. 把 3 个模型 checkpoint 复制到 sv-ecapa/pretrained/ 正确位置
#   2. patch infer_sv_batch.py / infer_sv_new.py：
#      - 跳过 is_official_hub_path 联网验证
#      - 模型从本地 pretrained 目录加载（不联网）
#
# 用法: bash install.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ★ 改成你服务器的 sv-ecapa 目录
SV_ECAPA="${SV_ECAPA:-$HOME/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa}"
SPEAKERLAB_BIN="$SV_ECAPA/../../../speakerlab/bin"

echo "============================================"
echo "  断网评测部署"
echo "  SV_ECAPA = $SV_ECAPA"
echo "============================================"

if [ ! -d "$SV_ECAPA" ]; then
    echo "[ERROR] sv-ecapa 目录不存在: $SV_ECAPA"
    echo "  请设置 SV_ECAPA 环境变量指向正确的 sv-ecapa 目录"
    exit 1
fi

# ============================================================
# 1/3: 部署模型到 sv-ecapa/pretrained/
# ============================================================
echo ""
echo "==== 1/3: 部署模型 ===="
PRETRAINED_DIR="$SV_ECAPA/pretrained"
mkdir -p "$PRETRAINED_DIR/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k"
mkdir -p "$PRETRAINED_DIR/speech_eres2net_large_sv_zh-cn_3dspeaker_16k"
mkdir -p "$PRETRAINED_DIR/redimnet2"

cp -f models/ecapa-tdnn.ckpt \
      "$PRETRAINED_DIR/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k/ecapa-tdnn.ckpt"
cp -f models/eres2net_large_model.ckpt \
      "$PRETRAINED_DIR/speech_eres2net_large_sv_zh-cn_3dspeaker_16k/eres2net_large_model.ckpt"
cp -f models/b6-vb2+vox2+cnc2_v0-lm.pt \
      "$PRETRAINED_DIR/redimnet2/b6-vb2+vox2+cnc2_v0-lm.pt"

echo "  模型已部署:"
ls -la "$PRETRAINED_DIR"/*/ 2>/dev/null | grep -E "ckpt|pt|:$" || true

# ============================================================
# 2/3: 替换 + patch 推理脚本
# ============================================================
echo ""
echo "==== 2/3: 替换 + patch 推理脚本 (跳过联网) ===="

# 先备份服务器原脚本
cp "$SPEAKERLAB_BIN/infer_sv_batch.py" "$SPEAKERLAB_BIN/infer_sv_batch.py.bak" 2>/dev/null || true
cp "$SPEAKERLAB_BIN/infer_sv_new.py" "$SPEAKERLAB_BIN/infer_sv_new.py.bak" 2>/dev/null || true

# 用 AI/8.15 里的修改版替换 (本地已加 --pretrained / ReDimNet2 支持)
cp -f "$SCRIPT_DIR/infer_sv_batch.py" "$SPEAKERLAB_BIN/infer_sv_batch.py"
cp -f "$SCRIPT_DIR/infer_sv_new.py" "$SPEAKERLAB_BIN/infer_sv_new.py"
echo "  已替换 infer_sv_batch.py / infer_sv_new.py"

# 再 patch (跳过 is_official_hub_path + else 本地加载)
export SPEAKERLAB_BIN="$SPEAKERLAB_BIN"
python << 'PYEOF'
import os

speakerlab_bin = os.path.abspath(os.environ["SPEAKERLAB_BIN"])

# ---- patch infer_sv_batch.py ----
batch_path = os.path.join(speakerlab_bin, "infer_sv_batch.py")
with open(batch_path) as f:
    code = f.read()

# 1) 跳过 is_official_hub_path 联网验证
old = 'is_official_hub_path(args.model_id)'
new = 'True'
assert old in code, "infer_sv_batch.py: 找不到 is_official_hub_path 行"
code = code.replace(old, new)

# 2) else 分支优先本地加载
old_else = '''    else:
        # ---- Original: download from ModelScope ----
        cache_dir = snapshot_download(
                    args.model_id,
                    revision=conf['revision'],
                    )
        cache_dir = pathlib.Path(cache_dir)

        # link
        download_files = ['examples', conf['model_pt']]
        for src in cache_dir.glob('*'):
            if re.search('|'.join(download_files), src.name):
                dst = args.local_model_dir / src.name
                try:
                    dst.unlink()
                except FileNotFoundError:
                    pass
                dst.symlink_to(src)

        pretrained_model = args.local_model_dir / conf['model_pt']
        pretrained_state = torch.load(pretrained_model, map_location='cpu')
        embedding_model.load_state_dict(pretrained_state)'''
new_else = '''    else:
        local_pt = args.local_model_dir / conf['model_pt']
        if not local_pt.exists():
            # ---- Original: download from ModelScope ----
            cache_dir = snapshot_download(
                        args.model_id,
                        revision=conf['revision'],
                        )
            cache_dir = pathlib.Path(cache_dir)

            # link
            download_files = ['examples', conf['model_pt']]
            for src in cache_dir.glob('*'):
                if re.search('|'.join(download_files), src.name):
                    dst = args.local_model_dir / src.name
                    try:
                        dst.unlink()
                    except FileNotFoundError:
                        pass
                    dst.symlink_to(src)

        pretrained_model = args.local_model_dir / conf['model_pt']
        pretrained_state = torch.load(pretrained_model, map_location='cpu')
        embedding_model.load_state_dict(pretrained_state)'''
assert old_else in code, "infer_sv_batch.py: 找不到 else 分支"
code = code.replace(old_else, new_else)

with open(batch_path, "w") as f:
    f.write(code)
print(f"  [OK] patched infer_sv_batch.py")

# ---- patch infer_sv_new.py ----
new_path = os.path.join(speakerlab_bin, "infer_sv_new.py")
with open(new_path) as f:
    code = f.read()

# 1) 跳过 is_official_hub_path
old = 'is_official_hub_path(args.model_id)'
new = 'True'
assert old in code, "infer_sv_new.py: 找不到 is_official_hub_path 行"
code = code.replace(old, new)

# 2) ReDimNet2 本地加载
old_red = '''def _load_redimnet2_from_github(model_name, train_type, dataset):
    """Download ReDimNet2 weights from GitHub and build the model.

    Returns (embedding_model, is_raw_wav_input, embedding_dir)
    where is_raw_wav_input=True means the model expects raw waveform directly.
    """
    from redimnet2 import load_custom

    print(f'[INFO]: Loading ReDimNet2-{model_name} ({dataset}, {train_type}) from GitHub...')
    model = load_custom(model_name=model_name, train_type=train_type, dataset=dataset)
    model.eval()
    print(f'[INFO]: ReDimNet2 loaded successfully.')
    return model, True  # ReDimNet2 takes raw waveform as input'''
new_red = '''def _load_redimnet2_from_github(model_name, train_type, dataset):
    """Load ReDimNet2 weights (local file first, fallback to GitHub)."""
    from redimnet2.redimnet2 import ReDimNet2Wrap

    weight_name = f"{model_name}-{dataset}-{train_type}.pt"
    local_pt = os.path.join(os.getcwd(), 'pretrained', 'redimnet2', weight_name)
    if os.path.exists(local_pt):
        print(f'[INFO]: Loading ReDimNet2 from local {local_pt} ...')
        full_state_dict = torch.load(local_pt, map_location='cpu')
        model = ReDimNet2Wrap(**full_state_dict["model_config"])
        model.load_state_dict(full_state_dict["state_dict"])
        model.eval()
        return model, True
    from redimnet2 import load_custom
    print(f'[INFO]: Loading ReDimNet2-{model_name} ({dataset}, {train_type}) from GitHub...')
    model = load_custom(model_name=model_name, train_type=train_type, dataset=dataset)
    model.eval()
    return model, True'''
assert old_red in code, "infer_sv_new.py: 找不到 _load_redimnet2_from_github"
code = code.replace(old_red, new_red)

with open(new_path, "w") as f:
    f.write(code)
print(f"  [OK] patched infer_sv_new.py")
PYEOF

# ============================================================
# 3/3: 验证
# ============================================================
echo ""
echo "==== 3/3: 验证 ===="
echo "  模型文件:"
for f in "$PRETRAINED_DIR/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k/ecapa-tdnn.ckpt" \
         "$PRETRAINED_DIR/speech_eres2net_large_sv_zh-cn_3dspeaker_16k/eres2net_large_model.ckpt" \
         "$PRETRAINED_DIR/redimnet2/b6-vb2+vox2+cnc2_v0-lm.pt"; do
    if [ -s "$f" ]; then
        echo "  [OK] $f ($(du -h "$f" | cut -f1))"
    else
        echo "  [ERROR] $f 缺失"
    fi
done

echo ""
echo "============================================"
echo "  部署完成！"
echo "============================================"
echo ""
echo "  现在可以断网跑基线评测:"
echo "    cd $SV_ECAPA"
echo "    bash evaluate_clean.sh"
echo "    bash evaluate_overlap.sh"
echo "    bash evaluate_noise.sh"
echo "    bash evaluate_reverb.sh"
