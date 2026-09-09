#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FULLSUBNET_ROOT="$(cd "$SCRIPT_DIR/../8.17/code" && pwd)"
FULLSUBNET_PYROOT="$FULLSUBNET_ROOT/speech_enhance"
SPEAKERLAB_ROOT="$(cd "$SCRIPT_DIR/../../code/3D-speaker/3D-speaker" && pwd)"

export PYTHONPATH="$FULLSUBNET_PYROOT:$FULLSUBNET_ROOT:$SPEAKERLAB_ROOT:$SCRIPT_DIR:$PYTHONPATH"

# 使用物理 GPU 1 和 3；进程内会映射为 cuda:0 / cuda:1
GPUS="${GPUS:-1,3}"
NPROC="${NPROC:-2}"

echo "============================================"
echo "  Speaker Preservation fine-tuning (DDP)"
echo "  Physical GPUs: $GPUS"
echo "  Processes: $NPROC"
echo "  FullSubNet root: $FULLSUBNET_ROOT"
echo "  FullSubNet Python root: $FULLSUBNET_PYROOT"
echo "  3D-Speaker root: $SPEAKERLAB_ROOT"
echo "============================================"

echo
echo "==== Python import sanity ===="
python - <<'PY'
import audio_zen
import speakerlab
print("audio_zen import OK:", getattr(audio_zen, "__file__", None))
print("speakerlab import OK:", getattr(speakerlab, "__file__", None))
PY
echo

ARGS=()
if [ "${1:-}" == "--sanity" ]; then
    echo "  [sanity run] 3 epochs, 均匀抽取 10000 条 clean"
    ARGS+=(--sanity --sanity-samples 10000)
else
    echo "  [正式训练] 10 epochs, 均匀抽取条数见 train_spk_preserve.toml (max_train_samples)"
fi

CUDA_VISIBLE_DEVICES="$GPUS" \
python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC" \
    train_spk_preserve.py "${ARGS[@]}"
