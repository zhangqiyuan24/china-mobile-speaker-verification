#!/bin/bash
set -e

CONFIG=${1:-}

if [ -z "$CONFIG" ]; then
    echo "Usage: $0 <config.yaml> [num_gpus] [training overrides ...]"
    exit 1
fi

shift
NUM_GPUS=8
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
    NUM_GPUS=$1
    shift
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/applications/speaker_rec:${PYTHONPATH}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29400}"

export LOGLEVEL="ERROR"
export PYTHONFAULTHANDLER=1

echo "========================================================"
echo "  TRAINING"
echo "  Config: $CONFIG"
echo "  GPUs:   $NUM_GPUS"
echo "========================================================"

torchrun --nnodes=1:1 --nproc_per_node="$NUM_GPUS" --rdzv-backend=static \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" --master-addr="$MASTER_ADDR" \
    "$REPO_ROOT/applications/speaker_rec/wespeaker_lite/bin/train.py" \
    --config "$CONFIG" "$@"

echo ""
echo "========================================================"
echo "  TRAINING DONE"
echo "========================================================"
