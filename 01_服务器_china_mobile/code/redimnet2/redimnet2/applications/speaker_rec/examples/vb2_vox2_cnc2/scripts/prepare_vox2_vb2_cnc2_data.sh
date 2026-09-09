#!/bin/bash
set -e

# =============================================================================
# prepare_vox2_vb2_cnc2_data.sh — pretraining data prep for the
# VoxCeleb2 + VoxBlink2 + CN-Celeb2 (OpenSLR-82) mix.
#
# Differences from prepare_vox2_vb2_data.sh:
#   * Adds CN-Celeb2 as a third source.
#   * No filtering on VB2 / CN-Celeb2 — full datalists.
#   * No proportional up/down-sampling — sources joined verbatim and shuffled,
#     speakers (and keys) namespaced A_/B_/C_ to avoid id collisions.
#
# This script does not download or convert the source corpora. Prepare
# VoxCeleb2/VoxCeleb1/MUSAN with applications/speaker_rec/examples/vox2/scripts/prepare_data.py, acquire
# VoxBlink2 and CN-Celeb2 separately, then use this script to build manifests.
#
# Steps:
#   1. Validate the existing CN-Celeb2 FLAC and VoxBlink2 WAV trees
#   2. Generate CN-Celeb2 datalist (FLAC, flat layout)
#   3. Generate full VoxBlink2 datalist (no filter, WAV)
#   4. Reuse existing VoxCeleb2 datalist
#   5. Join all three with A_/B_/C_ prefix and shuffle; print stats
#
# Usage:
#   ./scripts/prepare_vox2_vb2_cnc2_data.sh \
#       --vox2-list /data/wespeaker_data/vox2_dev \
#       --vb2-root  /data/voxblink2 \
#       --cnc2-root /data/cnceleb2 \
#       --output-root /data/wespeaker_data/vox2_vb2_cnc2_v0
# =============================================================================

VOX2_LIST=""        # dir containing existing data.list + utt2spk
VB2_ROOT=""         # dir containing audio/ subdir with speaker folders
CNC2_ROOT=""        # dir containing CN-Celeb2_flac/data speaker folders
OUTPUT_ROOT=""
SEED=42

while [[ $# -gt 0 ]]; do
    case $1 in
        --vox2-list)   VOX2_LIST="$2"; shift 2;;
        --vb2-root)    VB2_ROOT="$2"; shift 2;;
        --cnc2-root)   CNC2_ROOT="$2"; shift 2;;
        --output-root) OUTPUT_ROOT="$2"; shift 2;;
        --seed)        SEED="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if [ -z "$VOX2_LIST" ] || [ -z "$VB2_ROOT" ] || [ -z "$CNC2_ROOT" ] || [ -z "$OUTPUT_ROOT" ]; then
    cat <<EOF
Usage: $0 --vox2-list <dir> --vb2-root <dir> --cnc2-root <dir> --output-root <dir>

  --vox2-list  Existing VC2 datalist dir (must contain data.list + utt2spk)
  --vb2-root   VB2 root with an audio/ subdir of speaker folders
  --cnc2-root  CN-Celeb2 root containing CN-Celeb2_flac/data
  --output-root  Where datalists will be written

Optional:
  --seed       Shuffle seed for the final join (default: 42)
EOF
    exit 1
fi

# Sanity: VC2 datalist must already exist (this pipeline doesn't regen it).
if [ ! -f "$VOX2_LIST/data.list" ] || [ ! -f "$VOX2_LIST/utt2spk" ]; then
    echo "ERROR: VC2 datalist missing at $VOX2_LIST (need data.list + utt2spk)"
    exit 1
fi
if [ ! -d "$VB2_ROOT/audio" ]; then
    echo "ERROR: VB2 audio directory not found: $VB2_ROOT/audio"
    exit 1
fi
if [ ! -d "$CNC2_ROOT/CN-Celeb2_flac/data" ]; then
    echo "ERROR: CN-Celeb2 data directory not found: $CNC2_ROOT/CN-Celeb2_flac/data"
    echo "       Download/extract CN-Celeb2 from https://www.openslr.org/82/ first."
    exit 1
fi

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "================================================================"
echo "  VC2 + VB2 + CN-Celeb2 data preparation pipeline"
echo "================================================================"
echo "  VC2 datalist (reuse): $VOX2_LIST"
echo "  VB2 audio root:       $VB2_ROOT"
echo "  CN-Celeb2 root:       $CNC2_ROOT"
echo "  Output root:          $OUTPUT_ROOT"
echo "  Seed:                 $SEED"
echo ""

mkdir -p "$OUTPUT_ROOT"

# =====================================================================
# STEP 1: Validate existing source trees
# =====================================================================
echo "[STEP 1] Source trees found"
echo "  VB2:       $VB2_ROOT/audio"
echo "  CN-Celeb2: $CNC2_ROOT/CN-Celeb2_flac/data"
echo ""

# =====================================================================
# STEP 2: CN-Celeb2 datalist (FLAC, flat layout — spk/utt.flac)
# =====================================================================
CNC2_LISTS="$OUTPUT_ROOT/cnc2_full"
if [ -f "$CNC2_LISTS/data.list" ]; then
    echo "[STEP 2] CN-Celeb2 datalist already exists at $CNC2_LISTS — skipping"
else
    echo "[STEP 2] Generating CN-Celeb2 datalist..."
    python3 "$SCRIPTS_DIR/gen_datalists.py" "$CNC2_ROOT" "$CNC2_LISTS" \
        --audio-subdir CN-Celeb2_flac/data \
        --exts .flac
fi
echo ""

# =====================================================================
# STEP 3: Full VoxBlink2 datalist (no filtering)
# =====================================================================
VB2_LISTS="$OUTPUT_ROOT/vb2_full"
if [ -f "$VB2_LISTS/data.list" ]; then
    echo "[STEP 3] VB2 datalist already exists at $VB2_LISTS — skipping"
else
    echo "[STEP 3] Generating full VB2 datalist (no filter)..."
    python3 "$SCRIPTS_DIR/gen_datalists.py" "$VB2_ROOT" "$VB2_LISTS" \
        --audio-subdir audio
fi
echo ""

# =====================================================================
# STEP 4: VoxCeleb2 — reuse existing datalist (no regen)
# =====================================================================
echo "[STEP 4] Using existing VC2 datalist at $VOX2_LIST (no regeneration)"
echo ""

# =====================================================================
# STEP 5: Join all three with A_/B_/C_ prefix and shuffle
# =====================================================================
MIX_DIR="$OUTPUT_ROOT/mix_join"
if [ -f "$MIX_DIR/data.list" ]; then
    echo "[STEP 5] Joined datalist already exists at $MIX_DIR — skipping"
    echo "         (delete $MIX_DIR/data.list to force a rebuild)"
else
    echo "[STEP 5] Joining all sources..."
    python3 "$SCRIPTS_DIR/join_datalists.py" \
        --source "vc2:A:$VOX2_LIST" \
        --source "vb2:B:$VB2_LISTS" \
        --source "cnc2:C:$CNC2_LISTS" \
        --output "$MIX_DIR" \
        --seed "$SEED"
fi
echo ""

# =====================================================================
# Summary
# =====================================================================
echo "================================================================"
echo "  DONE"
echo "================================================================"
echo ""
echo "  Outputs:"
echo "    CN-Celeb2 lists:  $CNC2_LISTS/"
echo "    VB2 full lists:   $VB2_LISTS/"
echo "    VC2 lists (used): $VOX2_LIST/  (unchanged)"
echo "    Joined train mix: $MIX_DIR/"
echo ""
echo "  For training configs, use:"
echo "    train_data:  $MIX_DIR/data.list"
echo "    train_label: $MIX_DIR/utt2spk"
