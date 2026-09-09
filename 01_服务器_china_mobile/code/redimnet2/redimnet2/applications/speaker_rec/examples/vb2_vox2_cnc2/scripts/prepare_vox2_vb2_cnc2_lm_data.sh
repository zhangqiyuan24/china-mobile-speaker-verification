#!/bin/bash
set -e

# =============================================================================
# prepare_vox2_vb2_cnc2_lm_data.sh — LARGE-MARGIN data prep for the
# VoxCeleb2 + VoxBlink2 + CN-Celeb2 mix.
#
# This is the LM companion to prepare_vox2_vb2_cnc2_data.sh (the PTN prep).
# It takes the SAME three per-source datalists that PTN was built from, filters
# out short utterances, and verbatim-joins the survivors with the SAME A_/B_/C_
# prefixes and source order — so the LM speaker set is a strict subset of the
# PTN set and projection-head reassembly maps 1:1 onto the PTN spk2id.json.
#
# Differences from prepare_vox2_vb2_data.sh (the vox2+vb2 LM prep):
#   * 3 sources (adds CN-Celeb2).
#   * Per-source DURATION filter only — NO min-sessions/min-utts speaker pruning
#     by default, NO mixup ratio forcing, NO upsampling of minority/vox2.
#   * Sources are verbatim-joined (join_datalists.py), so dataset proportions
#     follow the natural post-filter survivor counts, exactly like the PTN mix.
#
# Steps:
#   0. Ensure a durations parquet for each source (reuse VB2's; infer VC2/CNC2)
#   1. Filter VC2  by min-duration            -> <out>/vox2_lm
#   2. Filter VB2  by min-duration            -> <out>/vb2_lm
#   3. Filter CNC2 by min-duration            -> <out>/cnc2_lm
#   4. Verbatim-join A_/B_/C_ + shuffle       -> <out>/mix_join_lm
#
# Usage:
#   ./scripts/prepare_vox2_vb2_cnc2_lm_data.sh \
#       --vox2-list /data/wespeaker_data/vox2_dev \
#       --vb2-list  /data/wespeaker_data/vox2_vb2_cnc2_v0/vb2_full \
#       --cnc2-list /data/wespeaker_data/vox2_vb2_cnc2_v0/cnc2_full \
#       --output-root /data/wespeaker_data/vox2_vb2_cnc2_v0 \
#       --min-duration 5.0
# =============================================================================

VOX2_LIST=""        # dir with VC2 data.list + utt2spk (PTN's vc2:A source)
VB2_LIST=""         # dir with VB2 data.list + utt2spk (PTN's vb2:B source)
CNC2_LIST=""        # dir with CN-Celeb2 data.list + utt2spk (PTN's cnc2:C source)
OUTPUT_ROOT=""

# Durations parquets. Set --vb2-durations if your VoxBlink2 package includes
# one; otherwise all sources are inferred into the manifest directories.
# "AUTO" => derive + infer as <listdir>/durations.parquet.
VB2_DURATIONS="AUTO"
VOX2_DURATIONS="AUTO"
CNC2_DURATIONS="AUTO"

MIN_DUR=5.0             # short-utterance floor (seconds)
MIN_UTTS=0              # 0 = duration-only (keep all speakers). >0 prunes speakers.
DURATION_WORKERS=96     # ffprobe workers for VC2/CNC2 inference
SEED=42

while [[ $# -gt 0 ]]; do
    case $1 in
        --vox2-list)        VOX2_LIST="$2"; shift 2;;
        --vb2-list)         VB2_LIST="$2"; shift 2;;
        --cnc2-list)        CNC2_LIST="$2"; shift 2;;
        --output-root)      OUTPUT_ROOT="$2"; shift 2;;
        --vb2-durations)    VB2_DURATIONS="$2"; shift 2;;
        --vox2-durations)   VOX2_DURATIONS="$2"; shift 2;;
        --cnc2-durations)   CNC2_DURATIONS="$2"; shift 2;;
        --min-duration)     MIN_DUR="$2"; shift 2;;
        --min-utts-per-speaker) MIN_UTTS="$2"; shift 2;;
        --duration-workers) DURATION_WORKERS="$2"; shift 2;;
        --seed)             SEED="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if [ -z "$VOX2_LIST" ] || [ -z "$VB2_LIST" ] || [ -z "$CNC2_LIST" ] || [ -z "$OUTPUT_ROOT" ]; then
    cat <<EOF
Usage: $0 --vox2-list <dir> --vb2-list <dir> --cnc2-list <dir> --output-root <dir>

  --vox2-list  VC2 datalist dir       (data.list + utt2spk)  -> joined as A_
  --vb2-list   VoxBlink2 datalist dir (data.list + utt2spk)  -> joined as B_
  --cnc2-list  CN-Celeb2 datalist dir (data.list + utt2spk)  -> joined as C_
  --output-root  Where filtered + joined LM datalists are written

Optional:
  --vb2-durations  PARQUET   (default: AUTO -> infer next to VB2 list)
  --vox2-durations PARQUET   (default: AUTO -> infer next to VC2 audio)
  --cnc2-durations PARQUET   (default: AUTO -> infer next to CNC2 audio)
  --min-duration   SECONDS   (default: 5.0)
  --min-utts-per-speaker N   (default: 0 = duration-only, keep all speakers)
  --duration-workers N       (default: 96)
  --seed N                   (default: 42)
EOF
    exit 1
fi

for d in "$VOX2_LIST" "$VB2_LIST" "$CNC2_LIST"; do
    if [ ! -f "$d/data.list" ] || [ ! -f "$d/utt2spk" ]; then
        echo "ERROR: missing data.list or utt2spk in $d"; exit 1
    fi
done

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUTPUT_ROOT"

echo "================================================================"
echo "  VC2 + VB2 + CN-Celeb2  LM data preparation (duration filter)"
echo "================================================================"
echo "  VC2  list:    $VOX2_LIST"
echo "  VB2  list:    $VB2_LIST"
echo "  CNC2 list:    $CNC2_LIST"
echo "  Output root:  $OUTPUT_ROOT"
echo "  Min duration: ${MIN_DUR}s   Min utts/spk: ${MIN_UTTS}   Seed: $SEED"
echo ""

# Ensure a durations parquet exists for a source; infer (soundfile) if missing.
# Inference reads the data.list directly, so coverage is exactly the datalist.
#   $1 listdir   $2 durations-path-or-AUTO   echoes the resolved parquet path
ensure_durations() {
    local listdir="$1"; local parquet="$2"
    if [ "$parquet" = "AUTO" ]; then
        parquet="$listdir/durations.parquet"
    fi
    if [ -f "$parquet" ]; then
        echo "    durations present: $parquet" 1>&2
    else
        echo "    inferring durations (soundfile, $DURATION_WORKERS workers) -> $parquet" 1>&2
        python3 "$SCRIPTS_DIR/infer_durations_sf.py" "$listdir/data.list" \
            --num-workers "$DURATION_WORKERS" \
            --output "$parquet" 1>&2
    fi
    echo "$parquet"
}

# =====================================================================
# STEP 0: durations for every source
# =====================================================================
echo "[STEP 0] Resolving durations files"
echo "  VB2:"
if [ ! -f "$VB2_DURATIONS" ]; then
    VB2_DURATIONS="$(ensure_durations "$VB2_LIST" "AUTO")"
else
    echo "    durations present: $VB2_DURATIONS"
fi
echo "  VC2:";  VOX2_DURATIONS="$(ensure_durations "$VOX2_LIST" "$VOX2_DURATIONS")"
echo "  CNC2:"; CNC2_DURATIONS="$(ensure_durations "$CNC2_LIST" "$CNC2_DURATIONS")"
echo ""

# =====================================================================
# STEPS 1-3: per-source duration filter
# =====================================================================
VOX2_LM="$OUTPUT_ROOT/vox2_lm"
VB2_LM="$OUTPUT_ROOT/vb2_lm"
CNC2_LM="$OUTPUT_ROOT/cnc2_lm"

filter_one() {
    local name="$1"; local listdir="$2"; local durations="$3"; local outdir="$4"
    if [ -f "$outdir/data.list" ]; then
        echo "[filter $name] exists at $outdir — skipping"
        return
    fi
    echo "[filter $name] $listdir  (min ${MIN_DUR}s) -> $outdir"
    python3 "$SCRIPTS_DIR/filter_datalist_by_duration.py" \
        --in-dir "$listdir" \
        --durations "$durations" \
        --out-dir "$outdir" \
        --min-duration "$MIN_DUR" \
        --min-utts-per-speaker "$MIN_UTTS" \
        --seed "$SEED"
    echo ""
}

echo "[STEP 1] Filter VC2"
filter_one vc2  "$VOX2_LIST" "$VOX2_DURATIONS" "$VOX2_LM"
echo "[STEP 2] Filter VB2"
filter_one vb2  "$VB2_LIST"  "$VB2_DURATIONS"  "$VB2_LM"
echo "[STEP 3] Filter CNC2"
filter_one cnc2 "$CNC2_LIST" "$CNC2_DURATIONS" "$CNC2_LM"

# =====================================================================
# STEP 4: verbatim join (A_/B_/C_, same order as the PTN mix)
# =====================================================================
MIX_LM="$OUTPUT_ROOT/mix_join_lm"
if [ -f "$MIX_LM/data.list" ]; then
    echo "[STEP 4] Joined LM datalist already exists at $MIX_LM — skipping"
    echo "         (delete $MIX_LM/data.list to force a rebuild)"
else
    echo "[STEP 4] Verbatim-joining filtered sources (A_/B_/C_)..."
    python3 "$SCRIPTS_DIR/join_datalists.py" \
        --source "vc2:A:$VOX2_LM" \
        --source "vb2:B:$VB2_LM" \
        --source "cnc2:C:$CNC2_LM" \
        --output "$MIX_LM" \
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
echo "  Filtered sources:  $VOX2_LM/  $VB2_LM/  $CNC2_LM/"
echo "  Joined LM mix:     $MIX_LM/"
echo ""
echo "  For the LM config (do_lm: true, speed_perturb: false), set:"
echo "    train_data:  $MIX_LM/data.list"
echo "    train_label: $MIX_LM/utt2spk"
echo "    num_class is auto-computed from train_label by train.py"
echo ""
echo "  Init the LM stage from the PTN checkpoint; head reassembly uses the PTN"
echo "  spk2id.json (matching A_/B_/C_ speakers by name)."
