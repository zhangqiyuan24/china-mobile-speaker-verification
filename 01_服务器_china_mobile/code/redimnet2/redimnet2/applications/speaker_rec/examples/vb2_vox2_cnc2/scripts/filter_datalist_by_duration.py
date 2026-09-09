#!/usr/bin/env python3
"""Filter an EXISTING WeSpeaker datalist by per-utterance duration.

Unlike ``filter_datalists.py`` (which *rebuilds* a datalist from a durations file
by re-walking and re-parsing paths into spk/sess), this operates directly on an
existing ``data.list`` + ``utt2spk`` pair and a durations parquet/csv, joining on
the absolute wav path. It preserves the original ``key`` / ``spk`` / ``wav``
strings verbatim, so the filtered list is guaranteed to be a strict *subset* of
the input with identical speaker ids.

That subset property is the whole point for the PTN -> LM transition: the LM
speaker set is exactly the PTN speaker set (minus speakers all of whose utts were
shorter than the floor), so projection-head reassembly maps 1:1 onto the PTN
``spk2id.json`` and ``num_class`` is unchanged (under the default duration-only
mode).

Default behaviour: duration-only filter (drop utts shorter than --min-duration).
No upsampling, no speaker pruning. Optional speaker-level pruning is available via
--min-utts-per-speaker (0 = disabled).

The ``spk`` and ``key`` are read straight from ``data.list`` (every line in this
repo carries them), so ``utt2spk`` is *regenerated* from the surviving rows rather
than filtered in parallel -- this makes a data.list/utt2spk row-order mismatch
impossible.

Usage:
    python filter_datalist_by_duration.py \\
        --in-dir   /path/to/source_listdir \\
        --durations /path/to/durations.parquet \\
        --out-dir  /path/to/filtered_listdir \\
        --min-duration 5.0

    # also prune speakers left with < 15 utts after the duration cut:
    python filter_datalist_by_duration.py ... --min-utts-per-speaker 15
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def load_durations(path: str) -> pd.DataFrame:
    """Load a durations file (parquet or csv) -> DataFrame[path, duration]."""
    pl = path.lower()
    if pl.endswith('.parquet'):
        df = pd.read_parquet(path, columns=['path', 'duration'])
    elif pl.endswith('.csv'):
        df = pd.read_csv(path, usecols=['path', 'duration'])
    else:
        raise ValueError(f"Unknown durations file extension: {path}")
    # Collapse any duplicate paths (keep the first) so the merge stays 1:1.
    if df['path'].duplicated().any():
        n_dup = int(df['path'].duplicated().sum())
        print(f"  NOTE: durations file has {n_dup:,} duplicate paths; keeping first")
        df = df.drop_duplicates('path', keep='first')
    return df


def load_datalist(data_list_path: str) -> pd.DataFrame:
    """Load data.list (JSON lines) -> DataFrame[key, spk, wav]."""
    keys, spks, wavs = [], [], []
    with open(data_list_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            keys.append(e['key'])
            spks.append(e['spk'])
            wavs.append(e['wav'])
    return pd.DataFrame({'key': keys, 'spk': spks, 'wav': wavs})


def write_outputs(df: pd.DataFrame, out_dir: str) -> None:
    """Write out_dir/data.list (key,spk,wav json lines) + out_dir/utt2spk."""
    os.makedirs(out_dir, exist_ok=True)
    data_path = os.path.join(out_dir, 'data.list')
    utt_path = os.path.join(out_dir, 'utt2spk')

    # Stable, reproducible order.
    df = df.sort_values('key', kind='stable').reset_index(drop=True)

    # Vectorised JSON-line build (.map(json.dumps) escapes each field correctly).
    k = df['key'].map(json.dumps)
    s = df['spk'].map(json.dumps)
    w = df['wav'].map(json.dumps)
    data_lines = '{"key": ' + k + ', "spk": ' + s + ', "wav": ' + w + '}'
    with open(data_path, 'w') as f:
        f.write('\n'.join(data_lines.tolist()))
        f.write('\n')

    utt_lines = df['key'] + ' ' + df['spk']
    with open(utt_path, 'w') as f:
        f.write('\n'.join(utt_lines.tolist()))
        f.write('\n')

    print(f"  Wrote {data_path} ({len(df):,} lines)")
    print(f"  Wrote {utt_path}  ({len(df):,} lines)")


def _spk_stats(df: pd.DataFrame):
    return df['spk'].nunique(), len(df), df['duration'].sum() / 3600.0


def main():
    ap = argparse.ArgumentParser(
        description="Filter an existing datalist by utterance duration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--in-dir', required=True,
                    help='Source listdir containing data.list (utt2spk optional)')
    ap.add_argument('--durations', required=True,
                    help='Durations parquet/csv with columns [path, duration]')
    ap.add_argument('--out-dir', required=True, help='Output listdir')
    ap.add_argument('--min-duration', type=float, default=5.0,
                    help='Drop utterances shorter than this (seconds)')
    ap.add_argument('--max-duration', type=float, default=None,
                    help='Drop utterances longer than this (seconds)')
    ap.add_argument('--min-utts-per-speaker', type=int, default=0,
                    help='If >0, drop speakers left with fewer surviving utts')
    ap.add_argument('--drop-missing', action='store_true', default=True,
                    help='Drop utts with no duration entry (default: on)')
    ap.add_argument('--seed', type=int, default=42,
                    help='Unused; accepted for pipeline symmetry')
    args = ap.parse_args()

    data_list_path = os.path.join(args.in_dir, 'data.list')
    if not os.path.isfile(data_list_path):
        sys.exit(f"ERROR: no data.list in {args.in_dir}")

    print(f"Loading datalist  {data_list_path} ...")
    df = load_datalist(data_list_path)
    n0, s0 = len(df), df['spk'].nunique()
    print(f"  {n0:,} utts, {s0:,} speakers")

    print(f"Loading durations {args.durations} ...")
    dur = load_durations(args.durations)
    print(f"  {len(dur):,} duration rows")

    # Join durations onto the datalist by absolute wav path.
    df = df.merge(dur, how='left', left_on='wav', right_on='path')
    df = df.drop(columns=['path'])

    n_missing = int(df['duration'].isna().sum())
    if n_missing:
        frac = 100.0 * n_missing / max(n0, 1)
        msg = (f"  {n_missing:,} utts ({frac:.3f}%) have NO duration entry "
               f"(missing/errored in the durations file)")
        if frac > 1.0:
            print(f"  WARNING:{msg[1:]}")
            print("  WARNING: >1% of utts unmeasured -- durations file may be "
                  "stale/incomplete for this source.")
        else:
            print(msg)
    if args.drop_missing:
        df = df[df['duration'].notna()].reset_index(drop=True)

    print()
    print("=" * 70)
    print(f"DURATION FILTER  (min={args.min_duration}s"
          f"{f', max={args.max_duration}s' if args.max_duration else ''})")
    print("=" * 70)
    pre_s, pre_n, pre_h = _spk_stats(df)
    keep = df['duration'] >= args.min_duration
    if args.max_duration is not None:
        keep &= df['duration'] <= args.max_duration
    df = df[keep].reset_index(drop=True)
    post_s, post_n, post_h = _spk_stats(df)
    print(f"  utts:     {pre_n:,} -> {post_n:,}  "
          f"({100.0 * post_n / max(pre_n, 1):.1f}% kept, "
          f"{pre_n - post_n:,} dropped)")
    print(f"  speakers: {pre_s:,} -> {post_s:,}  ({pre_s - post_s:,} emptied)")
    print(f"  hours:    {pre_h:.1f} -> {post_h:.1f}")

    if args.min_utts_per_speaker and args.min_utts_per_speaker > 0:
        print()
        print("=" * 70)
        print(f"SPEAKER FILTER  (>= {args.min_utts_per_speaker} surviving utts)")
        print("=" * 70)
        counts = df.groupby('spk')['wav'].transform('count')
        pre_s, pre_n, _ = _spk_stats(df)
        df = df[counts >= args.min_utts_per_speaker].reset_index(drop=True)
        post_s, post_n, _ = _spk_stats(df)
        print(f"  utts:     {pre_n:,} -> {post_n:,}")
        print(f"  speakers: {pre_s:,} -> {post_s:,}  ({pre_s - post_s:,} dropped)")

    print()
    print("=" * 70)
    print("FINAL")
    print("=" * 70)
    fs, fn, fh = _spk_stats(df)
    print(f"  utts={fn:,}  speakers={fs:,}  hours={fh:.1f}")
    print()
    write_outputs(df, args.out_dir)


if __name__ == '__main__':
    main()
