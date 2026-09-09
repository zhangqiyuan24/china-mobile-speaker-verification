#!/usr/bin/env python3
"""Join N datalist directories (data.list + utt2spk) by simple concatenation.

Unlike mix_datalists.py, this performs NO up/down-sampling — every utterance
from every source appears exactly once. Each source's speakers (and keys) get
namespaced with a unique prefix so cross-dataset speaker-id collisions
(e.g. VC2 id00800 vs CN-Celeb2 id00800) are impossible.

Sources sort in the order given on the command line, which determines the
contiguous spk2id index ranges (first source gets indices 0..n1-1, next
n1..n1+n2-1, etc. -- assuming prefixes are alphabetically increasing).

Usage:
    python join_datalists.py \\
        --source vc2:A:/path/to/vox2_dev \\
        --source vb2:B:/path/to/vb2_full \\
        --source cnc2:C:/path/to/cnc2_full \\
        --output /path/to/mix_join \\
        --seed 42
"""
import argparse
import json
import os
import random


def load_datalist(data_list_path):
    """Load data.list (JSON lines) and return list of dicts."""
    entries = []
    with open(data_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_utt2spk(utt2spk_path):
    """Load utt2spk and return list of (key, spk) tuples."""
    entries = []
    with open(utt2spk_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                entries.append((parts[0], parts[1]))
    return entries


def parse_source(spec):
    """Parse a 'name:prefix:dir' source spec."""
    parts = spec.split(':', 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--source must be 'name:prefix:dir', got: {spec}"
        )
    name, prefix, dir_path = parts
    if not name or not prefix or not dir_path:
        raise argparse.ArgumentTypeError(
            f"--source name/prefix/dir must all be non-empty: {spec}"
        )
    return name, prefix, dir_path


def main():
    parser = argparse.ArgumentParser(description="Join N datalist dirs (no resampling)")
    parser.add_argument('--source', action='append', required=True, type=parse_source,
                        help='Triple "name:prefix:dir" — repeatable, order matters')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for deterministic shuffle')
    parser.add_argument('--no-shuffle', action='store_true',
                        help='Skip the final shuffle (preserve source-by-source order)')
    args = parser.parse_args()

    if len(args.source) < 2:
        parser.error("Need at least 2 sources to join")

    prefixes = [p for _, p, _ in args.source]
    if len(set(prefixes)) != len(prefixes):
        parser.error(f"Source prefixes must be unique, got {prefixes}")

    # Load each source, applying the prefix to spk AND key.
    per_source_stats = []
    all_entries = []   # list of dicts going into data.list
    all_utt2spk = []   # list of (key, spk) going into utt2spk

    for name, prefix, dir_path in args.source:
        data_path = os.path.join(dir_path, 'data.list')
        utt_path = os.path.join(dir_path, 'utt2spk')
        print(f"[load] {name} (prefix={prefix}_) from {dir_path}")
        ds_data = load_datalist(data_path)
        ds_utt = load_utt2spk(utt_path)
        if len(ds_data) != len(ds_utt):
            raise SystemExit(
                f"  {name}: data.list ({len(ds_data)}) and utt2spk "
                f"({len(ds_utt)}) length mismatch"
            )
        n_utts = len(ds_data)
        n_spks = len(set(e['spk'] for e in ds_data))
        print(f"  -> {n_utts:,} utts, {n_spks:,} speakers")
        per_source_stats.append((name, prefix, n_utts, n_spks))

        for entry, (utt_key, spk) in zip(ds_data, ds_utt):
            # Sanity: utt2spk key should match data.list key.
            # (Tolerated if not — we just use whichever is in data.list.)
            new_spk = f"{prefix}_{spk}"
            new_key = f"{prefix}_{entry['key']}"
            new_utt_key = f"{prefix}_{utt_key}"
            all_entries.append({
                "key": new_key,
                "spk": new_spk,
                "wav": entry["wav"],
            })
            all_utt2spk.append((new_utt_key, new_spk))

    # Optional global shuffle (deterministic).
    if not args.no_shuffle:
        rng = random.Random(args.seed)
        order = list(range(len(all_entries)))
        rng.shuffle(order)
        all_entries = [all_entries[i] for i in order]
        all_utt2spk = [all_utt2spk[i] for i in order]

    # Write outputs.
    os.makedirs(args.output, exist_ok=True)
    data_list_out = os.path.join(args.output, 'data.list')
    utt2spk_out = os.path.join(args.output, 'utt2spk')
    with open(data_list_out, 'w') as f_data, open(utt2spk_out, 'w') as f_utt:
        for entry, (utt_key, spk) in zip(all_entries, all_utt2spk):
            f_data.write(json.dumps(entry) + '\n')
            f_utt.write(f"{utt_key} {spk}\n")

    # Stats summary.
    total_utts = sum(s[2] for s in per_source_stats)
    total_spks = sum(s[3] for s in per_source_stats)
    print()
    print("=" * 78)
    print(f"  JOINED DATALIST  (output: {args.output})")
    print("=" * 78)
    print(f"  {'name':<10} {'prefix':<8} {'utts':>14} {'%utts':>8} "
          f"{'speakers':>12} {'%spks':>8}")
    for name, prefix, n_utts, n_spks in per_source_stats:
        pct_utts = 100.0 * n_utts / total_utts if total_utts else 0
        pct_spks = 100.0 * n_spks / total_spks if total_spks else 0
        print(f"  {name:<10} {prefix + '_':<8} {n_utts:>14,} {pct_utts:>7.2f}% "
              f"{n_spks:>12,} {pct_spks:>7.2f}%")
    print(f"  {'-' * 76}")
    print(f"  {'TOTAL':<10} {'':<8} {total_utts:>14,} {'100.00%':>8} "
          f"{total_spks:>12,} {'100.00%':>8}")
    print()
    print(f"  Wrote {data_list_out} ({len(all_entries):,} lines)")
    print(f"  Wrote {utt2spk_out} ({len(all_utt2spk):,} lines)")
    if not args.no_shuffle:
        print(f"  Shuffled with seed={args.seed}")


if __name__ == '__main__':
    main()
