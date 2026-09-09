#!/usr/bin/env python3
"""Infer audio durations via libsndfile headers (soundfile.info) -> parquet.

A ~100x faster drop-in alternative to infer_durations.py for header-readable
formats (WAV/FLAC/OGG): it reads only the file header (frames + samplerate) with
no per-file subprocess spawn, so throughput is thousands of files/s per pool
instead of tens. Output schema is identical to infer_durations.py
(['path', 'duration', 'sample_rate', 'error']) so downstream filters are
unchanged.

INPUT may be either:
  * a data.list (JSON lines, each with a 'wav' field) -> probes exactly those
    files (preferred: guarantees 100% datalist coverage, no dir-walk surprises);
  * a directory -> walks it for the given --exts.

Usage:
    python infer_durations_sf.py /path/to/data.list --output durations.parquet
    python infer_durations_sf.py /path/to/audio_dir --exts .wav .flac -o out.parquet
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
import soundfile as sf


def get_duration(file_path: str) -> dict:
    """Read header only; duration = frames / samplerate."""
    try:
        info = sf.info(file_path)
        sr = int(info.samplerate)
        dur = info.frames / sr if sr else None
        return {'path': file_path, 'duration': dur,
                'sample_rate': sr, 'error': None}
    except Exception as e:
        return {'path': file_path, 'duration': None,
                'sample_rate': None, 'error': f"{type(e).__name__}: {e}"}


def paths_from_datalist(data_list_path: str) -> list:
    paths = []
    with open(data_list_path) as f:
        for line in f:
            line = line.strip()
            if line:
                paths.append(json.loads(line)['wav'])
    return paths


def paths_from_dir(directory: str, exts) -> list:
    exts_set = set(e.lower() for e in exts)
    out = []
    for dp, _, fns in os.walk(directory):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in exts_set:
                out.append(os.path.join(dp, fn))
    return out


def main():
    ap = argparse.ArgumentParser(description="Infer durations via soundfile headers")
    ap.add_argument('input', help='A data.list (JSON lines w/ wav) OR an audio dir')
    ap.add_argument('--num-workers', '-n', type=int, default=min(96, cpu_count()))
    ap.add_argument('--exts', nargs='+',
                    default=['.wav', '.flac', '.ogg', '.opus'],
                    help='Extensions to include when input is a directory')
    ap.add_argument('--output', '-o', required=True, help='Output parquet path')
    args = ap.parse_args()

    if os.path.isfile(args.input):
        print(f"Reading file list from data.list {args.input} ...")
        files = paths_from_datalist(args.input)
    elif os.path.isdir(args.input):
        print(f"Scanning {args.input} for {args.exts} ...")
        files = paths_from_dir(args.input, args.exts)
    else:
        sys.exit(f"ERROR: input is neither file nor dir: {args.input}")

    print(f"Found {len(files):,} files; probing with {args.num_workers} workers...")
    sys.stdout.flush()

    t0 = time.time()
    results = []
    with Pool(args.num_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(get_duration, files, chunksize=256)):
            results.append(r)
            done = i + 1
            if done % 500000 == 0 or done == len(files):
                el = time.time() - t0
                print(f"  [{done:,}/{len(files):,}] {done/el:.0f}/s")
                sys.stdout.flush()

    df = pd.DataFrame.from_records(results)
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df['sample_rate'] = pd.to_numeric(df['sample_rate'], errors='coerce')
    ok = int(df['duration'].notna().sum())
    hrs = df.loc[df['duration'].notna(), 'duration'].sum() / 3600.0
    print(f"\nSuccess: {ok:,}/{len(df):,}   total {hrs:.1f} h ({hrs/24:.1f} d)")
    errs = df[df['error'].notna()]
    if len(errs):
        print(f"Errors: {len(errs):,}")
        for _, row in errs.head(10).iterrows():
            print(f"  {row['path']}: {row['error']}")

    out = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    df.to_parquet(out)
    print(f"Wrote {out}   ({(time.time()-t0)/60:.1f} min)")


if __name__ == '__main__':
    main()
