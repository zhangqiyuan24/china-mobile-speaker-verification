#!/usr/bin/env python3
"""Generate data.list and utt2spk for a speaker verification dataset.

Expects directory structure: root_dir/audio/spk_id/video_id/NNNNN.wav
Produces:
  output_dir/data.list  - JSON lines with key, spk, wav
  output_dir/utt2spk    - "key spk_id" per line
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate datalists for training")
    parser.add_argument('root_dir', help='Root dir containing audio/ subdirectory with WAVs')
    parser.add_argument('output_dir', help='Output directory for data.list and utt2spk')
    parser.add_argument('--audio-subdir', default='audio',
                        help='Subdirectory under root_dir containing speaker folders')
    parser.add_argument('--exts', nargs='+', default=['.wav'],
                        help='Audio file extensions to include (e.g. .wav .flac)')
    args = parser.parse_args()

    audio_root = os.path.join(args.root_dir, args.audio_subdir)
    assert os.path.isdir(audio_root), f"Audio dir not found: {audio_root}"

    ext_tuple = tuple(ext.lower() for ext in args.exts)

    os.makedirs(args.output_dir, exist_ok=True)

    data_list_path = os.path.join(args.output_dir, 'data.list')
    utt2spk_path = os.path.join(args.output_dir, 'utt2spk')

    entries = []
    for dirpath, _, filenames in os.walk(audio_root):
        for fn in sorted(filenames):
            if not fn.lower().endswith(ext_tuple):
                continue
            full_path = os.path.join(dirpath, fn)
            rel_path = os.path.relpath(full_path, audio_root)
            # rel_path: spk_id/video_id/00001.wav (or spk_id/utt.flac for flat layouts)
            parts = rel_path.split(os.sep)
            if len(parts) < 2:
                continue
            spk_id = parts[0]
            key = rel_path
            entries.append((key, spk_id, full_path))

    entries.sort(key=lambda x: x[0])
    if not entries:
        raise SystemExit(
            f"No audio files with extensions {args.exts} found under {audio_root}"
        )

    spk_set = set(e[1] for e in entries)
    print(f"Found {len(entries)} utterances from {len(spk_set)} speakers")

    with open(data_list_path, 'w') as f_data, open(utt2spk_path, 'w') as f_utt:
        for key, spk_id, full_path in entries:
            line = json.dumps({"key": key, "spk": spk_id, "wav": full_path})
            f_data.write(line + '\n')
            f_utt.write(f"{key} {spk_id}\n")

    print(f"Wrote {data_list_path} ({len(entries)} lines)")
    print(f"Wrote {utt2spk_path} ({len(entries)} lines)")


if __name__ == '__main__':
    main()
