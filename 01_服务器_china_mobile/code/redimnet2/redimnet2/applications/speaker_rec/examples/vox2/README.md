# VoxCeleb2 training

This directory is a self-contained entry point for reproducing the
VoxCeleb2-only ReDimNet2 training recipes in `applications/speaker_rec`. It follows the
[upstream WeSpeaker VoxCeleb recipe](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb/v2),
but produces the `raw_fast` JSONL consumed by this fork instead of WeSpeaker
tar shards.

The preparation pipeline handles:

- VoxCeleb2 dev download or existing archives, extraction, AAC/M4A to 16 kHz
  mono PCM WAV conversion, and training manifests;
- VoxCeleb1 test download or existing archive, checksum verification,
  extraction into the validation layout, and installation/validation of the
  cleaned `veri_test2.txt` protocol;
- MUSAN and RIRS_NOISES download, checksum verification, extraction, and
  manifests;
- MUSAN and the 60,000 simulated RIRs packed into the LMDB format read by
  `wespeaker_lite`.

The VoxCeleb data is licensed data. Ensure that your use and any download
source comply with its terms. The default VoxCeleb URLs mirror the existing
project data utility and may require `HF_TOKEN`; use `--vox2-archive` and
`--vox1-test-archive` for archives obtained through another authorized route.

## Prerequisites

Run commands from the repository root. Preparation requires Python 3.10+,
`ffmpeg`, and the Python `lmdb` package:

```bash
ffmpeg -version
python -m pip install lmdb
```

The training environment must also contain the dependencies imported by
`wespeaker_lite` (PyTorch, torchaudio, soundfile, scipy, pandas, PyYAML,
python-fire, and tableprint).

Training performs VoxCeleb1-O validation every epoch. The preparation script
therefore creates `/data/wespeaker_data/vox1-test` with `veri_test2.txt` and
`wav/<speaker>/<video>/<utterance>.wav`; this path can be used directly as
`validation.vox1`.

## Prepare the data

The default command downloads and prepares all four datasets. Every stage is
restartable; completed downloads, extraction markers, WAVs, manifests, and
complete LMDBs are reused.

```bash
export HF_TOKEN=...  # if required by the VoxCeleb2 source

python applications/speaker_rec/examples/vox2/scripts/prepare_data.py \
  --data-root /data/wespeaker_data
```

The complete VoxCeleb2 dev set is checked against 1,092,009 utterances and
5,994 speakers (the speaker count is reported when manifests are written).
VoxCeleb1 test is checked against 4,874 WAVs and 37,611 verification trials;
every one of the protocol's 4,708 unique utterance references must exist.
MUSAN is checked against 2,016 files and the selected `simulated_rirs` subset
against 60,000 files. A count mismatch stops preparation; the
`--allow-incomplete` option is intended only for tests or deliberate subsets.

For archives already on disk:

```bash
python applications/speaker_rec/examples/vox2/scripts/prepare_data.py \
  --data-root /data/wespeaker_data \
  --vox2-archive /archives/vox2_aac_1.zip \
  --vox2-archive /archives/vox2_aac_2.zip \
  --vox1-test-archive /archives/vox1_test_wav.zip \
  --vox1-protocol /archives/veri_test2.txt \
  --musan-archive /archives/musan.tar.gz \
  --rirs-archive /archives/rirs_noises.zip \
  --stages extract convert manifests lmdb
```

For an already extracted VoxCeleb2 tree, point at either the AAC/M4A tree or
an existing WAV tree. Existing WAV files are used directly:

```bash
python applications/speaker_rec/examples/vox2/scripts/prepare_data.py \
  --data-root /data/wespeaker_data \
  --vox2-source /datasets/voxceleb2/dev/aac \
  --vox1-test-source /datasets/voxceleb1/test \
  --vox1-protocol /datasets/voxceleb1/test/veri_test2.txt \
  --musan-source /datasets/musan \
  --rirs-source /datasets/RIRS_NOISES \
  --stages extract convert manifests lmdb
```

The stages are `download`, `extract`, `convert`, `manifests`, and `lmdb`.
Select a suffix of the pipeline with `--stages`, as in the examples above.
Use `--workers N` to control parallel `ffmpeg` jobs. If disk capacity is
tight, `--delete-converted-source` removes only successfully converted
AAC/M4A files under `<data-root>/raw`; it refuses to delete an external
`--vox2-source`. `--delete-archives` likewise deletes only archives held in
the generated downloads directory.

An interrupted LMDB build is not assumed valid. Rerun its stage with
`--rebuild-lmdb` to replace the incomplete generated database:

```bash
python applications/speaker_rec/examples/vox2/scripts/prepare_data.py \
  --data-root /data/wespeaker_data \
  --stages lmdb \
  --rebuild-lmdb
```

The relevant output layout is:

```text
/data/wespeaker_data/
├── vox1-test/
│   ├── wav/<speaker>/<video>/<utterance>.wav
│   └── veri_test2.txt
├── vox2_dev/
│   ├── wav/<speaker>/<video>/<utterance>.wav
│   ├── data.list
│   ├── utt2spk
│   ├── spk2utt
│   └── wav.scp
├── musan/
│   ├── wav.scp
│   └── lmdb/
└── rirs/
    ├── wav.scp
    └── lmdb/
```

If `--vox2-source` points at an existing WAV tree, `data.list` contains paths
to that tree and the generated `vox2_dev/wav` directory is not used.

## Configure training with a YAML overlay

`recipes/` contains the b3 and b6 convolution + attention recipe pairs. The
base recipes use generic `/data/wespeaker_data` defaults. Create a small local
overlay for machine-specific paths instead of editing every recipe. For example,
`/tmp/vox2-ptn.local.yaml`:

```yaml
train_data: /data/wespeaker_data/vox2_dev/data.list
train_label: /data/wespeaker_data/vox2_dev/utt2spk
exp_dir: exp/vox2/redimnet2_b3_conv_att/ptn

dataset_args:
  # The two LMDBs are about 14 GiB together. Enable this only when /dev/shm
  # has sufficient free space; otherwise they are read from data-root.
  lmdb_to_ram: false
  aug_setup:
    noise_lmdb_file: /data/wespeaker_data/musan/lmdb
    reverb_lmdb_file: /data/wespeaker_data/rirs/lmdb

validation:
  vox1: /data/wespeaker_data/vox1-test
```

The overlay is recursively merged into the base recipe, so unspecified model,
optimizer, scheduler, and augmentation settings remain unchanged. Top-level
command-line flags are applied after the overlay and therefore take final
precedence.

Available recipe pairs are:

- `redimnet2/b3`: b3 with convolution + attention blocks;
- `redimnet2/b6`: larger b6 with convolution + attention blocks.

The PTN recipes use speed perturbation and 2-second crops for 120 epochs. The
LM recipes disable speed perturbation and use 6-second crops for 5 epochs.
The training code derives `projection_args.num_class` from `utt2spk`, so its
copied value does not need to be changed.

## Run pretraining

The launcher now forwards training overrides to `train.py`:

```bash
./applications/speaker_rec/launch.sh \
  applications/speaker_rec/examples/vox2/recipes/redimnet2/b3/ptn.yaml \
  8 \
  --config_overrides /tmp/vox2-ptn.local.yaml
```

The second positional argument is the number of local GPU processes and
defaults to 8. For a short smoke run, a top-level CLI override can be added:

```bash
./applications/speaker_rec/launch.sh \
  applications/speaker_rec/examples/vox2/recipes/redimnet2/b3/ptn.yaml \
  1 \
  --config_overrides /tmp/vox2-ptn.local.yaml \
  --num_epochs 1 \
  --exp_dir exp/vox2/smoke
```

`launch.sh` automatically resumes from the newest `model_*.pt` in the
configured `exp_dir/models`. Use a new or empty `exp_dir` when a clean run is
required. The fully merged runtime configuration is saved as
`<exp_dir>/config.yaml`.

## Run large-margin fine-tuning

After PTN reaches epoch 120, create `/tmp/vox2-lm.local.yaml` with the same
data, augmentation, and validation overrides, plus stage-specific paths:

```yaml
train_data: /data/wespeaker_data/vox2_dev/data.list
train_label: /data/wespeaker_data/vox2_dev/utt2spk
exp_dir: exp/vox2/redimnet2_b3_conv_att/lm
model_ckpt: exp/vox2/redimnet2_b3_conv_att/ptn/models/model_120.pt

dataset_args:
  aug_setup:
    noise_lmdb_file: /data/wespeaker_data/musan/lmdb
    reverb_lmdb_file: /data/wespeaker_data/rirs/lmdb

validation:
  vox1: /data/wespeaker_data/vox1-test
```

Then launch the matching LM recipe:

```bash
./applications/speaker_rec/launch.sh \
  applications/speaker_rec/examples/vox2/recipes/redimnet2/b3/lm.yaml \
  8 \
  --config_overrides /tmp/vox2-lm.local.yaml
```

The LM projection head is reassembled from the PTN experiment's saved
`spk2id.json`; keep that file next to the PTN `models/` directory.

## Common failures

- `No speaker/video/audio tree`: point `--vox2-source` at the directory above
  the speaker directories, or run extraction first.
- VoxCeleb2 count mismatch: confirm that both dev archives were extracted and
  rerun `convert`; existing valid WAVs are skipped.
- `Incomplete LMDB exists`: rerun `lmdb` with `--rebuild-lmdb`.
- failure while staging LMDB to RAM: set `dataset_args.lmdb_to_ram: false` in
  the overlay or provide enough `/dev/shm` capacity.
- validation assertions: ensure the VoxCeleb1 root contains both
  `veri_test2.txt` and every referenced file below `wav/`.
