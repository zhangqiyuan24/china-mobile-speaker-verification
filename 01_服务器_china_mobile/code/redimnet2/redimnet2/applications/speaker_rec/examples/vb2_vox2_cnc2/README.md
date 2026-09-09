# VoxCeleb2 + VoxBlink2 + CN-Celeb2 training

This example prepares the manifest mix used by the VoxCeleb2 + VoxBlink2 +
CN-Celeb2 ReDimNet2 recipes in `applications/speaker_rec`. It is intentionally manifest-only:
VoxCeleb2, VoxCeleb1-test, MUSAN, RIRS, VoxBlink2, and CN-Celeb2 acquisition or
audio conversion must happen before these scripts are run.

The data mix uses three stable namespaces:

- `A_`: VoxCeleb2
- `B_`: VoxBlink2
- `C_`: CN-Celeb2

The PTN and LM manifests use the same namespace order so LM projection-head
reassembly can map speakers from the PTN `spk2id.json`.

## Prerequisites

Run commands from the repository root. The manifest scripts need Python 3.10+
and the Python packages used for JSONL/parquet audio metadata:

```bash
python -m pip install pandas pyarrow soundfile
```

The VoxCeleb2 helper also needs `ffmpeg` and `lmdb`, as documented in
`applications/speaker_rec/examples/vox2/README.md`.

## Acquire the datasets

### VoxCeleb2, VoxCeleb1-test, MUSAN, and RIRS

Use the existing VoxCeleb2 example to download or prepare VoxCeleb2 dev,
VoxCeleb1-test, raw MUSAN, and RIRS:

```bash
python applications/speaker_rec/examples/vox2/scripts/prepare_data.py \
  --data-root /data/wespeaker_data
```

For this example the important outputs are:

```text
/data/wespeaker_data/
|-- vox2_dev/
|   |-- data.list
|   `-- utt2spk
|-- vox1-test/
|   |-- veri_test2.txt
|   `-- wav/
|-- raw/
|   `-- musan/
|       |-- music/
|       |-- noise/
|       `-- speech/
`-- rirs/
    `-- lmdb/
```

The combined recipes use raw MUSAN category directories, not the MUSAN LMDB.
RIRS is used through the LMDB produced by the VoxCeleb2 helper.

### CN-Celeb2

Download CN-Celeb2 from OpenSLR SLR82:

https://www.openslr.org/82/

The OpenSLR page publishes CN-Celeb2 as three split files:
`cn-celeb2_v2.tar.gzaa`, `cn-celeb2_v2.tar.gzab`, and
`cn-celeb2_v2.tar.gzac`. After download, extract them into one root:

```bash
mkdir -p /data/cnceleb2
cat /archives/cn-celeb2_v2.tar.gz{aa,ab,ac} | tar -xzf - -C /data/cnceleb2
```

The expected training layout is:

```text
/data/cnceleb2/CN-Celeb2_flac/data/<speaker>/<utterance>.flac
```

### VoxBlink2

Request access through the VoxBlink2 issue comment:

https://github.com/VoxBlink2/ScriptsForVoxBlink2/issues/10#issuecomment-2275188994

Prepare or convert the delivered corpus so the audio root contains WAV files
below speaker directories:

```text
/data/voxblink2/audio/<speaker>/.../*.wav
```

This example does not include VoxBlink2 download, extraction, or conversion
scripts.

## Build PTN manifests

Generate per-source manifests and the joined PTN manifest:

```bash
applications/speaker_rec/examples/vb2_vox2_cnc2/scripts/prepare_vox2_vb2_cnc2_data.sh \
  --vox2-list /data/wespeaker_data/vox2_dev \
  --vb2-root /data/voxblink2 \
  --cnc2-root /data/cnceleb2 \
  --output-root /data/wespeaker_data/vox2_vb2_cnc2_v0
```

Outputs:

```text
/data/wespeaker_data/vox2_vb2_cnc2_v0/
|-- cnc2_full/
|   |-- data.list
|   `-- utt2spk
|-- vb2_full/
|   |-- data.list
|   `-- utt2spk
`-- mix_join/
    |-- data.list
    `-- utt2spk
```

Use `mix_join/data.list` and `mix_join/utt2spk` for PTN training.

## Build LM manifests

The LM stage filters short utterances from the same three source manifests and
joins the survivors with the same `A_`, `B_`, `C_` prefixes:

```bash
applications/speaker_rec/examples/vb2_vox2_cnc2/scripts/prepare_vox2_vb2_cnc2_lm_data.sh \
  --vox2-list /data/wespeaker_data/vox2_dev \
  --vb2-list /data/wespeaker_data/vox2_vb2_cnc2_v0/vb2_full \
  --cnc2-list /data/wespeaker_data/vox2_vb2_cnc2_v0/cnc2_full \
  --output-root /data/wespeaker_data/vox2_vb2_cnc2_v0 \
  --min-duration 5.0 \
  --duration-workers 32
```

If your VoxBlink2 package includes a duration parquet, pass it explicitly:

```bash
  --vb2-durations /data/voxblink2/durations.parquet
```

Otherwise durations are inferred into each manifest directory as
`durations.parquet`.

Outputs:

```text
/data/wespeaker_data/vox2_vb2_cnc2_v0/
|-- vox2_lm/
|-- vb2_lm/
|-- cnc2_lm/
`-- mix_join_lm/
    |-- data.list
    `-- utt2spk
```

Use `mix_join_lm/data.list` and `mix_join_lm/utt2spk` for LM training.

## Configure training

The copied recipes keep the source experiment schedule and model defaults, with
generic `/data/...` paths. Use a local overlay instead of editing the recipes in
place.

For PTN, create `/tmp/vb2_vox2_cnc2-ptn.local.yaml`:

```yaml
train_data: /data/wespeaker_data/vox2_vb2_cnc2_v0/mix_join/data.list
train_label: /data/wespeaker_data/vox2_vb2_cnc2_v0/mix_join/utt2spk
exp_dir: exp/vb2_vox2_cnc2/redimnet2_b3/ptn

dataset_args:
  aug_setup:
    aug_type: Sequential
    prob: 0.9
    augmentors:
    - aug_type: OneOf
      prob: 0.8
      augmentors:
      - aug_type: CustomNoise
        dataset_dir: /data/wespeaker_data/raw/musan/music
        snr: [2, 12]
        prob: 1.0
      - aug_type: CustomNoise
        dataset_dir: /data/wespeaker_data/raw/musan/noise
        snr: [0, 12]
        prob: 1.0
      - aug_type: CustomNoise
        dataset_dir: /data/wespeaker_data/raw/musan/speech
        snr: [7, 18]
        babble: true
        prob: 1.0
    - aug_type: Reverb
      lmdb_file: /data/wespeaker_data/rirs/lmdb
      prob: 0.2
    - aug_type: OneOf
      prob: 0.25
      augmentors:
      - aug_type: LowPassAug
        cutoff_ratio: 0.6
        transition_width: 0.08
        prob: 1.0

validation:
  vox1: /data/wespeaker_data/vox1-test
```

The overlay merge replaces YAML lists, so include the full `aug_setup` tree when
changing the MUSAN or RIRS paths.

Run PTN:

```bash
./applications/speaker_rec/launch.sh \
  applications/speaker_rec/examples/vb2_vox2_cnc2/recipes/redimnet2/b3/ptn.yaml \
  8 \
  --config_overrides /tmp/vb2_vox2_cnc2-ptn.local.yaml
```

For b6, use the b6 recipe and set a matching `exp_dir`:

```bash
./applications/speaker_rec/launch.sh \
  applications/speaker_rec/examples/vb2_vox2_cnc2/recipes/redimnet2/b6/ptn.yaml \
  8 \
  --config_overrides /tmp/vb2_vox2_cnc2-ptn.local.yaml \
  --exp_dir exp/vb2_vox2_cnc2/redimnet2_b6/ptn
```

## Run large-margin fine-tuning

After PTN finishes, create `/tmp/vb2_vox2_cnc2-lm.local.yaml`. The example
below is for b3:

```yaml
train_data: /data/wespeaker_data/vox2_vb2_cnc2_v0/mix_join_lm/data.list
train_label: /data/wespeaker_data/vox2_vb2_cnc2_v0/mix_join_lm/utt2spk
exp_dir: exp/vb2_vox2_cnc2/redimnet2_b3/lm
model_ckpt: exp/vb2_vox2_cnc2/redimnet2_b3/ptn/models/model_110.pt
spk2id_map_path: exp/vb2_vox2_cnc2/redimnet2_b3/ptn/spk2id.json

dataset_args:
  aug_setup:
    aug_type: Sequential
    prob: 0.7
    augmentors:
    - aug_type: OneOf
      prob: 0.8
      augmentors:
      - aug_type: CustomNoise
        dataset_dir: /data/wespeaker_data/raw/musan/music
        snr: [2, 12]
        prob: 1.0
      - aug_type: CustomNoise
        dataset_dir: /data/wespeaker_data/raw/musan/noise
        snr: [0, 12]
        prob: 1.0
      - aug_type: CustomNoise
        dataset_dir: /data/wespeaker_data/raw/musan/speech
        snr: [7, 18]
        babble: true
        prob: 1.0
    - aug_type: Reverb
      lmdb_file: /data/wespeaker_data/rirs/lmdb
      prob: 0.2
    - aug_type: OneOf
      prob: 0.25
      augmentors:
      - aug_type: LowPassAug
        cutoff_ratio: 0.6
        transition_width: 0.08
        prob: 1.0

validation:
  vox1: /data/wespeaker_data/vox1-test
```

Launch LM:

```bash
./applications/speaker_rec/launch.sh \
  applications/speaker_rec/examples/vb2_vox2_cnc2/recipes/redimnet2/b3/lm.yaml \
  8 \
  --config_overrides /tmp/vb2_vox2_cnc2-lm.local.yaml
```

For b6, use `applications/speaker_rec/examples/vb2_vox2_cnc2/recipes/redimnet2/b6/lm.yaml` and point
`model_ckpt` plus `spk2id_map_path` at the b6 PTN experiment. The b6 source
recipe defaults to epoch 115; set the checkpoint path in the overlay to the
checkpoint you actually want to use.

The training code derives `projection_args.num_class` from `utt2spk`, so do not
set it manually in overlays.

## Common failures

- `VC2 datalist missing`: run `applications/speaker_rec/examples/vox2/scripts/prepare_data.py` first or
  point `--vox2-list` at a directory with `data.list` and `utt2spk`.
- `VB2 audio directory not found`: ensure VoxBlink2 is arranged as
  `<vb2-root>/audio/<speaker>/.../*.wav`.
- `CN-Celeb2 data directory not found`: ensure CN-Celeb2 is extracted as
  `<cnc2-root>/CN-Celeb2_flac/data/<speaker>/<utterance>.flac`.
- Missing parquet support: install `pyarrow`.
- Missing duration rows during LM filtering: regenerate the durations parquet
  or remove the stale parquet and rerun the LM manifest script.
