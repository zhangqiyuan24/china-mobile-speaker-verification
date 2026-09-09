# Speaker Recognition Training

This application contains the lightweight [wespeaker](https://github.com/wenet-e2e/wespeaker)-forked training pipeline used for
ReDimNet2 speaker-recognition experiments.

## Contents

- `wespeaker_lite/`: training code, datasets, augmentations, schedulers, and
  checkpoint utilities.
- `examples/vox2/`: VoxCeleb2-only data preparation and b3/b6 recipes.
- `examples/vb2_vox2_cnc2/`: VoxCeleb2 + VoxBlink2 + CN-Celeb2 manifest
  preparation and b3/b6 recipes.
- `launch.md`: torchrun launch details and environment setup.
- `pyproject.toml`: Poetry environment with the Python dependencies needed by
  `wespeaker_lite` and the included data-preparation scripts.

Run commands from the repository root so both `redimnet2` and
`applications/speaker_rec/wespeaker_lite` are importable.

## Environment

The training code is GPU-oriented and uses `torchrun` with NCCL. Install a
PyTorch/torchaudio build that matches your CUDA driver. The Poetry file keeps
the dependency list lightweight and does not pin a CUDA wheel source.

```bash
cd applications/speaker_rec
poetry install --no-root
cd ../..
```

If your cluster requires a specific PyTorch wheel index, install matching
`torch` and `torchaudio` in the Poetry environment first, then run
`poetry install --no-root`:

```bash
cd applications/speaker_rec
poetry run python -m pip install torch torchaudio --index-url <pytorch-wheel-index>
poetry install --no-root
```

System tools are still required for data preparation and optional codec
augmentation:

- `ffmpeg` for VoxCeleb audio conversion and codec augmentation.
- `libsndfile` for the Python `soundfile` package.
- Enough `/dev/shm` capacity if `dataset_args.lmdb_to_ram: true` is enabled.

## Typical Workflow

Prepare data with one of the example READMEs, create a small local YAML overlay
for machine-specific paths, then launch from the repository root:

```bash
cd applications/speaker_rec
poetry run bash -lc 'cd ../.. && ./applications/speaker_rec/launch.sh \
  applications/speaker_rec/examples/vox2/recipes/redimnet2/b3/ptn.yaml \
  8 \
  --config_overrides /tmp/vox2-ptn.local.yaml'
```

For the combined VoxCeleb2 + VoxBlink2 + CN-Celeb2 recipes, start with
`examples/vb2_vox2_cnc2/README.md`. That example uses `pyarrow` and
`soundfile` for duration manifests in addition to the core training
dependencies.
