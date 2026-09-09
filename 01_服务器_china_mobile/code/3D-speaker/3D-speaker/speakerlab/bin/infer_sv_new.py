# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
Extended version of infer_sv.py with ReDimNet2 support.

This script supports two model sources:
  1. ModelScope models (original behavior, unchanged):
     `python infer_sv_new.py --model_id iic/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k --wavs a.wav b.wav`

  2. ReDimNet2 models from GitHub Releases:
     `python infer_sv_new.py --model_id redimnet2/b6/vb2+vox2+cnc2_v0/lm --wavs a.wav b.wav`

     ReDimNet2 model_id format: redimnet2/<model_name>/<dataset>/<train_type>
       model_name: b0, b1, b2, b3, b4, b5, b6
       dataset:    vox2, vb2+vox2_v0, vb2+vox2+cnc2_v0
       train_type: ptn (pretrain), lm (large-margin, recommended)

Usage:
    1. extract the embedding from the wav file.
        `python infer_sv_new.py --model_id $model_id --wavs $wav_path`
    2. extract embeddings from two wav files and compute the similarity score.
        `python infer_sv_new.py --model_id $model_id --wavs $wav_path1 $wav_path2`
    3. extract embeddings from the wav list.
        `python infer_sv_new.py --model_id $model_id --wavs $wav_list`
"""

import os
import sys
import re
import pathlib
import numpy as np
import argparse
import torch
import torchaudio
from tqdm import tqdm

try:
    from speakerlab.process.processor import FBank
except ImportError:
    sys.path.append('%s/../..' % os.path.dirname(__file__))
    from speakerlab.process.processor import FBank

from speakerlab.utils.builder import dynamic_import

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines.util import is_official_hub_path

from kaldiio import WriteHelper

parser = argparse.ArgumentParser(description='Extract speaker embeddings (supports ModelScope + ReDimNet2).')
parser.add_argument('--model_id', default='', type=str, help='Model id: iic/... (ModelScope) or redimnet2/b6/vb2+vox2+cnc2_v0/lm (GitHub)')
parser.add_argument('--wavs', nargs='+', type=str, help='Wavs')
parser.add_argument('--local_model_dir', default='pretrained', type=str, help='Local model dir')
parser.add_argument('--feat_out_format', choices=['npy', 'ark'], default='npy', type=str,
                    help='Output format: npy (single wav) or ark (wav list, compatible with compute_score_metrics.py)')
parser.add_argument('--redimnet2_path', default='', type=str,
                    help='Path to redimnet2 repo (e.g. /path/to/redimnet2/redimnet2). '
                         'Auto-detected relative to current dir if empty.')

# ============================================================================
# Model configs (same as original infer_sv.py)
# ============================================================================

CAMPPLUS_VOX = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {'feat_dim': 80, 'embedding_size': 512},
}

CAMPPLUS_COMMON = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {'feat_dim': 80, 'embedding_size': 192},
}

ERes2Net_VOX = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {'feat_dim': 80, 'embedding_size': 192},
}

ERes2NetV2_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
    'args': {'feat_dim': 80, 'embedding_size': 192, 'baseWidth': 26, 'scale': 2, 'expansion': 2},
}

ERes2NetV2_w24s4ep4_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
    'args': {'feat_dim': 80, 'embedding_size': 192, 'baseWidth': 24, 'scale': 4, 'expansion': 4},
}

ERes2Net_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2Net_huge.ERes2Net',
    'args': {'feat_dim': 80, 'embedding_size': 192},
}

ERes2Net_base_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {'feat_dim': 80, 'embedding_size': 512, 'm_channels': 32},
}

ERes2Net_Base_3D_Speaker = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {'feat_dim': 80, 'embedding_size': 512, 'm_channels': 32},
}

ERes2Net_Large_3D_Speaker = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {'feat_dim': 80, 'embedding_size': 512, 'm_channels': 64},
}

ECAPA_CNCeleb = {
    'obj': 'speakerlab.models.ecapa_tdnn.ECAPA_TDNN.ECAPA_TDNN',
    'args': {'input_size': 80, 'lin_neurons': 192, 'channels': [1024, 1024, 1024, 1024, 3072]},
}

# ============================================================================
# Model registry
# ============================================================================

supports = {
    # --- ModelScope models (original, unchanged) ---
    'iic/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0',
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_common.bin',
    },
    'iic/speech_eres2net_sv_zh-cn_16k-common': {
        'revision': 'v1.0.5',
        'model': ERes2Net_COMMON,
        'model_pt': 'pretrained_eres2net_aug.ckpt',
    },
    'iic/speech_eres2netv2_sv_zh-cn_16k-common': {
        'revision': 'v1.0.1',
        'model': ERes2NetV2_COMMON,
        'model_pt': 'pretrained_eres2netv2.ckpt',
    },
    'iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common': {
        'revision': 'v1.0.1',
        'model': ERes2NetV2_w24s4ep4_COMMON,
        'model_pt': 'pretrained_eres2netv2w24s4ep4.ckpt',
    },
    'iic/speech_eres2net_base_200k_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0',
        'model': ERes2Net_base_COMMON,
        'model_pt': 'pretrained_eres2net.pt',
    },
    'iic/speech_campplus_sv_zh_en_16k-common_advanced': {
        'revision': 'v1.0.0',
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_en_common.pt',
    },
    'iic/speech_campplus_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2',
        'model': CAMPPLUS_VOX,
        'model_pt': 'campplus_voxceleb.bin',
    },
    'iic/speech_eres2net_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2',
        'model': ERes2Net_VOX,
        'model_pt': 'pretrained_eres2net.ckpt',
    },
    'iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.1',
        'model': ERes2Net_Base_3D_Speaker,
        'model_pt': 'eres2net_base_model.ckpt',
    },
    'iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.0',
        'model': ERes2Net_Large_3D_Speaker,
        'model_pt': 'eres2net_large_model.ckpt',
    },
    'iic/speech_ecapa-tdnn_sv_zh-cn_cnceleb_16k': {
        'revision': 'v1.0.0',
        'model': ECAPA_CNCeleb,
        'model_pt': 'ecapa-tdnn.ckpt',
    },
    'iic/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.0',
        'model': ECAPA_CNCeleb,
        'model_pt': 'ecapa-tdnn.ckpt',
    },
    'iic/speech_ecapa-tdnn_sv_en_voxceleb_16k': {
        'revision': 'v1.0.1',
        'model': ECAPA_CNCeleb,
        'model_pt': 'ecapa_tdnn.bin',
    },
}

# ReDimNet2 GitHub release URL template
REDIMNET2_URL = "https://github.com/PalabraAI/redimnet2/releases/download/v1.0.0/{weight_name}"


# ============================================================================
# Helper: try to locate and import redimnet2
# ============================================================================

def _find_redimnet2_root(user_path=''):
    """Resolve the redimnet2 repository root. Returns str or None."""
    if user_path and os.path.isdir(user_path):
        return os.path.abspath(user_path)

    # Auto-search: go up from <repo>/speakerlab/bin/ to find code/redimnet2/redimnet2
    # Path layout: .../code/3D-speaker/3D-speaker/speakerlab/bin/infer_sv_new.py
    # Target:      .../code/redimnet2/redimnet2/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # go up 4 levels: bin → speakerlab → 3D-speaker(inner) → 3D-speaker(outer) → code → china_mobile
    project_parent = script_dir
    for _ in range(4):
        project_parent = os.path.dirname(project_parent)
    candidates = [
        os.path.join(project_parent, 'redimnet2', 'redimnet2'),
        os.path.join(os.getcwd(), 'redimnet2', 'redimnet2'),
        os.path.join(os.getcwd(), '..', 'redimnet2', 'redimnet2'),
        os.path.join(project_parent, '..', 'redimnet2', 'redimnet2'),
    ]
    for cand in candidates:
        cand = os.path.normpath(cand)
        if os.path.isdir(os.path.join(cand, 'redimnet2')):
            return cand
    return None


def _ensure_redimnet2_importable(path):
    """Add redimnet2 to sys.path so 'from redimnet2 import load_custom' works."""
    if path and path not in sys.path:
        sys.path.insert(0, path)


# ============================================================================
# ReDimNet2 model loading
# ============================================================================

def _load_redimnet2_from_github(model_name, train_type, dataset):
    """Load ReDimNet2 weights (local file first, fallback to GitHub)."""
    from redimnet2.redimnet2 import ReDimNet2Wrap

    weight_name = f"{model_name}-{dataset}-{train_type}.pt"
    local_pt = os.path.join(os.getcwd(), 'pretrained', 'redimnet2', weight_name)
    if os.path.exists(local_pt):
        print(f'[INFO]: Loading ReDimNet2 from local {local_pt} ...')
        full_state_dict = torch.load(local_pt, map_location='cpu')
        model = ReDimNet2Wrap(**full_state_dict["model_config"])
        model.load_state_dict(full_state_dict["state_dict"])
        model.eval()
        return model, True
    from redimnet2 import load_custom
    print(f'[INFO]: Loading ReDimNet2-{model_name} ({dataset}, {train_type}) from GitHub...')
    model = load_custom(model_name=model_name, train_type=train_type, dataset=dataset)
    model.eval()
    return model, True


# ============================================================================
# Wav loading
# ============================================================================

def load_wav(wav_file, obj_fs=16000):
    wav, fs = torchaudio.load(wav_file)
    if fs != obj_fs:
        print(f'[WARNING]: The sample rate of {wav_file} is not {obj_fs}, resample it.')
        wav, fs = torchaudio.sox_effects.apply_effects_tensor(
            wav, fs, effects=[['rate', str(obj_fs)]]
        )
    if wav.shape[0] > 1:
        wav = wav[0, :].unsqueeze(0)
    return wav


# ============================================================================
# Main
# ============================================================================

def main():
    args = parser.parse_args()

    # ---------- Detect model source ----------
    if args.model_id.startswith('redimnet2/'):
        _run_redimnet2(args)
    else:
        _run_modelscope(args)


def _run_modelscope(args):
    """Original ModelScope inference path — identical to infer_sv.py."""
    assert isinstance(args.model_id, str) and \
        True, "Invalid modelscope model id."
    if args.model_id.startswith('damo/'):
        args.model_id = args.model_id.replace('damo/', 'iic/', 1)
    assert args.model_id in supports, f"Model id '{args.model_id}' not currently supported."

    save_dir = os.path.join(args.local_model_dir, args.model_id.split('/')[1])
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    conf = supports[args.model_id]
    # download models from modelscope
    cache_dir = snapshot_download(args.model_id, revision=conf['revision'])
    cache_dir = pathlib.Path(cache_dir)

    embedding_dir = save_dir / 'embeddings'
    embedding_dir.mkdir(exist_ok=True, parents=True)

    # link
    download_files = ['examples', conf['model_pt']]
    for src in cache_dir.glob('*'):
        if re.search('|'.join(download_files), src.name):
            dst = save_dir / src.name
            try:
                dst.unlink()
            except FileNotFoundError:
                pass
            dst.symlink_to(src)

    pretrained_model = save_dir / conf['model_pt']
    pretrained_state = torch.load(pretrained_model, map_location='cpu')

    if torch.cuda.is_available():
        msg = 'Using gpu for inference.'
        print(f'[INFO]: {msg}')
        device = torch.device('cuda')
    else:
        msg = 'No cuda device is detected. Using cpu.'
        print(f'[INFO]: {msg}')
        device = torch.device('cpu')

    # load model
    model_conf = conf['model']
    embedding_model = dynamic_import(model_conf['obj'])(**model_conf['args'])
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.to(device)
    embedding_model.eval()

    feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

    def compute_embedding(wav_file, save=True):
        wav = load_wav(wav_file)
        feat = feature_extractor(wav).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = embedding_model(feat).detach().squeeze(0).cpu().numpy()
        if save:
            save_path = embedding_dir / (
                '%s.npy' % (os.path.basename(wav_file).rsplit('.', 1)[0]))
            np.save(save_path, embedding)
            print(f'[INFO]: The extracted embedding from {wav_file} is saved to {save_path}.')
        return embedding

    # extract embeddings
    print(f'[INFO]: Extracting embeddings...')
    _dispatch_wavs(args, embedding_dir, compute_embedding, save_dir=save_dir)


def _run_redimnet2(args):
    """ReDimNet2 inference path — loads from GitHub, raw waveform input."""
    # Parse model_id: redimnet2/<model_name>/<dataset>/<train_type>
    parts = args.model_id.split('/')
    if len(parts) != 4:
        raise ValueError(
            f"ReDimNet2 model_id must be: redimnet2/<model_name>/<dataset>/<train_type>\n"
            f"  Example: redimnet2/b6/vb2+vox2+cnc2_v0/lm\n"
            f"  Got: {args.model_id}"
        )
    _, model_name, dataset, train_type = parts

    # Locate redimnet2 code
    redimnet2_root = _find_redimnet2_root(args.redimnet2_path)
    if redimnet2_root is None:
        print("[ERROR]: Cannot find redimnet2 source code. "
              "Please pass --redimnet2_path /path/to/redimnet2/redimnet2")
        sys.exit(1)
    _ensure_redimnet2_importable(redimnet2_root)

    # Save dir
    save_name = f"redimnet2_{model_name}_{dataset}_{train_type}".replace('/', '_')
    save_dir = os.path.join(args.local_model_dir, save_name)
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    embedding_dir = save_dir / 'embeddings'
    embedding_dir.mkdir(exist_ok=True, parents=True)

    # Device
    if torch.cuda.is_available():
        print('[INFO]: Using gpu for inference.')
        device = torch.device('cuda')
    else:
        print('[INFO]: No cuda device is detected. Using cpu.')
        device = torch.device('cpu')

    # Load model
    embedding_model, is_raw = _load_redimnet2_from_github(model_name, train_type, dataset)
    embedding_model.to(device)
    embedding_model.eval()

    def compute_embedding(wav_file, save=True):
        wav = load_wav(wav_file)
        wav = wav.to(device)
        with torch.no_grad():
            embedding = embedding_model(wav).detach().squeeze(0).cpu().numpy()
        if save:
            save_path = embedding_dir / (
                '%s.npy' % (os.path.basename(wav_file).rsplit('.', 1)[0]))
            np.save(save_path, embedding)
            print(f'[INFO]: The extracted embedding from {wav_file} is saved to {save_path}.')
        return embedding

    print(f'[INFO]: Extracting embeddings...')

    # If given a wav list and ark format, do batch extraction (like infer_sv_batch.py)
    if len(args.wavs) == 1 and not args.wavs[0].endswith('.wav') and args.feat_out_format == 'ark':
        wav_list_file = args.wavs[0]
        with open(wav_list_file, 'r') as f:
            wav_list = [line.strip() for line in f.readlines()]
        assert len(wav_list) > 0, 'Empty wav list.'

        emb_ark = os.path.join(str(embedding_dir), 'xvector_0.ark')
        emb_scp = os.path.join(str(embedding_dir), 'xvector_0.scp')
        if os.path.exists(emb_ark):
            print(f'[INFO]: Removing existing {emb_ark}')
            os.remove(emb_ark)
        if os.path.exists(emb_scp):
            os.remove(emb_scp)

        with WriteHelper(f'ark,scp:{emb_ark},{emb_scp}') as writer:
            for wav_path in tqdm(wav_list, desc='Extracting'):
                wav_path = wav_path.strip()
                key = os.path.splitext(os.path.basename(wav_path))[0]
                wav = load_wav(wav_path)
                wav = wav.to(device)
                with torch.no_grad():
                    emb = embedding_model(wav).detach().squeeze(0).cpu().numpy()
                writer(key, emb)
        print(f'[INFO]: Embeddings saved to {emb_ark} / {emb_scp}')
    else:
        _dispatch_wavs(args, embedding_dir, compute_embedding, save_dir=save_dir)


def _dispatch_wavs(args, embedding_dir, compute_embedding, save_dir=None):
    """Shared logic: route 0/1/2 wavs or a wav list to the right action."""
    if args.wavs is None or len(args.wavs) == 2:
        if args.wavs is None:
            # use example wavs
            try:
                examples_dir = save_dir / 'examples'
                wav_path1, wav_path2 = list(examples_dir.glob('*.wav'))[0:2]
                print(f'[INFO]: No wavs input, use example wavs instead.')
            except Exception:
                raise Exception('[ERROR]: Invalid input wav.')
        else:
            wav_path1, wav_path2 = args.wavs

        embedding1 = compute_embedding(wav_path1)
        embedding2 = compute_embedding(wav_path2)

        print('[INFO]: Computing the similarity score...')
        similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        scores = similarity(
            torch.from_numpy(embedding1).unsqueeze(0),
            torch.from_numpy(embedding2).unsqueeze(0)
        ).item()
        print('[INFO]: The similarity score between two input wavs is %.4f' % scores)

    elif len(args.wavs) == 1:
        if args.wavs[0].endswith('.wav'):
            wav_path = args.wavs[0]
            embedding = compute_embedding(wav_path)
        else:
            try:
                wav_list_file = args.wavs[0]
                with open(wav_list_file, 'r') as f:
                    wav_list = f.readlines()
            except Exception:
                raise Exception('[ERROR]: Input should be wav file or wav list.')
            for wav_path in wav_list:
                wav_path = wav_path.strip()
                embedding = compute_embedding(wav_path)
    else:
        raise Exception('[ERROR]: Supports up to two input files')


if __name__ == '__main__':
    main()
