# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
Teacher-Student Consistency Training for Robust Speaker Verification.

  clean wav --> FBank --> feat_clean --> [Teacher: frozen] --> e_teacher
                                                                │
  same wav --> RobustSpkAug --> FBank --> feat_aug --> [Student: partial] --> e_student --> classifier --> L_arcface
                                                                    │
                                                    L_cons = 1 - cos(e_student, e_teacher.detach())
                                                                    │
                                                    L = L_arcface + lambda_cons * L_cons

Usage: Replace speakerlab/bin/train.py
"""

import os
import sys
import argparse
import time
import csv
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import numpy as np

from speakerlab.utils.utils import set_seed, get_logger, AverageMeters, ProgressMeter, accuracy
from speakerlab.utils.config import build_config
from speakerlab.utils.builder import build
from speakerlab.utils.epoch import EpochCounter, EpochLogger


parser = argparse.ArgumentParser(description='Teacher-Student Speaker Network Training')
parser.add_argument('--config', default='', type=str, help='Config file for training')
parser.add_argument('--resume', default=True, type=bool, help='Resume from recent checkpoint or not')
parser.add_argument('--pretrained', default='', type=str, help='Path to pretrained embedding model checkpoint')
parser.add_argument('--valid_data', default='', type=str, help='Path to validation CSV')
parser.add_argument('--valid_noise', default='', type=str, help='Path to validation noise scp (separate from train)')
parser.add_argument('--valid_spk2utt', default='', type=str, help='Path to valid spk2utt json (for overlap interferer)')
parser.add_argument('--seed', default=1234, type=int, help='Random seed for training.')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')


# ============================================================
# StudentModel: returns (embedding, classifier_output) for DDP
# ============================================================
class StudentModel(nn.Module):
    """
    Thin wrapper that returns both embedding and classifier output.
    Necessary for DDP — both outputs must go through DDP forward
    so that gradient sync hooks fire correctly.
    """
    def __init__(self, embedding_model, classifier):
        super().__init__()
        self.embedding_model = embedding_model
        self.classifier = classifier

    def forward(self, x):
        emb = self.embedding_model(x)
        out = self.classifier(emb)
        return emb, out


# ============================================================
# Freeze / BN utilities (unchanged from original train.py)
# ============================================================
def _apply_freeze(embedding_model, frozen_patterns, logger):
    """Freeze modules matching patterns. Frozen BN -> eval mode."""
    frozen_names = []
    trainable_names = []
    frozen_bn_states = {}

    for name, module in embedding_model.named_modules():
        should_freeze = any(p in name for p in frozen_patterns)
        if should_freeze:
            for param in module.parameters(recurse=False):
                param.requires_grad = False
            frozen_names.append(name)
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()
                frozen_bn_states[name] = (
                    module.running_mean.clone().detach().cpu(),
                    module.running_var.clone().detach().cpu(),
                )
        else:
            if len(list(module.parameters(recurse=False))) > 0:
                trainable_names.append(name)

    total = sum(p.numel() for p in embedding_model.parameters())
    trainable = sum(p.numel() for p in embedding_model.parameters() if p.requires_grad)

    logger.info(f"[FREEZE] Total params: {total:,}  "
                f"Trainable: {trainable:,}  ({100*trainable/total:.1f}%)")
    logger.info(f"[FREEZE] Frozen BN count: {len(frozen_bn_states)}")
    return frozen_names, trainable_names, frozen_bn_states


def _refreeze_bn(embedding_model, frozen_bn_states):
    """Re-set frozen BN layers to eval mode after model.train()."""
    for name, module in embedding_model.named_modules():
        if name in frozen_bn_states:
            module.eval()


def _check_bn_frozen(embedding_model, frozen_bn_states, logger):
    """Verify frozen BN stats haven't changed."""
    changed = []
    for name, module in embedding_model.named_modules():
        if name in frozen_bn_states:
            old_mean, old_var = frozen_bn_states[name]
            new_mean = module.running_mean.detach().cpu()
            new_var = module.running_var.detach().cpu()
            if not torch.allclose(old_mean, new_mean, atol=1e-6):
                changed.append(f"{name}.running_mean")
            if not torch.allclose(old_var, new_var, atol=1e-6):
                changed.append(f"{name}.running_var")
    if changed:
        logger.warning(f"[BN-CHECK] {len(changed)} frozen BN stats changed: {changed[:5]}...")
    else:
        logger.info(f"[BN-CHECK] All {len(frozen_bn_states)} frozen BN stats unchanged.")


def _build_optimizer_with_layerwise_lr(embedding_model, classifier, config, logger):
    """Build optimizer with per-module absolute learning rates."""
    if hasattr(config, 'layerwise_lr') and config.layerwise_lr:
        param_groups = []
        lr_config = config.layerwise_lr
        for name, module in list(embedding_model.named_modules()) + [('classifier', classifier)]:
            module_params = [p for p in module.parameters(recurse=False) if p.requires_grad]
            if not module_params:
                continue
            lr = config.lr
            for pattern, abs_lr in lr_config.items():
                if pattern in name:
                    lr = abs_lr
                    break
            param_groups.append({'params': module_params, 'lr': lr, 'name': name})

        # Catch any remaining
        grouped_ids = set()
        for pg in param_groups:
            for p in pg['params']:
                grouped_ids.add(id(p))
        remaining = [p for p in list(embedding_model.parameters()) + list(classifier.parameters())
                     if p.requires_grad and id(p) not in grouped_ids]
        if remaining:
            param_groups.append({'params': remaining, 'lr': config.lr, 'name': 'rest'})

        if logger and dist.get_rank() == 0:
            for pg in param_groups:
                logger.info(f"  [LR] {pg['name']}: lr={pg['lr']:.2e}, "
                            f"params={sum(p.numel() for p in pg['params'])}")
        config.optimizer['args']['params'] = param_groups
    else:
        all_params = list(embedding_model.parameters()) + list(classifier.parameters())
        all_params = [p for p in all_params if p.requires_grad]
        config.optimizer['args']['params'] = all_params
    return build('optimizer', config)


# ============================================================
# Standard EER computation
# ============================================================
def _compute_eer(target_scores, nontarget_scores):
    """
    Standard EER: find threshold where |FAR - FRR| is minimized.
    FAR(t) = mean(nontarget_scores >= t)
    FRR(t) = mean(target_scores < t)
    """
    import numpy as np
    all_scores = np.concatenate([target_scores, nontarget_scores])
    thresholds = np.sort(np.unique(all_scores))

    min_diff = 1.0
    eer = 1.0
    for t in thresholds:
        far_t = np.mean(nontarget_scores >= t)
        frr_t = np.mean(target_scores < t)
        diff = abs(far_t - frr_t)
        if diff < min_diff:
            min_diff = diff
            eer = (far_t + frr_t) / 2
    return eer * 100


# ============================================================
# Deterministic noise / overlap mixing (no external deps)
# ============================================================
def _mix_noise(clean_wav, noise_wav, snr_db, sample_rate=16000):
    """Mix noise at specified SNR. Both wavs are 1D torch tensors."""
    clean_rms = torch.sqrt(torch.mean(clean_wav ** 2) + 1e-12)
    noise_rms = torch.sqrt(torch.mean(noise_wav ** 2) + 1e-12)
    desired_rms = clean_rms / (10 ** (snr_db / 20.0))
    noise_wav = noise_wav * (desired_rms / (noise_rms + 1e-12))
    # Match lengths
    if noise_wav.shape[0] < clean_wav.shape[0]:
        n_rep = clean_wav.shape[0] // noise_wav.shape[0] + 1
        noise_wav = noise_wav.repeat(n_rep)[:clean_wav.shape[0]]
    else:
        noise_wav = noise_wav[:clean_wav.shape[0]]
    return clean_wav + noise_wav


def _mix_overlap(target_wav, int_wav, tir_db, sample_rate=16000):
    """Mix two speech signals at specified TIR. Both 1D torch tensors."""
    target_rms = torch.sqrt(torch.mean(target_wav ** 2) + 1e-12)
    int_rms = torch.sqrt(torch.mean(int_wav ** 2) + 1e-12)
    desired_int_rms = target_rms / (10 ** (tir_db / 20.0))
    int_wav = int_wav * (desired_int_rms / (int_rms + 1e-12))
    if int_wav.shape[0] < target_wav.shape[0]:
        n_rep = target_wav.shape[0] // int_wav.shape[0] + 1
        int_wav = int_wav.repeat(n_rep)[:target_wav.shape[0]]
    else:
        int_wav = int_wav[:target_wav.shape[0]]
    return target_wav + int_wav


def _chunk_wav(wav, duration, sample_rate, rng):
    """Deterministic chunk/pad to fixed duration.
    - wav > duration: random crop with fixed-seed rng
    - wav < duration: repeat (tile) then trim
    Returns 1D tensor of length int(duration * sample_rate).
    """
    chunk_len = int(duration * sample_rate)
    if wav.shape[0] > chunk_len:
        start = rng.randint(0, wav.shape[0] - chunk_len)
        return wav[start:start + chunk_len]
    else:
        n_repeat = chunk_len // wav.shape[0] + 1
        return wav.repeat(n_repeat)[:chunk_len]


# ============================================================
# Multi-condition deterministic validation
# ============================================================
def _validate(embedding_model, valid_csv, device, logger,
              condition='clean', valid_noise_wavs=None,
              valid_spk2utt=None, num_spk=400, utts_per_spk=4,
              valid_duration=3.0, seed=42):
    """
    Deterministic multi-condition validation with fixed 3s duration.

    Args:
        condition: 'clean' | 'noise_m5db' | 'overlap2_0db'
        valid_noise_wavs: list of noise wav paths (for noise condition)
        valid_spk2utt: {spk: [wav_paths]} (for overlap interferer selection)
        num_spk: max speakers to use
        utts_per_spk: max utterances per speaker
        valid_duration: fixed audio duration in seconds (default 3.0)
        seed: fixed random seed for deterministic selection & chunking
    """
    import csv, random as _random
    from collections import defaultdict
    from speakerlab.process.processor import FBank
    import torchaudio
    import numpy as np

    # ---- Save global random states ----
    py_rng = _random.getstate()
    np_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_rng = torch.cuda.get_rng_state(device)

    try:
        # ---- Fixed seed for deterministic validation ----
        _random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ---- Debug log ----
        logger.info(
            f"[VALID:{condition}] START  |  duration={valid_duration}s  "
            f"speakers={num_spk}  utts_per_spk={utts_per_spk}  seed={seed}")

        # ---- Read validation data ----
        spk2utts = defaultdict(list)
        with open(valid_csv) as f:
            for row in csv.DictReader(f):
                spk2utts[row['spk']].append(row['path'])

        all_spks = sorted(spk2utts.keys())
        if len(all_spks) < 2:
            logger.warning(f"[VALID:{condition}] Not enough speakers ({len(all_spks)})")
            return None

        # ---- RNGs: separate for selection vs chunking to avoid coupling ----
        sel_rng = _random.Random(seed + 1)    # speaker/utt/noise/interferer selection
        chunk_rng = _random.Random(seed + 10000)  # 3s crop positions

        # ---- Select speakers deterministically ----
        n_spk = min(num_spk, len(all_spks))
        selected_spks = sel_rng.sample(all_spks, n_spk)

        # ---- Extract embeddings ----
        sample_rate = 16000
        fbank = FBank(80, sample_rate=sample_rate, mean_nor=True)
        spk2embs = {}

        for spk in selected_spks:
            utts = spk2utts[spk]
            n_utt = min(utts_per_spk, len(utts))
            chosen_utts = sel_rng.sample(utts, n_utt)  # same utts for all 3 conditions
            embs = []

            for wav_path in chosen_utts:
                try:
                    # ---- Load target wav ----
                    wav, sr = torchaudio.load(wav_path)
                    if sr != sample_rate:
                        wav = torchaudio.functional.resample(wav, sr, sample_rate)
                    wav = wav[0]  # mono

                    # ---- Deterministic 3s chunk ----
                    wav = _chunk_wav(wav, valid_duration, sample_rate, chunk_rng)

                    # ---- Apply condition (on already-chunked 3s wav) ----
                    if condition == 'noise_m5db' and valid_noise_wavs:
                        noise_path = sel_rng.choice(valid_noise_wavs)
                        noise_wav, nsr = torchaudio.load(noise_path)
                        if nsr != sample_rate:
                            noise_wav = torchaudio.functional.resample(noise_wav, nsr, sample_rate)
                        noise_wav = noise_wav[0]
                        # Chunk noise to same duration, then mix
                        noise_wav = _chunk_wav(noise_wav, valid_duration, sample_rate, chunk_rng)
                        wav = _mix_noise(wav, noise_wav, snr_db=-5.0, sample_rate=sample_rate)

                    elif condition == 'overlap2_0db' and valid_spk2utt:
                        int_spk = sel_rng.choice([s for s in selected_spks if s != spk])
                        int_utts = valid_spk2utt.get(int_spk, spk2utts.get(int_spk, []))
                        if int_utts:
                            int_path = sel_rng.choice(int_utts)
                            int_wav, isr = torchaudio.load(int_path)
                            if isr != sample_rate:
                                int_wav = torchaudio.functional.resample(int_wav, isr, sample_rate)
                            int_wav = int_wav[0]
                            # Chunk interferer to same duration, then mix
                            int_wav = _chunk_wav(int_wav, valid_duration, sample_rate, chunk_rng)
                            wav = _mix_overlap(wav, int_wav, tir_db=0.0, sample_rate=sample_rate)

                    # condition == 'clean': 3s chunk only, no modification

                    feat = fbank(wav).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = embedding_model(feat).squeeze(0).cpu().numpy()
                    embs.append(emb)
                except Exception:
                    pass

            if len(embs) >= 2:
                spk2embs[spk] = np.stack(embs)

        if len(spk2embs) < 2:
            logger.warning(f"[VALID:{condition}] Speakers with >=2 valid embeddings: {len(spk2embs)}")
            return None

        # ---- Build trial pairs ----
        target_scores = []
        nontarget_scores = []
        valid_spks = list(spk2embs.keys())

        for i, s1 in enumerate(valid_spks):
            e1 = spk2embs[s1]
            for a in range(len(e1)):
                for b in range(a + 1, len(e1)):
                    target_scores.append(
                        np.dot(e1[a], e1[b]) /
                        (np.linalg.norm(e1[a]) * np.linalg.norm(e1[b]) + 1e-12))
            for j in range(i + 1, len(valid_spks)):
                e2 = spk2embs[valid_spks[j]]
                nontarget_scores.append(
                    np.dot(e1[0], e2[0]) /
                    (np.linalg.norm(e1[0]) * np.linalg.norm(e2[0]) + 1e-12))

        target_scores = np.array(target_scores)
        nontarget_scores = np.array(nontarget_scores)

        if len(target_scores) == 0 or len(nontarget_scores) == 0:
            logger.warning(f"[VALID:{condition}] No trial pairs")
            return None

        eer = _compute_eer(target_scores, nontarget_scores)
        logger.info(
            f"[VALID:{condition}] DONE  speakers={len(valid_spks)}  "
            f"target={len(target_scores)}  nontarget={len(nontarget_scores)}  "
            f"EER={eer:.4f}%")
        return eer

    finally:
        # ---- Restore global random states ----
        _random.setstate(py_rng)
        np.random.set_state(np_rng)
        torch.set_rng_state(torch_rng)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng)


# ============================================================
# Main
# ============================================================
def main():
    args, overrides = parser.parse_known_args(sys.argv[1:])
    config = build_config(args.config, overrides, True)

    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(args.gpu[rank])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl')

    set_seed(args.seed)

    os.makedirs(config.exp_dir, exist_ok=True)
    logger = get_logger('%s/train.log' % config.exp_dir)
    logger.info(f"Use GPU: {gpu} for training.")

    # ---- dataset ----
    if hasattr(config, 'overlap_spk2utt') and isinstance(config.overlap_spk2utt, str):
        import json as _json
        with open(config.overlap_spk2utt) as _f:
            config.overlap_spk2utt = _json.load(_f)
        if rank == 0:
            logger.info(f"[DATA] Loaded overlap interferer pool: {len(config.overlap_spk2utt)} speakers")

    train_dataset = build('dataset', config)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    config.dataloader['args']['sampler'] = train_sampler
    config.dataloader['args']['batch_size'] = int(config.batch_size / world_size)
    train_dataloader = build('dataloader', config)

    # ---- Student: embedding_model + classifier ----
    embedding_model = build('embedding_model', config)

    if args.pretrained:
        if rank == 0:
            logger.info(f"Loading pretrained weights from {args.pretrained} ...")
        pretrained_state = torch.load(args.pretrained, map_location='cpu')
        if 'state_dict' in pretrained_state:
            pretrained_state = pretrained_state['state_dict']
        if 'embedding_model' in pretrained_state:
            pretrained_state = pretrained_state['embedding_model']
        # Remove classifier keys if present
        pretrained_state_filtered = {
            k: v for k, v in pretrained_state.items()
            if not k.startswith('classifier') and not k.startswith('module.classifier')
        }
        missing, unexpected = embedding_model.load_state_dict(pretrained_state_filtered, strict=False)
        if rank == 0:
            logger.info(f"Student pretrained weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        pretrained_state_filtered = None
        if rank == 0:
            logger.warning("[WARN] No --pretrained provided. Student will start from random init.")

    # ---- Freeze Student modules ----
    frozen_bn_states = {}
    if hasattr(config, 'frozen_modules') and config.frozen_modules:
        _, _, frozen_bn_states = _apply_freeze(embedding_model, config.frozen_modules, logger)

    # ---- Teacher: fully frozen, loaded from SAME pretrained weights ----
    teacher_model = build('embedding_model', config)
    if pretrained_state_filtered is not None:
        teacher_model.load_state_dict(pretrained_state_filtered, strict=False)
        if rank == 0:
            logger.info("[TEACHER] Loaded original pretrained weights, fully frozen.")
    else:
        if rank == 0:
            logger.warning("[TEACHER] No pretrained weights available, Teacher random — check your setup!")

    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.eval()
    teacher_model.cuda()

    # ---- Classifier ----
    if hasattr(config, 'speed_pertub') and config.speed_pertub:
        config.num_classes = len(config.label_encoder) * 3
    else:
        config.num_classes = len(config.label_encoder)

    classifier = build('classifier', config)

    # Wrap Student in DDP via StudentModel
    student_model = StudentModel(embedding_model, classifier)
    student_model.cuda()
    student_model = torch.nn.parallel.DistributedDataParallel(student_model)

    # ---- Validation data (pre-load for speed) ----
    valid_noise_wavs = None
    valid_spk2utt_dict = None
    if args.valid_noise and os.path.exists(args.valid_noise):
        with open(args.valid_noise) as f:
            valid_noise_wavs = [line.strip().split()[1] for line in f if line.strip()]
        if rank == 0:
            logger.info(f"[VALID] Loaded {len(valid_noise_wavs)} validation noise files")
    if args.valid_spk2utt and os.path.exists(args.valid_spk2utt):
        import json as _json2
        with open(args.valid_spk2utt) as f:
            valid_spk2utt_dict = _json2.load(f)
        if rank == 0:
            logger.info(f"[VALID] Loaded overlap interferer pool: {len(valid_spk2utt_dict)} speakers")
    optimizer = _build_optimizer_with_layerwise_lr(embedding_model, classifier, config, logger)

    # ---- Losses ----
    criterion = build('loss', config)
    lambda_cons = getattr(config, 'lambda_cons', 0.25)
    if rank == 0:
        logger.info(f"[LOSS] lambda_cons = {lambda_cons}")

    # ---- Schedulers ----
    config.lr_scheduler['args']['step_per_epoch'] = len(train_dataloader)
    lr_scheduler = build('lr_scheduler', config)
    config.margin_scheduler['args']['step_per_epoch'] = len(train_dataloader)
    margin_scheduler = build('margin_scheduler', config)

    # ---- Checkpoint + tracking ----
    epoch_counter = build('epoch_counter', config)
    checkpointer = build('checkpointer', config)
    epoch_logger = EpochLogger(save_file=os.path.join(config.exp_dir, 'train_epoch.log'))
    best_eer = float('inf')
    best_epoch = 0
    best_robust_score = float('inf')
    best_robust_epoch = 0

    if args.resume:
        checkpointer.recover_if_possible(device='cuda')

    cudnn.benchmark = True

    # ============================================================
    # Training loop
    # ============================================================
    for epoch in epoch_counter:
        train_sampler.set_epoch(epoch)

        train_stats = train(
            train_dataloader, student_model, teacher_model,
            criterion, optimizer, epoch,
            lr_scheduler, margin_scheduler, logger, config, rank,
            frozen_bn_states, embedding_model, lambda_cons,
        )

        if rank == 0:
            epoch_logger.log_stats(stats_meta={"epoch": epoch}, stats=train_stats)
            checkpointer.save_checkpoint(epoch=epoch)

            # ---- Validation (3 conditions, deterministic 3s) ----
            if args.valid_data:
                embedding_model.eval()

                eer_clean = _validate(
                    embedding_model, args.valid_data,
                    torch.device(f'cuda:{gpu}'), logger,
                    condition='clean', num_spk=400, utts_per_spk=4,
                    valid_duration=3.0, seed=42)
                eer_noise = _validate(
                    embedding_model, args.valid_data,
                    torch.device(f'cuda:{gpu}'), logger,
                    condition='noise_m5db',
                    valid_noise_wavs=valid_noise_wavs,
                    num_spk=400, utts_per_spk=4,
                    valid_duration=3.0, seed=42)
                eer_overlap = _validate(
                    embedding_model, args.valid_data,
                    torch.device(f'cuda:{gpu}'), logger,
                    condition='overlap2_0db',
                    valid_spk2utt=valid_spk2utt_dict,
                    num_spk=400, utts_per_spk=4,
                    valid_duration=3.0, seed=42)

                embedding_model.train()
                _refreeze_bn(embedding_model, frozen_bn_states)

                # ---- Safe formatting helper ----
                def _fmt(v):
                    return f"{v:.4f}%" if v is not None else "N/A"

                # ---- Log all three (even if some are None) ----
                logger.info(f"[VALID] Epoch {epoch}:  "
                            f"EER_clean={_fmt(eer_clean)}  "
                            f"EER_noise_m5db={_fmt(eer_noise)}  "
                            f"EER_overlap2={_fmt(eer_overlap)}")

                # ---- Best by clean EER ----
                if eer_clean is not None:
                    if eer_clean < best_eer:
                        best_eer = eer_clean
                        best_epoch = epoch
                        best_dir = os.path.join(config.exp_dir, 'models', 'best_clean')
                        os.makedirs(best_dir, exist_ok=True)
                        torch.save(embedding_model.state_dict(),
                                   os.path.join(best_dir, 'embedding_model.ckpt'))
                        torch.save(classifier.state_dict(),
                                   os.path.join(best_dir, 'classifier.ckpt'))
                        logger.info(f"[VALID] New best CLEAN EER: {best_eer:.4f}% at epoch {best_epoch}")

                # ---- Best by robust score (only if ALL three valid) ----
                if eer_clean is not None and eer_noise is not None and eer_overlap is not None:
                    robust_score = eer_clean + eer_noise + eer_overlap
                    if robust_score < best_robust_score:
                        best_robust_score = robust_score
                        best_robust_epoch = epoch
                        best_robust_dir = os.path.join(config.exp_dir, 'models', 'best_robust')
                        os.makedirs(best_robust_dir, exist_ok=True)
                        torch.save(embedding_model.state_dict(),
                                   os.path.join(best_robust_dir, 'embedding_model.ckpt'))
                        torch.save(classifier.state_dict(),
                                   os.path.join(best_robust_dir, 'classifier.ckpt'))
                        logger.info(f"[VALID] New best ROBUST: {best_robust_score:.4f} "
                                    f"(clean={eer_clean:.2f} + noise={eer_noise:.2f} "
                                    f"+ overlap={eer_overlap:.2f}) at epoch {best_robust_epoch}")
                else:
                    logger.warning(f"[VALID] Epoch {epoch}: robust_score skipped — "
                                   f"clean={_fmt(eer_clean)} noise={_fmt(eer_noise)} "
                                   f"overlap={_fmt(eer_overlap)}")

            # BN freeze check
            if epoch == 1 or epoch == config.num_epoch:
                _check_bn_frozen(embedding_model, frozen_bn_states, logger)

        dist.barrier()

    if rank == 0:
        logger.info(f"[DONE] Best clean EER: {best_eer:.4f}% at epoch {best_epoch}")
        logger.info(f"[DONE] Best robust score: {best_robust_score:.4f} at epoch {best_robust_epoch}")
        logger.info(f"[DONE] Models saved:")
        logger.info(f"  best_clean:  {config.exp_dir}/models/best_clean/embedding_model.ckpt")
        logger.info(f"  best_robust: {config.exp_dir}/models/best_robust/embedding_model.ckpt")


# ============================================================
# Training function
# ============================================================
def train(train_loader, student_model, teacher_model,
          criterion, optimizer, epoch, lr_scheduler,
          margin_scheduler, logger, config, rank,
          frozen_bn_states, embedding_model, lambda_cons):
    train_stats = AverageMeters()
    train_stats.add('Time', ':6.3f')
    train_stats.add('Data', ':6.3f')
    train_stats.add('Loss', ':.4e')
    train_stats.add('L_arc', ':.4e')
    train_stats.add('L_cons', ':.4e')
    train_stats.add('Acc@1', ':6.2f')
    train_stats.add('Lr', ':.3e')
    train_stats.add('Margin', ':.3f')
    progress = ProgressMeter(
        len(train_loader), train_stats, prefix="Epoch: [{}]".format(epoch))

    student_model.train()
    _refreeze_bn(embedding_model, frozen_bn_states)

    end = time.time()
    for i, batch in enumerate(train_loader):
        train_stats.update('Data', time.time() - end)

        # Teacher-Student mode: batch = (feat_clean, feat_aug, y)
        feat_clean, feat_aug, y = batch

        iter_num = (epoch - 1) * len(train_loader) + i
        lr_scheduler.step(iter_num)
        margin_scheduler.step(iter_num)

        # Per-group LR check (first batch of each epoch)
        if rank == 0 and i == 0:
            lr_info = '  '.join(
                f"{pg.get('name', '?'):30s} lr={pg['lr']:.2e}"
                for pg in optimizer.param_groups
                if pg.get('name')
            )
            logger.info(f"[LR-EPOCH-{epoch}] {lr_info}")

        feat_clean = feat_clean.cuda(non_blocking=True)
        feat_aug = feat_aug.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        # ---- Teacher forward (clean, no grad) ----
        with torch.no_grad():
            e_teacher = teacher_model(feat_clean)

        # ---- Student forward (augmented) ----
        e_student, output = student_model(feat_aug)

        # ---- Losses ----
        L_arcface = criterion(output, y)
        L_cons = (1.0 - F.cosine_similarity(e_student, e_teacher.detach(), dim=1)).mean()
        loss = L_arcface + lambda_cons * L_cons

        acc1 = accuracy(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_stats.update('Loss', loss.item(), feat_aug.size(0))
        train_stats.update('L_arc', L_arcface.item(), feat_aug.size(0))
        train_stats.update('L_cons', L_cons.item(), feat_aug.size(0))
        train_stats.update('Acc@1', acc1.item(), feat_aug.size(0))
        train_stats.update('Lr', optimizer.param_groups[0]["lr"])
        train_stats.update('Margin', margin_scheduler.get_margin())
        train_stats.update('Time', time.time() - end)

        if rank == 0 and i % config.log_batch_freq == 0:
            logger.info(progress.display(i))

        end = time.time()

    return {
        'Avg_loss': train_stats.avg('Loss'),
        'Avg_L_arc': train_stats.avg('L_arc'),
        'Avg_L_cons': train_stats.avg('L_cons'),
        'Avg_acc': train_stats.avg('Acc@1'),
        'Lr_value': train_stats.val('Lr'),
    }


if __name__ == '__main__':
    main()
