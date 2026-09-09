# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import argparse
import time
import copy

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import numpy as np

from speakerlab.utils.utils import set_seed, get_logger, AverageMeters, ProgressMeter, accuracy
from speakerlab.utils.config import build_config
from speakerlab.utils.builder import build
from speakerlab.utils.epoch import EpochCounter, EpochLogger


parser = argparse.ArgumentParser(description='Speaker Network Training')
parser.add_argument('--config', default='', type=str, help='Config file for training')
parser.add_argument('--resume', default=True, type=bool, help='Resume from recent checkpoint or not')
parser.add_argument('--pretrained', default='', type=str, help='Path to pretrained embedding model checkpoint')
parser.add_argument('--valid_data', default='', type=str, help='Path to validation CSV')
parser.add_argument('--seed', default=1234, type=int, help='Random seed for training.')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')


def _apply_freeze(embedding_model, frozen_patterns, logger):
    """Freeze modules matching patterns. Frozen BN → eval mode.
    Returns (frozen_bn_names, frozen_bn_initial_states).
    """
    frozen_names = []
    trainable_names = []
    frozen_bn_states = {}  # name → (running_mean, running_var)

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
    """Build optimizer. layerwise_lr values are absolute LRs (not multipliers)."""
    if hasattr(config, 'layerwise_lr') and config.layerwise_lr:
        param_groups = []
        lr_config = config.layerwise_lr
        for name, module in list(embedding_model.named_modules()) + [('classifier', classifier)]:
            module_params = [p for p in module.parameters(recurse=False) if p.requires_grad]
            if not module_params:
                continue
            lr = config.lr  # default
            for pattern, abs_lr in lr_config.items():
                if pattern in name:
                    lr = abs_lr
                    break
            param_groups.append({'params': module_params, 'lr': lr, 'name': name})
        # catch any remaining
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


def _validate(embedding_model, valid_csv, device, logger):
    """Simple speaker verification validation: same-speaker vs different-speaker pairs."""
    import csv
    from collections import defaultdict
    from speakerlab.process.processor import FBank
    import torchaudio

    spk2utts = defaultdict(list)
    with open(valid_csv) as f:
        for row in csv.DictReader(f):
            spk2utts[row['spk']].append(row['path'])

    spk_ids = list(spk2utts.keys())
    if len(spk_ids) < 2:
        logger.warning("[VALID] Not enough speakers for EER computation")
        return None

    # Extract embeddings
    fbank = FBank(80, sample_rate=16000, mean_nor=True)
    spk2embs = {}
    for spk in spk_ids[:50]:  # limit to 50 speakers for speed
        embs = []
        for wav_path in spk2utts[spk][:10]:  # max 10 utts per speaker
            try:
                wav, sr = torchaudio.load(wav_path)
                if sr != 16000:
                    wav = torchaudio.functional.resample(wav, sr, 16000)
                wav = wav[0].unsqueeze(0)
                feat = fbank(wav).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = embedding_model(feat).squeeze(0).cpu().numpy()
                embs.append(emb)
            except Exception:
                pass
        if len(embs) >= 2:
            spk2embs[spk] = np.stack(embs)

    if len(spk2embs) < 2:
        return None

    # Build trial pairs
    target_scores = []
    nontarget_scores = []
    valid_spks = list(spk2embs.keys())
    for i, s1 in enumerate(valid_spks):
        e1 = spk2embs[s1]
        # Target: within-speaker
        for a in range(len(e1)):
            for b in range(a + 1, len(e1)):
                target_scores.append(np.dot(e1[a], e1[b]) /
                                    (np.linalg.norm(e1[a]) * np.linalg.norm(e1[b]) + 1e-12))
        # Nontarget: cross-speaker
        for j in range(i + 1, len(valid_spks)):
            e2 = spk2embs[valid_spks[j]]
            nontarget_scores.append(np.dot(e1[0], e2[0]) /
                                    (np.linalg.norm(e1[0]) * np.linalg.norm(e2[0]) + 1e-12))

    target_scores = np.array(target_scores)
    nontarget_scores = np.array(nontarget_scores)
    if len(target_scores) == 0 or len(nontarget_scores) == 0:
        return None

    eer = _compute_eer(target_scores, nontarget_scores)
    return eer


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

    # --- dataset ---
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

    # --- model ---
    embedding_model = build('embedding_model', config)

    if args.pretrained:
        if rank == 0:
            logger.info(f"Loading pretrained weights from {args.pretrained} ...")
        pretrained_state = torch.load(args.pretrained, map_location='cpu')
        if 'state_dict' in pretrained_state:
            pretrained_state = pretrained_state['state_dict']
        if 'embedding_model' in pretrained_state:
            pretrained_state = pretrained_state['embedding_model']
        pretrained_state = {k: v for k, v in pretrained_state.items()
                            if not k.startswith('classifier') and not k.startswith('module.classifier')}
        missing, unexpected = embedding_model.load_state_dict(pretrained_state, strict=False)
        if rank == 0:
            logger.info(f"Pretrained weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    # --- freeze ---
    frozen_bn_states = {}
    if hasattr(config, 'frozen_modules') and config.frozen_modules:
        _, _, frozen_bn_states = _apply_freeze(embedding_model, config.frozen_modules, logger)

    # --- classifier ---
    if hasattr(config, 'speed_pertub') and config.speed_pertub:
        config.num_classes = len(config.label_encoder) * 3
    else:
        config.num_classes = len(config.label_encoder)

    classifier = build('classifier', config)
    model = nn.Sequential(embedding_model, classifier)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)

    # --- optimizer ---
    optimizer = _build_optimizer_with_layerwise_lr(embedding_model, classifier, config, logger)

    # --- loss ---
    criterion = build('loss', config)

    # --- schedulers ---
    config.lr_scheduler['args']['step_per_epoch'] = len(train_dataloader)
    lr_scheduler = build('lr_scheduler', config)
    config.margin_scheduler['args']['step_per_epoch'] = len(train_dataloader)
    margin_scheduler = build('margin_scheduler', config)

    # --- checkpoint + tracking ---
    epoch_counter = build('epoch_counter', config)
    checkpointer = build('checkpointer', config)
    epoch_logger = EpochLogger(save_file=os.path.join(config.exp_dir, 'train_epoch.log'))
    best_eer = float('inf')
    best_epoch = 0

    if args.resume:
        checkpointer.recover_if_possible(device='cuda')

    cudnn.benchmark = True

    for epoch in epoch_counter:
        train_sampler.set_epoch(epoch)

        train_stats = train(
            train_dataloader, model, criterion, optimizer, epoch,
            lr_scheduler, margin_scheduler, logger, config, rank,
            frozen_bn_states, embedding_model,
        )

        if rank == 0:
            epoch_logger.log_stats(stats_meta={"epoch": epoch}, stats=train_stats)
            # Always save per epoch
            checkpointer.save_checkpoint(epoch=epoch)

            # Validation
            if args.valid_data:
                embedding_model.eval()
                valid_eer = _validate(embedding_model, args.valid_data,
                                      torch.device(f'cuda:{gpu}'), logger)
                embedding_model.train()
                _refreeze_bn(embedding_model, frozen_bn_states)

                if valid_eer is not None:
                    logger.info(f"[VALID] Epoch {epoch}: valid EER = {valid_eer:.2f}%")
                    if valid_eer < best_eer:
                        best_eer = valid_eer
                        best_epoch = epoch
                        # Save best as separate copy
                        best_dir = os.path.join(config.exp_dir, 'models', 'best')
                        os.makedirs(best_dir, exist_ok=True)
                        torch.save(embedding_model.state_dict(),
                                   os.path.join(best_dir, 'embedding_model.ckpt'))
                        torch.save(classifier.state_dict(),
                                   os.path.join(best_dir, 'classifier.ckpt'))
                        logger.info(f"[VALID] New best EER: {best_eer:.2f}% at epoch {best_epoch}")

            # BN freeze check (first and last epoch only)
            if epoch == 1 or epoch == config.num_epoch:
                _check_bn_frozen(embedding_model, frozen_bn_states, logger)

        dist.barrier()

    if rank == 0:
        logger.info(f"[DONE] Best valid EER: {best_eer:.2f}% at epoch {best_epoch}")


def train(train_loader, model, criterion, optimizer, epoch, lr_scheduler,
          margin_scheduler, logger, config, rank, frozen_bn_states, embedding_model):
    train_stats = AverageMeters()
    train_stats.add('Time', ':6.3f')
    train_stats.add('Data', ':6.3f')
    train_stats.add('Loss', ':.4e')
    train_stats.add('Acc@1', ':6.2f')
    train_stats.add('Lr', ':.3e')
    train_stats.add('Margin', ':.3f')
    progress = ProgressMeter(
        len(train_loader), train_stats, prefix="Epoch: [{}]".format(epoch))

    model.train()
    # Re-freeze frozen BN after model.train()
    _refreeze_bn(embedding_model, frozen_bn_states)

    end = time.time()
    for i, (x, y) in enumerate(train_loader):
        train_stats.update('Data', time.time() - end)

        iter_num = (epoch - 1) * len(train_loader) + i
        lr_scheduler.step(iter_num)
        margin_scheduler.step(iter_num)

        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        output = model(x)
        loss = criterion(output, y)
        acc1 = accuracy(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_stats.update('Loss', loss.item(), x.size(0))
        train_stats.update('Acc@1', acc1.item(), x.size(0))
        train_stats.update('Lr', optimizer.param_groups[0]["lr"])
        train_stats.update('Margin', margin_scheduler.get_margin())
        train_stats.update('Time', time.time() - end)

        if rank == 0 and i % config.log_batch_freq == 0:
            logger.info(progress.display(i))

        end = time.time()

    return {
        'Avg_loss': train_stats.avg('Loss'),
        'Avg_acc': train_stats.avg('Acc@1'),
        'Lr_value': train_stats.val('Lr'),
    }


if __name__ == '__main__':
    main()
