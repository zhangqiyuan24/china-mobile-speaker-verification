# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from speakerlab.utils.utils import set_seed, get_logger, AverageMeters, ProgressMeter, accuracy
from speakerlab.utils.config import build_config
from speakerlab.utils.builder import build
from speakerlab.utils.epoch import EpochCounter, EpochLogger


parser = argparse.ArgumentParser(description='Speaker Network Training')
parser.add_argument('--config', default='', type=str, help='Config file for training')
parser.add_argument('--resume', default=True, type=bool, help='Resume from recent checkpoint or not')
parser.add_argument('--pretrained', default='', type=str, help='Path to pretrained embedding model checkpoint')
parser.add_argument('--seed', default=1234, type=int, help='Random seed for training.')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')


def _apply_freeze(embedding_model, frozen_patterns, logger):
    """Freeze modules whose names match any of the given patterns.
    Frozen BN layers are set to eval mode so running_mean/var are not updated.
    Returns list of frozen module names.
    """
    frozen_names = []
    trainable_names = []
    for name, module in embedding_model.named_modules():
        should_freeze = any(p in name for p in frozen_patterns)
        if should_freeze:
            for param in module.parameters(recurse=False):
                param.requires_grad = False
            frozen_names.append(name)
            # Keep BN in eval mode so running stats don't change
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()
        else:
            # Only count leaf modules with params
            if len(list(module.parameters(recurse=False))) > 0:
                trainable_names.append(name)

    total_params = sum(p.numel() for p in embedding_model.parameters())
    trainable_params = sum(p.numel() for p in embedding_model.parameters() if p.requires_grad)

    logger.info(f"[FREEZE] Total params: {total_params:,}  "
                f"Trainable: {trainable_params:,}  "
                f"({100*trainable_params/total_params:.1f}%)")
    logger.info(f"[FREEZE] Frozen modules: {frozen_names}")
    logger.info(f"[FREEZE] Trainable modules: {trainable_names}")
    return frozen_names, trainable_names


def _build_optimizer_with_layerwise_lr(embedding_model, classifier, config, logger):
    """Build optimizer with per-module learning rates if layerwise_lr is configured."""
    if hasattr(config, 'layerwise_lr') and config.layerwise_lr:
        param_groups = []
        lr_config = config.layerwise_lr
        base_lr = config.lr

        for name, module in list(embedding_model.named_modules()) + [('classifier', classifier)]:
            module_params = [p for p in module.parameters(recurse=False) if p.requires_grad]
            if not module_params:
                continue

            lr_mult = base_lr  # default
            for pattern, mult in lr_config.items():
                if pattern in name:
                    lr_mult = base_lr * mult
                    break

            param_groups.append({'params': module_params, 'lr': lr_mult, 'name': name})

        # Also add any ungrouped params
        grouped_ids = set()
        for pg in param_groups:
            for p in pg['params']:
                grouped_ids.add(id(p))
        remaining = [p for p in list(embedding_model.parameters()) + list(classifier.parameters())
                     if p.requires_grad and id(p) not in grouped_ids]
        if remaining:
            param_groups.append({'params': remaining, 'lr': base_lr, 'name': 'rest'})

        if logger and dist.get_rank() == 0:
            for pg in param_groups:
                logger.info(f"  [LR] {pg['name']}: lr={pg['lr']:.2e}, params={sum(p.numel() for p in pg['params'])}")

        config.optimizer['args']['params'] = param_groups
    else:
        all_params = list(embedding_model.parameters()) + list(classifier.parameters())
        all_params = [p for p in all_params if p.requires_grad]
        config.optimizer['args']['params'] = all_params

    return build('optimizer', config)


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
    if hasattr(config, 'frozen_modules') and config.frozen_modules:
        _apply_freeze(embedding_model, config.frozen_modules, logger)

    # --- classifier ---
    if hasattr(config, 'speed_pertub') and config.speed_pertub:
        config.num_classes = len(config.label_encoder) * 3
    else:
        config.num_classes = len(config.label_encoder)

    classifier = build('classifier', config)
    model = nn.Sequential(embedding_model, classifier)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)

    # --- optimizer (with layerwise LR support) ---
    optimizer = _build_optimizer_with_layerwise_lr(embedding_model, classifier, config, logger)

    # --- loss ---
    criterion = build('loss', config)

    # --- schedulers ---
    config.lr_scheduler['args']['step_per_epoch'] = len(train_dataloader)
    lr_scheduler = build('lr_scheduler', config)
    config.margin_scheduler['args']['step_per_epoch'] = len(train_dataloader)
    margin_scheduler = build('margin_scheduler', config)

    # --- checkpoint ---
    epoch_counter = build('epoch_counter', config)
    checkpointer = build('checkpointer', config)
    epoch_logger = EpochLogger(save_file=os.path.join(config.exp_dir, 'train_epoch.log'))

    if args.resume:
        checkpointer.recover_if_possible(device='cuda')

    cudnn.benchmark = True

    for epoch in epoch_counter:
        train_sampler.set_epoch(epoch)

        train_stats = train(
            train_dataloader, model, criterion, optimizer, epoch,
            lr_scheduler, margin_scheduler, logger, config, rank,
        )

        if rank == 0:
            epoch_logger.log_stats(stats_meta={"epoch": epoch}, stats=train_stats)
            if epoch % config.save_epoch_freq == 0:
                checkpointer.save_checkpoint(epoch=epoch)

        dist.barrier()


def train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, margin_scheduler, logger, config, rank):
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
