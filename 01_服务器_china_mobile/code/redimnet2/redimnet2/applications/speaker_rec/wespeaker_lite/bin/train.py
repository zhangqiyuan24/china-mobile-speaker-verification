# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import sys
import math
import copy
import time
import shutil
import logging
import traceback
import pandas as pd
from pathlib import Path
from pprint import pformat

# Set the root logger's level to CRITICAL
logging.getLogger().setLevel(logging.CRITICAL)

import fire
import tableprint as tp
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

import wespeaker_lite.utils.schedulers as schedulers
from wespeaker_lite.dataset.dataset import Dataset, Vox1TestEvalDataset, Vox1TestEvalDatasetSimple, collate_func_single, collate_func_mult
from wespeaker_lite.models.projections import get_projection
from wespeaker_lite.models.mix_augs import Cutmix, Mixup, MixCutUp, OverlapMix
from wespeaker_lite.models.speaker_model import get_speaker_model, SpeakerModel
from wespeaker_lite.utils.checkpoint import save_checkpoint
from wespeaker_lite.utils.executor import run_epoch, eval_epoch, eval_epoch_simple
from wespeaker_lite.utils.file_utils import read_table
from wespeaker_lite.utils.utils import get_logger, parse_config_or_kwargs, set_seed, \
    spk2id, save_spk2id, load_spk2id, reassemble_projection_head
from wespeaker_lite.utils.utils import seed_worker

def make_mix_aug(configs):
    if ('mixup_alpha' in configs) and (configs['mixup_alpha'] is not None):
        print(f"USING MixUp aug")
        return Mixup(configs['mixup_alpha'])
    elif 'mix_aug_setup' in configs:
        print(f"USING {configs['mix_aug_setup']['type']} aug")
        return eval(configs['mix_aug_setup']['type'])(**configs['mix_aug_setup']['params'])
    else:
        print(f"not USING any mix aug")
        return None

def stage_lmdb_to_ram(src_path, local_rank, logger,
                      ram_root='/dev/shm/wespeaker_lmdb_ram'):
    """Copy an LMDB env dir onto tmpfs once per node (only local rank 0 copies).

    The caller must issue a dist.barrier() afterwards so the other ranks wait,
    then use the returned target iff its marker exists (else keep the original
    path). Idempotent: a complete prior copy (matching data.mdb size + marker)
    is reused, so resumes/reruns are instant. Because tmpfs is shared OS-wide,
    all workers across all GPUs mmap a single RAM-resident copy.

    Returns (target_dir: Path, marker_path: Path).
    """
    src = Path(src_path)
    tgt = Path(ram_root) / f"{src.parent.name}__{src.name}"
    marker = tgt / '.stage_complete'
    if local_rank == 0:
        try:
            src_data = src / 'data.mdb'
            if not src_data.is_file():
                raise FileNotFoundError(f"no data.mdb under {src}")
            need = src_data.stat().st_size
            if (marker.is_file() and (tgt / 'data.mdb').is_file()
                    and (tgt / 'data.mdb').stat().st_size == need):
                logger.warning(f"[lmdb_to_ram] reusing {tgt} "
                               f"({need / 2**30:.1f} GiB already in RAM)")
            else:
                if tgt.exists():
                    shutil.rmtree(tgt)
                probe = Path(ram_root)
                while not probe.exists():
                    probe = probe.parent
                free = shutil.disk_usage(probe).free
                if free < int(need * 1.03):
                    raise OSError(f"need {need / 2**30:.1f} GiB, only "
                                  f"{free / 2**30:.1f} GiB free under {probe}")
                tgt.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                shutil.copyfile(src_data, tgt / 'data.mdb')
                if (src / 'lock.mdb').is_file():
                    shutil.copyfile(src / 'lock.mdb', tgt / 'lock.mdb')
                marker.touch()
                logger.warning(f"[lmdb_to_ram] staged {src} -> {tgt} "
                               f"({need / 2**30:.1f} GiB) in "
                               f"{time.time() - t0:.1f}s")
        except Exception as e:
            logger.warning(f"[lmdb_to_ram] could not stage {src}: {e}; "
                           f"using original path")
    return tgt, marker


def train(config='conf/config.yaml', **kwargs):
    """Trains a model on the given features and spk labels.

    :config: A training configuration. Note that all parameters in the
             config can also be manually adjusted with --ARG VALUE
    :returns: None
    """
    logging.basicConfig(level=logging.WARNING)

    configs = parse_config_or_kwargs(config, **kwargs)
    checkpoint = configs.get('checkpoint', None)
    # dist configs
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl')

    model_dir = os.path.join(configs['exp_dir'], "models")
    if rank == 0:
        try:
            os.makedirs(model_dir)
        except IOError:
            print(model_dir + " already exists !!!")

    checkpoint = None
    start_epoch = 1
    if os.path.exists(model_dir):
        epoch_pattern = re.compile(r'model_(\d+)\.pt$')
        models = os.listdir(model_dir)
        last_epoch_time = 0
        for model in models:
            match = epoch_pattern.search(model)
            if not match:
                continue
            full_path = os.path.join(model_dir, model)
            if Path(full_path).is_file():
                print(full_path)
                if os.path.getmtime(full_path) > last_epoch_time:
                    last_epoch_time = os.path.getmtime(full_path)
                    checkpoint = full_path
                    start_epoch = int(match.group(1)) + 1

    dist.barrier(device_ids=[gpu])  # let the rank 0 mkdir first

    logger = get_logger(configs['exp_dir'], 'train.log')
    if world_size > 1:
        logger.info('training on multiple gpus, this gpu {}'.format(gpu))

    if rank == 0:
        logger.info("exp_dir is: {}".format(configs['exp_dir']))
        logger.info("<== Passed Arguments ==>")
        # Print arguments into logs
        for line in pformat(configs).split('\n'):
            logger.info(line)

    # seed
    set_seed(configs['seed'] + rank)
    os.environ["NUM_DATALOADER_WORKERS"] = str(configs['dataloader_args']['num_workers'])
    os.environ["TRAIN_SEED"] = str(configs['seed'])

    # train data
    assert configs['data_type'] in ['shard', 'raw', 'raw_fast'], \
        f"Unsupported data_type: {configs['data_type']}"
    train_label = configs['train_label']
    train_utt_spk_list = read_table(train_label)
    spk2id_dict = spk2id(train_utt_spk_list)
    if rank == 0:
        logger.info("<== Data statistics ==>")
        logger.info("train data num: {}, spk num: {}".format(
            len(train_utt_spk_list), len(spk2id_dict)))
        # Auto-save spk2id mapping for head reassembly in later stages
        spk2id_path = os.path.join(configs['exp_dir'], 'spk2id.json')
        save_spk2id(spk2id_dict, spk2id_path)
        logger.info(f"Saved spk2id mapping to {spk2id_path}")

    # projection layer
    configs['projection_args']['embed_dim'] = configs['model_args'][
        'embed_dim']
    configs['projection_args']['num_class'] = len(spk2id_dict)
    configs['projection_args']['do_lm'] = configs.get('do_lm', False)

    if configs['data_type'] != 'feat' and configs['dataset_args'][
            'speed_perturb']:
        # diff speed is regarded as diff spk
        if configs.get('do_lm', False):
            logger.info(
                'No speed perturb while doing large margin fine-tuning')
            configs['dataset_args']['speed_perturb'] = False
        else:
            configs['projection_args']['num_class'] *= 3

    # dataset and dataloader
    if isinstance(configs['train_data'], str):
        configs['train_data'] = configs['train_data'].replace('raw_fast','raw')

    # ── Optional: stage augmentation LMDBs into RAM (tmpfs / /dev/shm) ──
    # Random reads over the 12 GB musan + 2.2 GB rirs LMDBs are virtiofs
    # round-trips (readahead=False) and dominate the aug stage. Copy them once
    # per node to /dev/shm so every noise/RIR fetch is a shared RAM read. Opt-in
    # via dataset_args.lmdb_to_ram. Uses a deep copy so the saved config.yaml
    # keeps the original paths (clean resume); falls back to the original path
    # on any failure. Only the legacy reverb/noise_lmdb_file keys are staged.
    ds_args_runtime = configs['dataset_args']
    if configs['dataset_args'].get('lmdb_to_ram', False):
        ds_args_runtime = copy.deepcopy(configs['dataset_args'])
        aug_rt = ds_args_runtime.get('aug_setup') or {}
        _pending = {}
        if isinstance(aug_rt, dict):
            for _k in ('reverb_lmdb_file', 'noise_lmdb_file'):
                if aug_rt.get(_k):
                    _pending[_k] = stage_lmdb_to_ram(aug_rt[_k], gpu, logger)
        dist.barrier(device_ids=[gpu])  # wait for local-rank-0 copies
        for _k, (_tgt, _marker) in _pending.items():
            if _marker.is_file():
                aug_rt[_k] = str(_tgt)
                if rank == 0:
                    logger.warning(f"[lmdb_to_ram] {_k} -> {_tgt}")
            elif rank == 0:
                logger.warning(f"[lmdb_to_ram] {_k} not staged; "
                               f"using original {aug_rt[_k]}")

    # For right shuffle
    seed_worker(0)
    train_dataset = Dataset(configs['data_type'],
                            configs['train_data'],
                            ds_args_runtime,
                            spk2id_dict,
                            reverb_lmdb_file=configs.get('reverb_data', None),
                            noise_lmdb_file=configs.get('noise_data', None))
    try:
        print(f"train_dataset len : {len(train_dataset)}")
    except:
        print(f"Sizeless dataset!")
    print(f"dataloader args : {configs['dataloader_args']}")

    dataloader_args = {k: v for k, v in configs['dataloader_args'].items()
                       if k != 'seed_worker'}
    train_dataloader = DataLoader(train_dataset,
                                  persistent_workers=True,
                                  worker_init_fn=seed_worker if configs['dataloader_args'].get(
                                      'seed_worker', False) else None,
                                  **dataloader_args)

    batch_size = configs['dataloader_args']['batch_size']
    if configs['dataset_args'].get('sample_num_per_epoch', 0) > 0:
        sample_num_per_epoch = configs['dataset_args']['sample_num_per_epoch']
    else:
        sample_num_per_epoch = len(train_utt_spk_list)
    epoch_iter = sample_num_per_epoch // world_size // batch_size
    # optimizer_epoch_iter = number of optimizer steps per epoch (for schedulers)
    optimizer_epoch_iter = epoch_iter

    # Load log_df if exists, or create empty
    val_log_path = os.path.join(configs['exp_dir'], "val_log.csv")
    if os.path.exists(val_log_path):
        log_df = pd.read_csv(val_log_path)
    else:
        log_df = pd.DataFrame()

    # For older compatibility with old recipes:
    if configs.get('validation') is None:
        configs['validation'] = {"vox1" : configs['vox1_test_root']}

    # Create validation dataloaders
    val_dls_dict = dict()
    val_type = configs.get('val_type', 'local')

    assert val_type == 'local', f"Only 'local' val_type is supported, got: {val_type}"
    for val_name, val_root in configs['validation'].items():
        valid_dataset_fast = Vox1TestEvalDatasetSimple(val_root,duration=configs.get('eval_dur',8.0),
                                                       wavs_subroot='wav' if val_name == 'vox1' else '')
        valid_sampler_fast = torch.utils.data.distributed.DistributedSampler(
                                    valid_dataset_fast, num_replicas=world_size, rank=rank,
                                    shuffle=False, seed=23, drop_last=False)
        valid_dataloader_fast = DataLoader(valid_dataset_fast,
                                num_workers=8, shuffle=False, prefetch_factor=1,
                                sampler = valid_sampler_fast, worker_init_fn=None,
                                batch_size=configs['dataloader_args']['batch_size']//4,
                                collate_fn=collate_func_mult)

        valid_dataset_full = Vox1TestEvalDataset(val_root,wavs_subroot='wav' if val_name == 'vox1' else '')
        valid_sampler_full = torch.utils.data.distributed.DistributedSampler(
                                    valid_dataset_full, num_replicas=world_size, rank=rank,
                                    shuffle=False, seed=23, drop_last=False)
        valid_dataloader_full = DataLoader(valid_dataset_full,
                                num_workers=8, shuffle=False, prefetch_factor=1,
                                sampler = valid_sampler_full, worker_init_fn=None,
                                batch_size=1, collate_fn=collate_func_single)

        val_dls_dict[val_name] = (valid_dataloader_fast, valid_dataloader_full)

    if rank == 0:
        logger.info("<== Dataloaders ==>")
        logger.info("train dataloaders created")
        logger.info('epoch iteration number (micro-batches): {}'.format(epoch_iter))
        logger.info('epoch iteration number (optimizer steps): {}'.format(optimizer_epoch_iter))

    # model
    logger.info("<== Model ==>")
    model = get_speaker_model(configs['model'])(**configs['model_args'])
    num_params = sum(param.numel() for param in model.parameters())
    if rank == 0:
        print(model)
        logger.info('speaker_model size: {}'.format(num_params))

    print(f"projection_args : {configs['projection_args']['num_class']}")
    projection = get_projection(configs['projection_args'])
    model  = SpeakerModel(model, projection)

    # Determine checkpoint to load (model_ckpt takes priority over checkpoint)
    ckpt_to_load = configs.get('model_ckpt', None) or checkpoint

    if ckpt_to_load is not None:
        logger.info('Load initial model from {}'.format(ckpt_to_load))
        state_dict = torch.load(ckpt_to_load)

        # Load backbone (everything except projection head)
        loading_res = model.load_state_dict(
            {k: v for k, v in state_dict.items() if k != 'projection.weight'},
            strict=False)
        print(f"LOADING MODEL : {loading_res}")

        # Load projection head
        old_proj = state_dict.get('projection.weight')
        if old_proj is None:
            print("WARNING: no projection.weight in checkpoint, head initialized randomly")
        elif old_proj.shape == model.projection.weight.shape:
            # Same-stage resume (e.g. PT→PT): ckpt head matches current head
            # exactly, including the ×3 speed-perturb expansion. Load as-is;
            # reassembly would collapse the ×3 expansion and break resume.
            load_dict = {'weight': old_proj}
            if 'projection.bias' in state_dict:
                load_dict['bias'] = state_dict['projection.bias']
            loading_res_head = model.projection.load_state_dict(
                load_dict, strict=False)
            print(f"LOADED HEAD (same-stage resume, shape={tuple(old_proj.shape)}): {loading_res_head}")
        else:
            # Try reassembly via spk2id mapping (explicit or auto-detected)
            old_spk2id_path = configs.get('spk2id_map_path')
            if old_spk2id_path is None:
                # Auto-detect: look for spk2id.json next to the checkpoint
                ckpt_dir = os.path.dirname(os.path.dirname(ckpt_to_load))  # exp_dir/models/model_N.pt → exp_dir
                candidate = os.path.join(ckpt_dir, 'spk2id.json')
                if os.path.isfile(candidate):
                    old_spk2id_path = candidate

            if old_spk2id_path is not None and os.path.isfile(old_spk2id_path):
                # Reassemble head using speaker-level index mapping
                old_spk2id_map = load_spk2id(old_spk2id_path)
                new_proj_weight, stats = reassemble_projection_head(
                    old_proj_weight=old_proj,
                    old_spk2id=old_spk2id_map,
                    new_spk2id=spk2id_dict,
                )
                loading_res_head = model.projection.load_state_dict(
                    {'weight': new_proj_weight}, strict=False)
                print(f"REASSEMBLED HEAD from {old_spk2id_path}: {stats}")
                print(f"  load result: {loading_res_head}")
            else:
                # Fallback: try direct load, then slice
                try:
                    loading_res_head = model.projection.load_state_dict(
                        {'weight': old_proj}, strict=False)
                    print(f"LOADED HEAD (exact match): {loading_res_head}")
                except:
                    multiplier = 1
                    if configs['projection_args'].get('project_type') == 'arc_margin_intertopk_subcenter':
                        multiplier = 3
                    try:
                        load_dict = {'weight': old_proj[:configs['projection_args']['num_class'] * multiplier, :]}
                        if 'projection.bias' in state_dict:
                            load_dict['bias'] = state_dict['projection.bias']
                        loading_res_head = model.projection.load_state_dict(
                            load_dict, strict=False)
                        print(f"LOADED HEAD (sliced): {loading_res_head}")
                    except:
                        print(f"FAILED LOADING HEAD!")
                        print(traceback.format_exc())
    else:
        logger.info('Train model from scratch ...')

    logger.info('start_epoch: {}'.format(start_epoch))

    # ddp_model
    model.cuda()

    # Optional: torch.compile the backbone. Loading must happen before this so
    # checkpoint keys map cleanly. Saving strips `_orig_mod.` (see
    # utils/checkpoint.py) so checkpoints stay compile-transparent.
    compile_cfg = configs.get('compile_backbone')
    if compile_cfg:
        compile_kwargs = dict(
            mode='max-autotune-no-cudagraphs',
            dynamic=False,
            fullgraph=False,
        )
        if isinstance(compile_cfg, dict):
            compile_kwargs.update(compile_cfg)
        if rank == 0:
            logger.warning(f"Compiling model.backbone with torch.compile({compile_kwargs}). "
                           f"First batch at each new shape will pay compile time.")
        model.backbone = torch.compile(model.backbone, **compile_kwargs)
        if rank == 0:
            logger.warning(f"After compile: model.backbone class = {type(model.backbone).__name__}")

    ddp_model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
    device = torch.device("cuda")

    criterion = getattr(torch.nn, configs['loss'])(**configs['loss_args'])
    if rank == 0:
        logger.info("<== Loss ==>")
        logger.info("loss criterion is: " + configs['loss'])

    optim_cls = getattr(torch.optim, configs['optimizer'])
    optimizer = optim_cls([
                {'params': model.backbone.parameters(), 'weight_decay' : configs['optimizer_args']['weight_decay']},
                {'params': model.projection.parameters(), 'weight_decay' : configs.get('loss_weight_decay_mult',1.0) * configs['optimizer_args']['weight_decay']}
            ],**configs['optimizer_args'])

    if rank == 0:
        print(f"OPTIMIZER PARAM GROUPS")
        for pg in optimizer.param_groups:
            print({k:v for k,v in pg.items() if k != 'params'})

    if rank == 0:
        logger.info("<== Optimizer ==>")
        logger.info("optimizer is: " + configs['optimizer'])

    # scheduler
    configs['scheduler_args']['num_epochs'] = configs['num_epochs']
    configs['scheduler_args']['epoch_iter'] = optimizer_epoch_iter
    configs['scheduler_args']['scale_ratio'] = 1.0 * world_size * configs[
        'dataloader_args']['batch_size'] / 64
    scheduler = getattr(schedulers,
                        configs['scheduler'])(optimizer,
                                              **configs['scheduler_args'])
    if rank == 0:
        logger.info("<== Scheduler ==>")
        logger.info("scheduler is: " + configs['scheduler'])

    # margin scheduler
    configs['margin_update']['epoch_iter'] = optimizer_epoch_iter
    margin_scheduler = getattr(schedulers, configs['margin_scheduler'])(
        model=model, **configs['margin_update'])
    if rank == 0:
        logger.info("<== MarginScheduler ==>")

    # save config.yaml
    if rank == 0:
        saved_config_path = os.path.join(configs['exp_dir'], 'config.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    # training
    dist.barrier(device_ids=[gpu])  # synchronize here
    if rank == 0:
        logger.info("<========== Training process ==========>")
        header = ['Epoch', 'Batch', 'Lr', 'Margin', 'Loss', "Acc"]
        for line in tp.header(header, width=10, style='grid').split('\n'):
            logger.info(line)
    dist.barrier(device_ids=[gpu])  # synchronize here

    scaler = torch.cuda.amp.GradScaler(enabled=configs['enable_amp'])

    training_loop(
        start_epoch, configs, train_dataset, train_dataloader, epoch_iter,
        ddp_model, model, criterion, optimizer, scaler, scheduler, margin_scheduler,
        rank, world_size, gpu, logger, log_df,
        val_dls_dict, val_type, model_dir
    )

    if rank == 0:
        final_model_link = os.path.join(model_dir, 'final_model.pt')
        if os.path.islink(final_model_link) or os.path.exists(final_model_link):
            os.remove(final_model_link)
        os.symlink('model_{}.pt'.format(configs['num_epochs']),
                   final_model_link)
        logger.info(tp.bottom(len(header), width=10, style='grid'))

def training_loop(
    start_epoch, configs, train_dataset, train_dataloader, epoch_iter,
    ddp_model, model, criterion, optimizer, scaler, scheduler, margin_scheduler,
    rank, world_size, gpu, logger, log_df,
    val_dls_dict, val_type, model_dir
):
    """Per-epoch loop: train, save checkpoint (+ prune), validate, log."""

    device = torch.device("cuda")
    # Use a mutable container so the per-epoch logging can update log_df
    _log_df_ref = [log_df]

    _mix_aug = make_mix_aug(configs)

    for epoch in range(start_epoch, configs['num_epochs'] + 1):
        train_dataset.set_epoch(epoch)

        run_epoch(train_dataloader,
                epoch_iter,
                ddp_model,
                criterion,
                optimizer,
                scheduler,
                margin_scheduler,
                epoch,
                rank,
                logger,
                scaler,
                mix_aug=_mix_aug,
                enable_amp=configs['enable_amp'],
                log_batch_interval=configs['log_batch_interval'],
                device=device)

        dist.barrier(device_ids=[gpu])

        eval_func_kwargs = dict(
            model=ddp_model,
            optimizer=optimizer,
            enable_amp=configs['enable_amp'],
            epoch=epoch,
            rank=rank,
            exp_dir=configs['exp_dir'],
            device=device)

        if rank == 0:
            curr_ckpt = os.path.join(model_dir, 'model_{}.pt'.format(epoch))
            prev_ckpt = os.path.join(model_dir, 'model_{}.pt'.format(max(epoch - 1, 0)))
            save_checkpoint(model, curr_ckpt)
            if not (max(epoch - 1, 0) % configs['save_epoch_interval'] == 0
                    or max(epoch - 1, 0) >= configs['num_epochs'] - configs['num_avg']):
                Path(prev_ckpt).unlink()

        tot_new_row = _run_validation(
            val_dls_dict, val_type, epoch, configs, rank, _log_df_ref[0], eval_func_kwargs
        )

        # Log results
        if len(tot_new_row):
            csv_out_path = os.path.join(configs['exp_dir'], "val_log.csv")
            new_row_df = pd.DataFrame([tot_new_row])
            if not _log_df_ref[0].empty:
                new_row_df = new_row_df.reindex(columns=_log_df_ref[0].columns)
            _log_df_ref[0] = pd.concat([_log_df_ref[0], new_row_df], ignore_index=True)
            _log_df_ref[0].to_csv(csv_out_path, index=False)


def _run_validation(val_dls_dict, val_type, epoch, configs, rank, log_df, eval_func_kwargs):
    """Extract validation loop to avoid 3x duplication."""
    tot_new_row = {}
    for val_name, val_dls in val_dls_dict.items():
        (valid_dataloader_fast, valid_dataloader_full) = val_dls
        if ((epoch % configs['save_epoch_interval'] == 0) or (epoch >= configs[
                    'num_epochs'] - configs['num_avg'])) and configs.get('full_val', False):
            if rank == 0:
                print(f"EVAL FULL")
            new_row = eval_epoch(valid_dataloader_full, val_ds_name=val_name,
                                 log_df=log_df, **eval_func_kwargs)
        else:
            if rank == 0:
                print(f"EVAL FAST")
            new_row = eval_epoch_simple(valid_dataloader_fast, val_ds_name=val_name,
                                        log_df=log_df, **eval_func_kwargs)

        if new_row is not None:
            tot_new_row.update(new_row)
    return tot_new_row

if __name__ == '__main__':
    fire.Fire(train)
