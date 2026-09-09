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

import tableprint as tp
import functools
import os
import torch
import pandas as pd
from tqdm import tqdm
import torchnet as tnt
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather

from torch import nn
from torch.distributions import Beta
from torch.nn.parameter import Parameter

import numpy as np
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d

class Mixup(nn.Module):
    def __init__(self, mix_beta):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device).to(X.dtype)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        Y = coeffs.view(-1, 1).to(Y.dtype) * Y + (1 - coeffs.view(-1, 1).to(Y.dtype)) * Y[perm]

        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            return X, Y, weight

def to_ohe(num_classes, targets):
    # target_mask = embeds.new_zeros(embeds.size())
    target_mask = torch.zeros((targets.size()[0], num_classes),
                              dtype=torch.float,device=targets.device)
    target_mask.scatter_(1, targets.view(-1, 1).long(), 1.0)
    return target_mask

def run_epoch(dataloader,
              epoch_iter,
              model,
              criterion,
              optimizer,
              scheduler,
              margin_scheduler,
              epoch,
              rank,
              logger,
              scaler,
              mix_aug,
              enable_amp,
              log_batch_interval=100,
              device=torch.device('cuda')):
    model.train()

    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    for i, batch in enumerate(tqdm(dataloader, total=epoch_iter,
                                   disable=(rank!=0), desc=f'Epoch: {epoch:03d}')):
        utts = batch['key']
        targets = batch['label']
        wav = batch['wav']
        # print(batch.keys())
        # features = batch['feat']

        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        margin_scheduler.step(cur_iter)

        # features = features.float().to(device)  # (B,T,F)
        wav = wav.float().to(device)  # (B,T,F)
        # print(f"wav : {wav.size()}")
        targets = targets.long().to(device)

        with torch.cuda.amp.autocast(enabled=enable_amp):
            target_mask = None
            if mix_aug is not None:
                num_classes = model.module.projection.out_features
                target_mask = to_ohe(num_classes, targets)
                wav, target_mask = mix_aug(wav, target_mask)
                # targets = None

            outputs = model(wav)  # (embed_a,embed_b) in most cases
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            # outputs = model.module.projection.float()(embeds.float(), targets)

        with torch.cuda.amp.autocast(enabled=False):
            outputs = model.module.projection(embeds.float(), targets, target_mask)
            if isinstance(outputs, tuple):
                outputs, loss = outputs
            else:
                loss = criterion(outputs, targets)

        # loss, acc
        if (i + 1) % 16 == 0:
            loss_meter.add(loss.item())
            acc_meter.add(outputs.cpu().detach().float().numpy(), targets.cpu().numpy())

        # updata the model
        optimizer.zero_grad()
        # scaler does nothing here if enable_amp=False
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # log
        if (i + 1) % log_batch_interval == 0:
            # loss_meter.add(loss.item())
            # acc_meter.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())
            print(f"wav.size : {wav.size()}")
            logger.warning(
                tp.row((epoch, i + 1, scheduler.get_lr(),
                        margin_scheduler.get_margin()) +
                       (loss_meter.value()[0], acc_meter.value()[0]),
                       width=10,
                       style='grid'))

        if (i + 1) == epoch_iter:
            break

    logger.info(
        tp.row(
            (epoch, i + 1, scheduler.get_lr(), margin_scheduler.get_margin()) +
            (loss_meter.value()[0], acc_meter.value()[0]),
            width=10,
            style='grid'))

import numpy as np
from sklearn import metrics
from six.moves import range
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d

# https://stackoverflow.com/questions/28339746/equal-error-rate-in-python
def EER(y, y_score):
    """
    :param y: True binary labels in range {0, 1} or {-1, 1}. If labels are not binary, pos_label should be explicitly given
    :param y_score: Target scores, can either be probability estimates of the positive class, confidence values,
    or non-thresholded measure of decisions (as returned by â€œdecision_functionâ€ on some classifiers).
    :return: eer, thresh
    """
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh.item()

def cos_sim_mat(x0,x1):
    """
    x0 shape : (N,D)
    x1 shape : (M,D)
    D - dimensionality of embeddings
    """
    x0n = x0 / (np.linalg.norm(x0,axis=-1,keepdims=True)+1e-12)
    x1n = x1 / (np.linalg.norm(x1,axis=-1,keepdims=True)+1e-12)
    return np.sum(x0n[:,None,:]*x1n[None,:,:],axis=-1)

def get_lr(opt):
    for g in opt.param_groups:
        return g['lr']

def get_wd(opt):
    for g in opt.param_groups:
        return g['weight_decay']

def wraped_all_gather(tensor,group=None):
    with torch.no_grad():
        gathered_tensors = all_gather(tensor, group)
    return torch.stack(gathered_tensors)

def eval_epoch(
    validation_dataloader,
    model,
    optimizer,
    enable_amp,
    epoch,
    rank,
    log_df,
    exp_dir,
    val_ds_name="vox1",
    device=torch.device('cuda')
):
    model.eval()

    predicted_batches = []

    with torch.no_grad():
        for batch in tqdm(validation_dataloader, disable=(rank!=0), desc='validating'):
            (x0,x1),labels = batch

            labels = labels.to(device)
            x0 = x0.to(device)
            x1 = x1.to(device)

            with torch.cuda.amp.autocast(enabled=enable_amp):
                embed0 = model(x0)
                embed1 = model(x1)

            embeds = torch.cat([embed0,embed1],dim=0)
            embeds = F.normalize(embeds, p=2, dim=1)

            embeds = wraped_all_gather(embeds)
            labels = wraped_all_gather(labels)

            if rank == 0:
                predicted_batches.append((embeds.cpu(),labels.cpu()))

            del embeds, labels

    # Perform validation
    if rank == 0:
        new_row = {
            "epoch" : epoch,
            "lr" : get_lr(optimizer),
            'wd' : optimizer.param_groups[0]['weight_decay'],
            "margin" : model.module.projection.margin
        }

        labels = []
        embeddings = []

        for batch_embeddings, batch_labels in predicted_batches:
            if len(batch_labels.size()) == 2:
                for rank in range(batch_labels.size()[0]):
                    labels.append(batch_labels[rank].numpy())
                    embeddings.append(batch_embeddings[rank].numpy()[None,...])
            if len(batch_labels.size()) == 1:
                labels.append(batch_labels.numpy())
                embeddings.append(batch_embeddings.numpy()[None,...])

        labels = np.concatenate(labels,axis=0)
        embeddings = np.concatenate(embeddings,axis=0)

        print(f"total validation embeddings.shape : {embeddings.shape}")
        print(f"total validation labels.shape : {labels.shape}")

        assert len(labels) == len(embeddings)
        emb_dict = {}
        for i in range(len(labels)):
            emb_dict[labels[i].item()] = embeddings[i]

        scr0 = []
        scr1 = []
        # scr_avg = []
        protocol = validation_dataloader.dataset.vox1_protocol.copy()
        for _, row in protocol.iterrows():
            scr0.append(np.mean(cos_sim_mat(emb_dict[row.enroll_ind][:1],emb_dict[row.verify_ind][:1])).item())
            scr1.append(np.mean(cos_sim_mat(emb_dict[row.enroll_ind][1:],emb_dict[row.verify_ind][1:])).item())

        protocol['scr0'] = scr0
        protocol['scr1'] = scr1
        protocol['scr_avg'] = (protocol['scr0']+protocol['scr1'])/2

        # eer, _ = EER(protocol.label.values,protocol.score.values)
        new_row[f"{val_ds_name}-eer-fl"] = np.around(EER(protocol.label.values,protocol.scr0.values)[0]*100,3)
        new_row[f"{val_ds_name}-eer-tta"] = np.around(EER(protocol.label.values,protocol.scr1.values)[0]*100,3)
        new_row[f"{val_ds_name}-eer-avg"] = np.around(EER(protocol.label.values,protocol.scr_avg.values)[0]*100,3)

        # csv_out_path = os.path.join(exp_dir, "val_log.csv")
        # log_df = log_df.append(new_row, ignore_index=True)
        # log_df.to_csv(csv_out_path, index=False)
        # return log_df
        return new_row

def eval_epoch_simple(
    validation_dataloader,
    model,
    optimizer,
    enable_amp,
    epoch,
    rank,
    log_df,
    exp_dir,
    val_ds_name="vox1",
    device=torch.device('cuda')
):
    model.eval()

    predicted_batches = []

    with torch.no_grad():
        for batch in tqdm(validation_dataloader, disable=(rank!=0), desc='validating'):
            x,labels = batch

            # print(f"x : {x.size()}, labels : {labels.size()}")

            labels = labels.to(device)
            x = x.to(device)

            with torch.cuda.amp.autocast(enabled=enable_amp):
                embeds = model(x)

            embeds = F.normalize(embeds, p=2, dim=1)

            embeds = wraped_all_gather(embeds)
            labels = wraped_all_gather(labels)

            if rank == 0:
                predicted_batches.append((embeds.cpu(),labels.cpu()))

            del embeds, labels

    # Perform validation
    if rank == 0:
        new_row = {
            "epoch" : epoch,
            "lr" : get_lr(optimizer),
            "wd" : get_wd(optimizer),
            "margin" : model.module.projection.margin
        }

        labels = []
        embeddings = []

        for batch_embeddings, batch_labels in predicted_batches:
            # print(f"batch_embeddings : {batch_embeddings.size()}, batch_labels : {batch_labels.size()}")
            if len(batch_labels.size()) == 3:
                for rank in range(batch_labels.size()[0]):
                    labels.append(batch_labels[rank].numpy())
                    embeddings.append(batch_embeddings[rank].numpy())
            if len(batch_labels.size()) == 2:
                labels.append(batch_labels.numpy())
                embeddings.append(batch_embeddings.numpy())

        labels = np.concatenate(labels,axis=0)
        embeddings = np.concatenate(embeddings,axis=0)

        print(f"total validation embeddings.shape : {embeddings.shape}")
        print(f"total validation labels.shape : {labels.shape}")

        assert len(labels) == len(embeddings)
        emb_dict = {}
        for i in range(len(labels)):
            emb_dict[labels[i].item()] = embeddings[i]

        scr0 = []
        protocol = validation_dataloader.dataset.vox1_protocol.copy()
        for _, row in protocol.iterrows():
            scr0.append(np.mean(cos_sim_mat(emb_dict[row.enroll_ind][None,:],emb_dict[row.verify_ind][None,:])).item())
            # scr1.append(np.mean(cos_sim_mat(emb_dict[row.enroll_ind][1:],emb_dict[row.verify_ind][1:])).item())

        protocol['scr0'] = scr0
        # protocol['scr1'] = scr1
        # protocol['scr_avg'] = (protocol['scr0']+protocol['scr1'])/2

        # eer, _ = EER(protocol.label.values,protocol.score.values)
        try:
            new_row[f"{val_ds_name}-eer-fl"] = np.around(EER(protocol.label.values,protocol.scr0.values)[0]*100,3)
        except:
            new_row[f"{val_ds_name}-eer-fl"] = -1
        # new_row[f"vox1-eer-tta"] = np.around(EER(protocol.label.values,protocol.scr1.values)[0]*100,3)
        # new_row[f"vox1-eer-avg"] = np.around(EER(protocol.label.values,protocol.scr_avg.values)[0]*100,3)

        # csv_out_path = os.path.join(exp_dir, "val_log.csv")
        # log_df = log_df.append(new_row, ignore_index=True)
        # log_df.to_csv(csv_out_path, index=False)
        # return log_df
        return new_row

def cos_sim_mat(x0,x1):
    """
    x0 shape : (N,D)
    x1 shape : (M,D)
    D - dimensionality of embeddings
    """
    x0n = x0 / (np.linalg.norm(x0,axis=-1,keepdims=True)+1e-8)
    x1n = x1 / (np.linalg.norm(x1,axis=-1,keepdims=True)+1e-8)
    return np.sum(x0n[:,None,:]*x1n[None,:,:],axis=-1)


def set_bn_eval(m,flag=False):
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.track_running_stats = flag