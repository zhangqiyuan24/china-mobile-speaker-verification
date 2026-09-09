# -*- coding: utf-8 -*-
"""
E2-proto trainer：在 8.20 trainer 基础上新增
  1) prototype 初始化 CosineClassifier（本轮唯一方法改动）
  2) 训练前初始化诊断（prototype vs random vs 理论随机水平）
  3) 预留接口（默认关闭）：Option A classifier-only warmup / Option B 分组学习率

复用 8.20 的全部构件（模型构建/冻结/checkpoint 校验/梯度累积/AMP 修复），
不重复实现。CosineClassifier（speakerlab campplus）结构：
    self.weight = Parameter[out_neurons, input_dim]，无 bias，
    forward = F.linear(F.normalize(x), F.normalize(weight))   # 真 cosine logits
=> 直接把每类 prototype（单位向量）写进对应 weight 行即可。
"""

import os
import sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_V820 = os.path.join(os.path.dirname(_HERE), "8.20")
if _V820 not in sys.path:
    sys.path.insert(0, _V820)

from trainer_eres2net_robust import (   # noqa: F401  (8.20 全部复用)
    build_eres2net_large, load_pretrained_embedding, build_classifier,
    build_aam_margin, apply_freeze, freeze_bn_stats, freeze_report,
    verify_ckpt_roundtrip, JointModel, RobustTrainer,
)


# ============================================================
# prototype 初始化
# ============================================================

def load_prototypes(npz_path):
    """读 prepare_prototypes.py 产物 -> (protos[S,D] float32, spk_ids[list], used_utt_ids set)"""
    d = np.load(npz_path, allow_pickle=False)
    protos = d["prototypes"].astype(np.float32)
    spk_ids = [str(s) for s in d["spk_ids"].tolist()]
    used = set(str(u) for u in d["used_utt_ids"].tolist()) if "used_utt_ids" in d else set()
    assert protos.shape[0] == len(spk_ids), "prototypes 行数与 spk_ids 不一致"
    assert np.isfinite(protos).all(), "prototypes 含 NaN/Inf"
    return protos, spk_ids, used


def init_classifier_prototypes(classifier, protos, spk_ids, spk2id,
                               verbose=True, allow_missing=False):
    """
    把 prototype 写进 CosineClassifier.weight 的对应行：行 j（= 类 id j）=
    说话人 s 的 prototype，其中 spk2id[s] == j。

    允许 prototype 说话人是训练类别的**超集**（正常情况：prototype 按
    train_spks.txt 全集 9900 人生成，而 30k 训练清单均匀抽取只含 9892 人，
    多出的 prototype 行不用即可）。反向不允许：每个训练类别必须有 prototype
    （除非 allow_missing=True 的冒烟模式，缺的行保持 xavier 随机）。
    写完立刻回读校验（逐位一致 + 范数 ≈1 + finite）。
    """
    W = classifier.weight                     # Parameter[S_cls, D]，无 bias
    assert W.shape[1] == protos.shape[1], \
        f"维度不匹配: classifier weight {tuple(W.shape)} vs prototype {protos.shape}"
    proto_row = {s: i for i, s in enumerate(spk_ids)}
    missing = [s for s in spk2id if s not in proto_row]
    assert not missing or allow_missing, \
        f"{len(missing)} 个训练说话人没有 prototype（如 {missing[:3]}）；" \
        f"正式训练必须重跑 prepare_prototypes.py，冒烟可传 allow_missing=True"

    order = sorted(spk2id, key=lambda s: spk2id[s])      # 按类 id 0..S-1 排列
    place, rows = [], []
    for s in order:
        if s in proto_row:
            place.append(spk2id[s])
            rows.append(proto_row[s])

    with torch.no_grad():
        idx_t = torch.tensor(place, dtype=torch.long)
        t = torch.from_numpy(protos[np.array(rows, dtype=np.int64)])
        W[idx_t] = t                            # 只写有 prototype 的行；其余保持原初始化
        assert torch.equal(W[idx_t], t), "prototype 写入后回读不一致"
        norms = W.norm(dim=1)
        assert torch.isfinite(W).all(), "初始化后 weight 含 NaN/Inf"
    if verbose:
        extra = len(spk_ids) - len(spk2id)
        print(f"[proto-init] classifier weight shape = {tuple(W.shape)}")
        print(f"[proto-init] prototype matrix shape = {protos.shape}"
              + (f"（超集，{extra} 行多余 prototype 不使用）" if extra > 0 else ""))
        print(f"[proto-init] 行范数 mean/min/max = {norms.mean():.6f}/"
              f"{norms.min():.6f}/{norms.max():.6f}（应 ≈1；forward 内部还会再归一化）")
        print(f"[proto-init] spk2id 行映射已逐位回读校验 ✓（{len(place)}/{W.shape[0]} 类用 prototype 初始化）")
        if missing:
            print(f"[proto-init] 警告：{len(missing)} 类无 prototype，保持随机初始化（冒烟模式）")
    return classifier


# ============================================================
# 初始化诊断（训练真正开始前）
# ============================================================

def init_diagnostics(joint, margin_loss, feats, labels, device,
                     num_classes, tag="prototype", batch_size=16):
    """
    用（尚未训练的）joint 模型在 clean 探针批上直接测分类能力。返回指标 dict。
    feats [N,T,80]（CPU tensor），labels [N]。forward 按 batch_size 分块（共享卡
    显存紧张，一次塞 256 条 3s 会 OOM）；指标在 CPU 上合并计算。
    """
    margin_loss.eval()
    with torch.no_grad():
        cos_list, lab_list = [], []
        for i in range(0, len(labels), batch_size):
            f = feats[i:i + batch_size].to(device)
            cos_list.append(joint(f).float().cpu())       # [n, S]
            lab_list.append(labels[i:i + batch_size])
        cosine = torch.cat(cos_list)                      # [N, S] CPU
        labels_d = torch.cat(lab_list)
        loss = margin_loss(cosine, labels_d).item()       # ArcMarginLoss 无参数，CPU 可算
        top1 = (cosine.argmax(dim=1) == labels_d).float().mean().item()
        tgt = cosine.gather(1, labels_d[:, None]).squeeze(1)
        keep = cosine.clone()
        keep[torch.arange(len(labels_d)), labels_d] = float("-inf")
        max_non = keep.max(dim=1)[0]
        res = {"top1": top1, "aam_loss": loss,
               "mean_target_cos": tgt.mean().item(),
               "mean_max_non_target_cos": max_non.mean().item()}
    rand_top1 = 1.0 / num_classes
    rand_loss = float(np.log(num_classes) + 32.0 * 0.2)   # ln(S) + s*m 理论值
    print(f"\n[{tag} init] clean 探针 {len(labels)} 条 / {num_classes} 类")
    print(f"  clean top1 acc          = {res['top1']*100:.2f}%   "
          f"(随机理论 ≈ {rand_top1*100:.3f}%)")
    print(f"  clean aam loss          = {res['aam_loss']:.4f}   "
          f"(随机理论 ≈ {rand_loss:.2f})")
    print(f"  mean target cosine      = {res['mean_target_cos']:.4f}")
    print(f"  mean max non-target cos = {res['mean_max_non_target_cos']:.4f}")
    return res


# ============================================================
# ProtoTrainer：8.20 RobustTrainer + 双参数组 + warmup 预留接口
# ============================================================

class ProtoTrainer(RobustTrainer):
    """
    训练语义与 8.20 完全一致（梯度累积/AMP/flush），仅把 optimizer 拆成
    body / head 两个参数组，以支持（默认关闭的）：
      Option A: head_warmup_steps > 0 时，前 N 个 optimizer step body 组 lr=0
                （只更新 classifier；lr 仍全部 <=2e-5，符合方案约束）
      Option B: head_lr > 0 时 classifier 用独立学习率（突破统一 lr 约束，慎用）
    两者都为 0 时与 8.20 的单一参数组 AdamW 数值等价。
    """

    def __init__(self, net, margin_loss, frozen_mods, lr=2e-5, weight_decay=5e-5,
                 device="cuda:0", amp=False, accum=1,
                 head_lr=None, head_warmup_steps=0, verbose=True):
        super().__init__(net, margin_loss, frozen_mods, lr=lr,
                         weight_decay=weight_decay, device=device, amp=amp, accum=accum)
        joint = self.net.unwrap()
        body_params = [p for p in joint.embedding_model.parameters() if p.requires_grad]
        head_params = list(joint.classifier.parameters())
        self.body_group = {"params": body_params, "lr": lr}
        self.head_group = {"params": head_params, "lr": head_lr if head_lr else lr}
        self.base_body_lr, self.base_head_lr = lr, self.head_group["lr"]
        self.head_warmup_steps = head_warmup_steps
        self.opt_steps = 0
        self.verbose = verbose
        # 覆盖父类单一参数组 optimizer（其余状态 scaler/micro 全部沿用）
        self.optimizer = torch.optim.AdamW([self.body_group, self.head_group],
                                           lr=lr, weight_decay=weight_decay)
        self._apply_phase(announce=True)

    def _apply_phase(self, announce=False):
        if self.head_warmup_steps and self.opt_steps < self.head_warmup_steps:
            self.body_group["lr"] = 0.0
            if announce and self.verbose:
                print(f"[phase] Option A warmup 开启：前 {self.head_warmup_steps} 个 "
                      f"optimizer step 只更新 classifier（body lr=0，lr 均未超上限）")
        else:
            self.body_group["lr"] = self.base_body_lr

    def _opt_step(self):
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.opt_steps += 1
        if self.head_warmup_steps and self.opt_steps == self.head_warmup_steps:
            self._apply_phase()
            if self.verbose:
                print(f"[phase] warmup 结束（{self.opt_steps} step 起 body 恢复 lr="
                      f"{self.base_body_lr:g}）")

    def train_step(self, feats, labels):
        """与 8.20 相同（含 AMP dtype 修复），仅 optimizer step 抽成 _opt_step 以挂 warmup。"""
        feats = feats.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.amp):
            cosine = self.net(feats)              # CosineClassifier 输出 cos logits
        cosine = cosine.float()                   # ArcMarginLoss sqrt 会升 fp32，统一回 fp32
        loss = self.margin_loss(cosine, labels)   # AAM-softmax（margin/scale 固定）
        self.micro += 1
        self.scaler.scale(loss / self.accum).backward()
        if self.micro % self.accum == 0:
            self._opt_step()
        with torch.no_grad():
            acc = (cosine.argmax(dim=1) == labels).float().mean().item()
        return loss.item(), acc

    def flush(self):
        """epoch 结束把不足一组的残余梯度用掉（与 8.20 相同语义）。"""
        if self.micro % self.accum == 0:
            return
        self._opt_step()
        self.micro = 0
