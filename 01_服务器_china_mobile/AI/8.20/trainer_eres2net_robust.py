# -*- coding: utf-8 -*-
"""
E2 trainer：ERes2Net-large 多条件鲁棒微调

要点（对应方案第 3/4 节）：
  - 冻结范围 = conv1/bn1/layer1/layer2/layer1_downsample/fuse_mode12（顶层模块名精确匹配，
    不用子串匹配 —— 修复旧代码 substring freeze 的隐患），其余全部训练
    （layer3/layer4/layer2_downsample/layer3_downsample/fuse_mode123/fuse_mode1234/
     pool/seg_1/seg_2 + 分类器）。
  - 冻结层同时冻结 BN running stats（train 阶段强制 eval）：防止带噪数据的统计量
    漂移污染冻结层、造成 clean 退化。
  - 损失 = AAM-softmax（ArcMarginLoss，margin/scale 固定，无 scheduler）。
  - checkpoint 校验：加载后 strict 比对 missing/unexpected keys 并断言为空。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 顶层模块冻结表（与 ERes2Net.forward 的数据流一一对应）----
# 冻结：stem + layer1 + layer2 及其专属融合 fuse_mode12（含只喂它的 layer1_downsample）
FROZEN_TOP = {"conv1", "bn1", "layer1", "layer1_downsample", "layer2", "fuse_mode12"}
# 训练：layer3 及以上 + 三个“喂给可训练融合块”的适配器之一 layer2_downsample
TRAINABLE_TOP = {"layer3", "layer4", "layer2_downsample", "layer3_downsample",
                 "fuse_mode123", "fuse_mode1234", "pool",
                 "seg_1", "seg_bn_1", "seg_2"}


def build_eres2net_large(embedding_size=512, m_channels=64):
    from speakerlab.models.eres2net.ERes2Net import ERes2Net
    return ERes2Net(feat_dim=80, embedding_size=embedding_size, m_channels=m_channels)


def load_pretrained_embedding(model, ckpt_path, device="cpu"):
    """加载官方 pretrained ERes2Net-large；strict 校验 missing/unexpected，断言为空。"""
    ckpt = torch.load(ckpt_path, map_location=device)
    for key in ("embedding_model", "model", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt and hasattr(ckpt[key], "items"):
            sd = ckpt[key]
            break
    else:
        sd = ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load_pretrained] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing 前10:", missing[:10])
    if unexpected:
        print("  unexpected 前10:", unexpected[:10])
    assert not missing and not unexpected, "pretrained 权重与模型结构不匹配，禁止继续"
    return model


def build_classifier(embedding_size, num_classes):
    from speakerlab.models.campplus.classifier import CosineClassifier
    return CosineClassifier(input_dim=embedding_size, out_neurons=num_classes)


def build_aam_margin(scale, margin):
    from speakerlab.loss.margin_loss import ArcMarginLoss
    return ArcMarginLoss(scale=scale, margin=margin, easy_margin=False)


def apply_freeze(embedding_model, verbose=True):
    """
    按顶层模块名冻结。返回 (frozen_module_names, trainable_module_names)。
    冻结模块 requires_grad=False 且每次 train() 后需调用 freeze_bn_stats()。
    """
    frozen_mods, trainable_mods = [], []
    for name, child in embedding_model.named_children():
        if name in FROZEN_TOP:
            for p in child.parameters():
                p.requires_grad = False
            frozen_mods.append(name)
        elif name in TRAINABLE_TOP:
            trainable_mods.append(name)
        else:
            raise ValueError(f"未知顶层模块 {name!r}，请人工确认冻结表后更新 FROZEN_TOP/TRAINABLE_TOP")
    if verbose:
        print(f"[freeze] 冻结模块: {sorted(frozen_mods)}")
        print(f"[freeze] 训练模块: {sorted(trainable_mods)}")
    return frozen_mods, trainable_mods


def freeze_bn_stats(embedding_model, frozen_mods):
    """把冻结模块切回 eval()，冻结其 BN running stats（在每个 epoch 开始时调用）。"""
    for name, child in embedding_model.named_children():
        if name in frozen_mods:
            child.eval()


def freeze_report(embedding_model, classifier):
    """打印可训练/冻结参数量，并做守门断言：冻结参数梯度必须为 False。"""
    n_train = sum(p.numel() for p in embedding_model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in embedding_model.parameters() if not p.requires_grad)
    n_cls = sum(p.numel() for p in classifier.parameters())
    total_emb = n_train + n_frozen
    print(f"[freeze_report] embedding: 可训练 {n_train/1e6:.2f}M / 冻结 {n_frozen/1e6:.2f}M "
          f"(共 {total_emb/1e6:.2f}M, 可训练占比 {100.0*n_train/total_emb:.1f}%)")
    print(f"[freeze_report] classifier: {n_cls/1e6:.2f}M (全训练)")
    for name, p in embedding_model.named_parameters():
        top = name.split(".")[0]
        want_frozen = top in FROZEN_TOP
        assert p.requires_grad != want_frozen, f"冻结状态错误: {name} requires_grad={p.requires_grad}"
    return {"trainable_emb": n_train, "frozen_emb": n_frozen, "classifier": n_cls}


def verify_ckpt_roundtrip(embedding_model, path):
    """保存后立即重载校验（strict），保证 checkpoint 一定可恢复。"""
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["embedding_model"] if isinstance(ckpt, dict) and "embedding_model" in ckpt else ckpt
    m = build_eres2net_large()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"checkpoint 校验失败: missing={missing[:5]} unexpected={unexpected[:5]}"


class JointModel(nn.Module):
    """embedding + cosine 分类器打包，便于整体套 DDP。"""

    def __init__(self, embedding_model, classifier):
        super().__init__()
        self.embedding_model = embedding_model
        self.classifier = classifier

    def forward(self, feats):
        return self.classifier(self.embedding_model(feats))

    def unwrap(self):
        m = self
        while hasattr(m, "module"):
            m = m.module
        return m


class RobustTrainer:
    """
    net: JointModel 或 DDP(JointModel)。
    """

    def __init__(self, net, margin_loss, frozen_mods,
                 lr=2e-5, weight_decay=5e-5, device="cuda:0", amp=False, accum=1):
        self.net = net.to(device)
        self.margin_loss = margin_loss.to(device)
        self.frozen_mods = frozen_mods
        self.device = device
        self.amp = amp
        self.accum = accum        # 梯度累积 micro-batch 数（有效 batch = batch*accum）
        self.micro = 0
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)
        joint = self.net.unwrap()
        params = [p for p in joint.embedding_model.parameters() if p.requires_grad] + \
                 list(joint.classifier.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.lr = lr

    @property
    def joint(self):
        return self.net.unwrap()

    def train_mode(self):
        self.net.train()
        freeze_bn_stats(self.joint.embedding_model, self.frozen_mods)  # 冻结层 BN stats 不动

    def train_step(self, feats, labels):
        """feats [B,T,80] fbank; labels [B] int64. 返回 (loss, acc)。
        梯度累积：loss/accum 反传，每 accum 个 micro-batch 才 step 一次 —— 共享显卡
        只塞得下小 batch 时用它对齐配置的 batch_size=96。AdamW 每次 step 的参数
        位移 ≈ lr 与 batch 无关，batch 12 会让同轮数据产生 8 倍步数、8 倍漂移
        （sanity 实测：clean 1.25%→5%，noise_sum 18.25→27.75）。"""
        feats = feats.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.amp):
            cosine = self.net(feats)              # CosineClassifier 输出 cos logits
        # ArcMarginLoss 内部 sqrt 会被 autocast 提升为 fp32，与 fp16 的 cosine 混算
        # 报 "expected scalar type float but found c10::Half"，故损失统一回 fp32
        cosine = cosine.float()
        loss = self.margin_loss(cosine, labels)   # AAM-softmax（margin/scale 固定）
        self.micro += 1
        self.scaler.scale(loss / self.accum).backward()
        if self.micro % self.accum == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            acc = (cosine.argmax(dim=1) == labels).float().mean().item()
        return loss.item(), acc

    def flush(self):
        """epoch 结束时把不足一组的残余梯度用掉，避免跨 epoch 泄漏。"""
        if self.micro % self.accum == 0:
            return
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.micro = 0

    def save_checkpoint(self, path, epoch, val_metrics=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "embedding_model": self.joint.embedding_model.state_dict(),
            "classifier": self.joint.classifier.state_dict(),
            "epoch": epoch,
            "lr": self.lr,
            "val_metrics": val_metrics or {},
        }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)                     # 原子写，避免半截 checkpoint
        verify_ckpt_roundtrip(self.joint.embedding_model, path)
