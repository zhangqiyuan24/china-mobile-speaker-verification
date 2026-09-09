import os, sys
import torch
_H = "/data/home/ligangrui/china_mobile/AI/8.26"
for _p in ("/data/home/ligangrui/china_mobile/AI/8.20", "/data/home/ligangrui/china_mobile/AI/8.21", "/data/home/ligangrui/china_mobile/AI/8.24", _H):
    sys.path.insert(0, _p)
from feature_gate import GateNet, GatedEmbedding
from trainer_eres2net_robust import (build_eres2net_large, load_pretrained_embedding,
    build_classifier, build_aam_margin, apply_freeze, JointModel)
from trainer_eres2net_robust_proto import load_prototypes, init_classifier_prototypes
import toml
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
cfg = toml.load(_H + "/config_feature_gate.toml")
sv = cfg["paths"]["sv_ecapa"]
CK = sv + "/pretrained/speech_eres2net_large_sv_zh-cn_3dspeaker_16k/eres2net_large_model.ckpt"
sys.path.insert(0, os.path.join(sv, "..", "..", ".."))
protos, spk_ids, _ = load_prototypes(cfg["paths"]["proto_npz"])
spk2id = {s: i for i, s in enumerate(spk_ids)}
def mem(t):
    torch.cuda.synchronize()
    print("[mem] %-28s alloc=%6.2f reserved=%6.2f" % (t, torch.cuda.memory_allocated()/2**30, torch.cuda.memory_reserved()/2**30), flush=True)
B, T = 18, 300
fb = torch.randn(B, T, 80, device="cuda")
lb = torch.randint(0, 9900, (B,), device="cuda")
ml = build_aam_margin(32.0, 0.2).to("cuda")
def build():
    enc = build_eres2net_large(); load_pretrained_embedding(enc, CK); apply_freeze(enc)
    cls = build_classifier(512, 9900); init_classifier_prototypes(cls, protos, spk_ids, spk2id)
    return enc, cls
def run(tag, embd, cls, detach_gate=False, frozen_gate=False):
    joint = JointModel(embd, cls).to("cuda")
    if frozen_gate:
        for p in embd.gate.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW([p for p in joint.parameters() if p.requires_grad], lr=2e-5)
    try:
        if detach_gate:
            with torch.no_grad(): y = embd.gate(fb)
            cos = embd.enc(cls is None and y or y)  # placeholder
        for it in range(2):
            cos = joint(fb)
            loss = ml(cos, lb); loss.backward(); opt.step(); opt.zero_grad()
        mem(tag + " it1 step后")
    except RuntimeError as e:
        print("[OOM]", tag, str(e)[:80], flush=True)
    del joint, opt; torch.cuda.empty_cache(); mem(tag + " 清理后")
# B0: 恒等(无门)
enc, cls = build(); run("B0-无门JointModel", enc.to("cuda"), cls.to("cuda"))
enc.cpu(); cls.cpu(); del enc, cls; torch.cuda.empty_cache()
# B1: 真门+冻结+detach(门输出不进图)
enc, cls = build(); g = GateNet(n_mels=80)
class DetachGate(torch.nn.Module):
    def __init__(s, g, e): super().__init__(); s.gate = g; s.enc = e
    def forward(s, x):
        with torch.no_grad(): y = s.gate(x)
        return s.enc(y.detach())
run("B1-冻结门+detach", DetachGate(g, enc).to("cuda"), cls.to("cuda"))
enc.cpu(); cls.cpu(); del enc, cls; torch.cuda.empty_cache()
# B2: 完整图(与训练同构)
enc, cls = build(); g2 = GateNet(n_mels=80)
run("B2-完整图门", GatedEmbedding(g2, enc).to("cuda"), cls.to("cuda"))
