#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型参数量普查 (9.3) — 只读统计, 不修改任何现有文件.
统计口径: state_dict 中所有浮点张量元素数 = 参数量(含 buffer 中的浮点统计量, 如 BN running stats);
整型 buffer (num_batches_tracked 等) 单独计, 不并入参数量."""
import os, sys, json, torch

ROOT = "/data/home/ligangrui/china_mobile"

MODELS = [
    # (标签, 相对路径, 备注)
    ("ECAPA-TDNN 官方预训练",      "code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa/pretrained/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k/ecapa-tdnn.ckpt", "经典基线, 3D-Speaker 官方权重"),
    ("ERes2Net-large 官方预训练",  "code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa/pretrained/speech_eres2net_large_sv_zh-cn_3dspeaker_16k/eres2net_large_model.ckpt", "强基线/主验收靶子"),
    ("ReDimNet2-B6 官方预训练",    "code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa/pretrained/redimnet2/b6-vb2+vox2+cnc2_v0-lm.pt", "跨框架 SOTA 基线"),
    ("FullSubNet+ 通用基座(8.18)", "AI/8.18/exp/spk_preserve/checkpoint_epoch10.tar", "TSE 之前的前端基座"),
    ("FullSubNet+ TSE 交付 r5",    "AI/8.27/exp/tse_v3_ns05duraug_r5/checkpoint_best.tar", "最终交付前端 (含 SpeakerFiLM)"),
    ("FullSubNet+ TSE v8d_sil",    "AI/8.27/exp/tse_v8_v8d_sil/checkpoint_best.tar", "claim 补救臂, 验证同构"),
    ("proto_robust (8.21)",        "AI/8.21/exp/proto_robust/checkpoints/checkpoint_best.tar", "ERes2Net 五条件鲁棒微调"),
    ("proto_robust_90k (8.24)",    "AI/8.24/exp/proto_robust_90k/checkpoints/checkpoint_best.tar", "90k 数据版"),
    ("matched_robust_90k (8.22)",  "AI/8.22/exp/matched_robust_90k/checkpoints/checkpoint_best.tar", "匹配域微调"),
    ("channel_aug_90k (8.24)",     "AI/8.24/exp/channel_aug_90k/checkpoints/checkpoint_best.tar", "通道增广微调"),
    ("soup5 权重平均 (8.24)",      "AI/8.24/exp/soup5/checkpoints/checkpoint_soup5.tar", "五模型汤, 冻结验证器"),
    ("soup_w50 (8.24)",            "AI/8.24/exp/soup_821_90k/checkpoints/checkpoint_soup_w50.tar", "8.21+90k 混合汤"),
    ("ECAPA robust_90k (8.26)",    "AI/8.26/exp/ecapa_robust_90k/checkpoints/checkpoint_best.tar", "ECAPA 镜像微调, 第二后端"),
    ("moe_lite (8.26, 已否决)",    "AI/8.26/exp/moe_lite/checkpoints/checkpoint_best.tar", "MoE-lite 支线"),
    ("feature_gate_90k (8.26,已否决)", "AI/8.26/exp/feature_gate_90k/checkpoints/checkpoint_best.tar", "GateNet 支线"),
    ("WeSep TSE avg_model (8.13)", "AI/8.13/third_party/models/avg_model.pt", "第三方 WeSep 探索, 已弃"),
]

FLOAT_DTYPES = {torch.float32, torch.float16, torch.float64, torch.bfloat16}

def unwrap(obj):
    """剥掉 checkpoint 外壳, 找到真正的 state_dict."""
    if not isinstance(obj, dict):
        return None
    for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
        if key in obj and isinstance(obj[key], dict):
            return unwrap(obj[key]) or obj[key]
    # 已经是平铺的 state_dict (key 是 str, value 是 tensor)
    if obj and all(isinstance(k, str) for k in list(obj)[:5]):
        sample = list(obj.values())[:5]
        if any(torch.is_tensor(v) for v in sample):
            return obj
    return None

def deep_find(obj, depth=0):
    """处理嵌套包裹 (如 {'speaker_model': {'state_dict': {...}}} 或 WeSep {'models': [sd]})."""
    sd = unwrap(obj)
    if sd is not None:
        return sd
    if isinstance(obj, dict) and depth < 4:
        for v in obj.values():
            r = deep_find(v, depth + 1)
            if r is not None:
                return r
    if isinstance(obj, (list, tuple)) and depth < 4:
        for v in obj:
            r = deep_find(v, depth + 1)
            if r is not None:
                return r
    return None

def count(path):
    p = os.path.join(ROOT, path)
    size_mb = os.path.getsize(p) / 1024 / 1024
    try:
        obj = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(p, map_location="cpu")
    sd = deep_find(obj)
    if sd is None:
        return {"path": path, "size_mb": round(size_mb, 1), "error": "no state_dict found"}
    nfloat = nint = 0
    nfloat_t = nint_t = 0
    for v in sd.values():
        if not torch.is_tensor(v):
            continue
        n = v.numel()
        if v.dtype in FLOAT_DTYPES:
            nfloat += n; nfloat_t += 1
        else:
            nint += n; nint_t += 1
    return {"path": path, "size_mb": round(size_mb, 1),
            "params": nfloat, "params_M": round(nfloat / 1e6, 2),
            "float_tensors": nfloat_t, "int_elems": nint}

results = []
for label, path, note in MODELS:
    try:
        r = count(path)
    except Exception as e:
        r = {"path": path, "error": repr(e)[:200]}
    r["label"] = label; r["note"] = note
    results.append(r)
    if "error" in r:
        print(f"[ERR] {label}: {r['error']}", file=sys.stderr)
    else:
        print(f"[OK] {label}: {r['params_M']}M / {r['size_mb']}MB", file=sys.stderr)

out = os.path.join(ROOT, "AI/9.3/params_census.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print("saved:", out, file=sys.stderr)
