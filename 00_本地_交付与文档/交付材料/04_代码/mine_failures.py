#!/usr/bin/env python
# mine_failures.py — #21 失败案例挖掘（8.31，纯 CPU）
# 在 EER 工作点上挖：漏识（target 低分）与误受（nontarget 高分），
# 聚合到说话人 / 说话人对 / 测试设备三个维度，另抽最惨说话人的具体案例句。
import json, os
from collections import defaultdict

AI = "/data/home/ligangrui/china_mobile/AI"
OUT = f"{AI}/8.30/failures"
os.makedirs(OUT, exist_ok=True)
PB = f"{AI}/8.27/exp/official_pipeline_3backends"
FE = f"{AI}/8.28/final_eval_r5"
SC = f"{AI}/8.28/scenes_r5/eres2net"
summ = {r["name"]: r for r in json.load(open(f"{AI}/8.30/roc_det/summary.json"))}

CELLS = {  # name -> score 路径
    "main_eres_0db_raw": f"{PB}/eres2net/raw_noise_0db/trials_cross_device.score",
    "main_eres_0db_fe":  f"{FE}/r5_eres_noise_0db_3s/trials_cross_device.score",
    "main_eres_m5db_raw":f"{PB}/eres2net/raw_noise_m5db/trials_cross_device.score",
    "main_eres_m5db_fe": f"{FE}/r5_eres_noise_m5db_3s/trials_cross_device.score",
    "scene_ov2cn_fe":    f"{SC}/fe_overlap2_cn_3s/trials_cross_device.score",
    "scene_ov3cn_fe":    f"{SC}/fe_overlap3_cn_3s/trials_cross_device.score",
}

def parse(utt):
    p = utt.split("_")          # 3D SPK 06154 001 Device03 Distance00 Dialect00 3s
    return p[2], p[4]

def mine(name, path):
    thr = summ[name]["eer_thr"]
    fn_spk = defaultdict(lambda: [0, 0])       # spk -> [FN数, target总数]
    fa_pair = defaultdict(int)                 # "A|B" -> FA 数（A注册 B测试）
    fn_dev = defaultdict(lambda: [0, 0])       # 测试设备 -> [FN, 总]
    fa_dev = defaultdict(int)
    worst = defaultdict(list)                  # spk -> 最低分 target 案例列表(维持小顶)
    n_fa = n_fn = 0
    for line in open(path):
        e, t, lab, s = line.split()
        s = float(s); es, _ = parse(e); ts, td = parse(t)
        if lab == "target":
            fn_spk[ts][1] += 1; fn_dev[td][1] += 1
            if s < thr:
                n_fn += 1; fn_spk[ts][0] += 1; fn_dev[td][0] += 1
                lst = worst[ts]
                lst.append((s, e, t))
                if len(lst) > 5: lst.remove(max(lst))
        else:
            if s >= thr:
                n_fa += 1; fa_pair[f"{es}|{ts}"] += 1; fa_dev[td] += 1
    tot_tar = sum(v[1] for v in fn_spk.values())
    top_fn = sorted(((v[0]/v[1], k, v[0], v[1]) for k, v in fn_spk.items() if v[1] >= 40), reverse=True)[:10]
    top_fa = sorted(fa_pair.items(), key=lambda kv: -kv[1])[:10]
    dev_fn = {d: round(v[0]/v[1]*100, 2) for d, v in sorted(fn_dev.items())}
    return dict(name=name, thr=thr, n_fn=n_fn, n_fa=n_fa,
                fn_rate=round(n_fn/tot_tar*100, 2),
                top_fn_speakers=[dict(spk=k, fn=v, tot=t, rate=round(r*100,1)) for r, k, v, t in top_fn],
                top_fa_pairs=[dict(pair=k, fa=v) for k, v in top_fa],
                fn_rate_by_test_device=dev_fn,
                worst_cases={k: sorted(v)[:3] for k, v in sorted(worst.items(), key=lambda kv: kv[1][0][0])[:5]})

res = []
for name, path in CELLS.items():
    if not os.path.exists(path):
        print(f"[miss] {name}"); continue
    r = mine(name, path); res.append(r)
    print(f"[ok] {name}: FN率={r['fn_rate']}% 最惨说话人={r['top_fn_speakers'][0]['spk']}({r['top_fn_speakers'][0]['rate']}%) 头号混淆对={r['top_fa_pairs'][0]['pair']}({r['top_fa_pairs'][0]['fa']})")
json.dump(res, open(f"{OUT}/failures.json", "w"), ensure_ascii=False, indent=1)
print(f"[mine-failures done] {OUT}/failures.json")
