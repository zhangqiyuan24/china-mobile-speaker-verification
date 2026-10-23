import csv, json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
base="/home/ligangrui/china_mobile/AI/10.17"
rows=list(csv.DictReader(open(os.path.join(base,"data/intrusive_metrics.csv"),encoding="utf-8")))
summary=json.load(open(os.path.join(base,"data/intrusive_metrics_summary.json"),encoding="utf-8"))
conds=["noisy","generic","tse"]; labels=["Noisy 0 dB","FullSubNet+","TSE (ours)"]
colors={"noisy":"#e69f00","generic":"#d55e00","tse":"#0072b2"}
fig, axes=plt.subplots(1,3,figsize=(13,4.0))
for ax,key,title,ylim in [(axes[0],"si_sdr_db","SI-SDR (dB)",None),(axes[1],"pesq_wb","PESQ-WB",(0.5,4.7)),(axes[2],"stoi","STOI",(0.0,1.0))]:
    vals=[summary[c][key]["mean"] for c in conds]; errs=[summary[c][key]["std"] for c in conds]
    bars=ax.bar(range(3),vals,yerr=errs,color=[colors[c] for c in conds],capsize=4)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=9)
    ax.set_title(title); ax.grid(axis="y",alpha=0.25)
    if ylim: ax.set_ylim(*ylim)
fig.suptitle("Intrusive metrics on 120 random test utterances (mean ± std)", y=0.99, fontsize=12)
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(os.path.join(base,"figs/intrusive_metrics.png"), bbox_inches="tight")
print("saved")
