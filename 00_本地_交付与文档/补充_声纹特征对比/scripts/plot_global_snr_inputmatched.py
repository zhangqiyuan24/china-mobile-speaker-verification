import csv, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
base="/home/ligangrui/china_mobile/AI/10.17"
rows=list(csv.DictReader(open(os.path.join(base,"data/global_snr_inputmatched.csv"),encoding="utf-8")))
for r in rows: r["snr_global_db"]=float(r["snr_global_db"])
conds=["noisy","generic","tse"]; labels=["Noisy","FullSubNet+","TSE"]; colors={"noisy":"#e69f00","generic":"#d55e00","tse":"#0072b2"}
fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),sharey=True)
for ax,ss,title in [(axes[0],"0db","0 dB set"),(axes[1],"m5db","-5 dB set")]:
    vals=[]; errs=[]
    for c in conds:
        a=np.array([float(r["snr_global_db"]) for r in rows if r["snr_set"]==ss and r["condition"]==c])
        vals.append(a.mean()); errs.append(a.std())
    bars=ax.bar(range(3),vals,yerr=errs,color=[colors[c] for c in conds],capsize=4)
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v,f"{v:.2f}",ha="center",va="bottom",fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels,rotation=10); ax.set_title(title); ax.grid(axis="y",alpha=0.25)
axes[0].set_ylabel("global SNR matched to input RMS (dB)")
fig.suptitle("Global SNR on denoised data (120 test utterances, mean ± std)",y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(base,"figs/global_snr_inputmatched.png"),bbox_inches="tight")
print("saved")
