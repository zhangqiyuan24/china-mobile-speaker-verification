import os, csv, json
import numpy as np, soundfile as sf
from scipy.signal import correlate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BASE="/home/ligangrui/china_mobile"; SV=os.path.join(BASE,"code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa"); OUT=os.path.join(BASE,"AI/10.17"); DATA=os.path.join(OUT,"data"); FIGS=os.path.join(OUT,"figs")
def map_wavs(root):
    m={}
    for dp,_,fs in os.walk(root):
        for fn in fs:
            if fn.endswith(".wav"): m[os.path.splitext(fn)[0]]=os.path.join(dp,fn)
    return m
def load(p):
    y,sr=sf.read(p,dtype="float32",always_2d=False)
    if y.ndim>1: y=y.mean(1)
    return y.astype(np.float32)
def align(est,ref,max_lag=800):
    n=min(len(est),len(ref)); est,ref=est[:n],ref[:n]
    c=correlate(est,ref,mode="full",method="fft"); lag=int(np.argmax(np.abs(c))-(n-1))
    if lag>0: est=est[lag:]
    elif lag<0: est=np.concatenate([np.zeros(-lag,dtype=est.dtype),est])
    n=min(len(est),len(ref)); return est[:n],ref[:n]
def active_mask(ref,win=512,hop=256,top=0.6):
    if len(ref)<win: ref=np.pad(ref,(0,win-len(ref)))
    n=1+(len(ref)-win)//hop; idx=np.arange(win)[None,:]+hop*np.arange(n)[:,None]
    e=np.mean(ref[idx]**2,axis=1)+1e-12; thr=np.percentile(e,100*(1-top)); af=idx[e>=thr]
    m=np.zeros(len(ref),bool); m[af.ravel()]=True; return m
def global_snr(est,ref,mode="matched"):
    est,ref=align(est,ref); m=active_mask(ref)
    if mode=="matched":
        rs=np.sqrt(np.mean(ref[m]**2)+1e-12); re=np.sqrt(np.mean(est[m]**2)+1e-12); est=est*(rs/re)
    return 10*np.log10((np.sum(ref**2)+1e-12)/(np.sum((est-ref)**2)+1e-12))
def si_sdr(est,ref):
    est,ref=align(est,ref); est=est-est.mean(); ref=ref-ref.mean()
    a=float(np.dot(est,ref)/(np.dot(ref,ref)+1e-10)); t=a*ref; e=est-t
    return 10*np.log10((np.sum(t**2)+1e-10)/(np.sum(e**2)+1e-10))
ids=sorted({r["test_id"] for r in csv.DictReader(open(os.path.join(DATA,"intrusive_metrics.csv"),encoding="utf-8"))})
CLEAN=map_wavs(os.path.join(SV,"data/data_for_test/clean/3s"))
paths={
 "0db":{"noisy":map_wavs(os.path.join(SV,"data/data_for_test/noise_0db/3s")),
        "generic":map_wavs(os.path.join(SV,"data/data_for_test/denoised/noise_0db/enhanced_0194")),
        "tse":map_wavs(os.path.join(BASE,"AI/8.27/exp/tse_v3_ns05duraug_r5/enhanced_official/noise_0db"))},
 "m5db":{"noisy":map_wavs(os.path.join(SV,"data/data_for_test/noise_m5db/3s")),
         "generic":map_wavs(os.path.join(SV,"data/data_for_test/denoised/noise_m5db/enhanced_0194")),
         "tse":map_wavs(os.path.join(BASE,"AI/8.27/exp/tse_v3_ns05duraug_r5/enhanced_official/noise_m5db"))},
}
rows=[]
for snr_set in ["0db","m5db"]:
    for u in ids:
        if u not in CLEAN or any(u not in paths[snr_set][c] for c in paths[snr_set]): continue
        s=load(CLEAN[u])
        for c in ["noisy","generic","tse"]:
            e=load(paths[snr_set][c][u])
            rows.append({"snr_set":snr_set,"test_id":u,"condition":c,
                         "snr_global_raw_db":global_snr(e,s,"raw"),
                         "snr_global_matched_db":global_snr(e,s,"matched"),
                         "si_sdr_db":si_sdr(e,s)})
# paired deltas vs noisy
by={(r["snr_set"],r["test_id"],r["condition"]):r for r in rows}
for r in rows:
    n=by[(r["snr_set"],r["test_id"],"noisy")]
    r["delta_snr_global_db"]=r["snr_global_matched_db"]-n["snr_global_matched_db"]
    r["si_sdri_db"]=r["si_sdr_db"]-n["si_sdr_db"]
out=os.path.join(DATA,"global_snr_denoised.csv")
fields=["snr_set","test_id","condition","snr_global_raw_db","snr_global_matched_db","delta_snr_global_db","si_sdr_db","si_sdri_db"]
with open(out,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
summary={}
for snr_set in ["0db","m5db"]:
    summary[snr_set]={}
    for c in ["noisy","generic","tse"]:
        sub=[r for r in rows if r["snr_set"]==snr_set and r["condition"]==c]
        summary[snr_set][c]={k:{"mean":float(np.mean([r[k] for r in sub])),"median":float(np.median([r[k] for r in sub])),"std":float(np.std([r[k] for r in sub]))}
                             for k in ["snr_global_raw_db","snr_global_matched_db","delta_snr_global_db","si_sdr_db","si_sdri_db"]}
        summary[snr_set][c]["n"]=len(sub)
with open(os.path.join(DATA,"global_snr_summary.json"),"w",encoding="utf-8") as f:
    json.dump(summary,f,ensure_ascii=False,indent=2)
for snr_set in ["0db","m5db"]:
    print("===",snr_set)
    for c in ["noisy","generic","tse"]:
        s=summary[snr_set][c]
        print("  %-8s globalSNR(matched) %6.2f dB (raw %7.2f)  dSNR %+6.2f  SI-SDR %6.2f (i%+6.2f)"%(c,s["snr_global_matched_db"]["mean"],s["snr_global_raw_db"]["mean"],s["delta_snr_global_db"]["mean"],s["si_sdr_db"]["mean"],s["si_sdri_db"]["mean"]))
# figure
fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),sharey=True)
colors={"noisy":"#e69f00","generic":"#d55e00","tse":"#0072b2"}; labels={"noisy":"Noisy","generic":"FullSubNet+","tse":"TSE"}
for ax,ss in zip(axes,["0db","m5db"]):
    vals=[summary[ss][c]["snr_global_matched_db"]["mean"] for c in ["noisy","generic","tse"]]
    errs=[summary[ss][c]["snr_global_matched_db"]["std"] for c in ["noisy","generic","tse"]]
    bars=ax.bar(range(3),vals,yerr=errs,color=[colors[c] for c in ["noisy","generic","tse"]],capsize=4)
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v,f"{v:.2f}",ha="center",va="bottom",fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels([labels[c] for c in ["noisy","generic","tse"]],rotation=10)
    ax.set_title("0 dB set" if ss=="0db" else "-5 dB set"); ax.grid(axis="y",alpha=0.25)
axes[0].set_ylabel("global SNR after level matching (dB)")
fig.suptitle("Global SNR on denoised data (120 test utterances, mean ± std)",y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(FIGS,"global_snr.png"),bbox_inches="tight")
print("saved",os.path.join(FIGS,"global_snr.png"))
