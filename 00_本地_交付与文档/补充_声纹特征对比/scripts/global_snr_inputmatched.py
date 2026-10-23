import os, csv, json
import numpy as np, soundfile as sf
from scipy.signal import correlate
BASE="/home/ligangrui/china_mobile"; SV=os.path.join(BASE,"code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa"); DATA=os.path.join(BASE,"AI/10.17/data")
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
def snr_matched_to_noisy(est,ref,noisy):
    est,ref=align(est,ref); noisy,_=align(noisy,ref)
    n=min(len(est),len(ref),len(noisy)); est,ref,noisy=est[:n],ref[:n],noisy[:n]
    beta=float(np.sqrt(np.mean(noisy**2)+1e-12)/np.sqrt(np.mean(est**2)+1e-12))
    e2=beta*est
    return 10*np.log10((np.sum(ref**2)+1e-12)/(np.sum((e2-ref)**2)+1e-12)), beta
ids=sorted({r["test_id"] for r in csv.DictReader(open(os.path.join(DATA,"intrusive_metrics.csv"),encoding="utf-8"))})
CLEAN=map_wavs(os.path.join(SV,"data/data_for_test/clean/3s"))
paths={"0db":{"noisy":map_wavs(os.path.join(SV,"data/data_for_test/noise_0db/3s")),
              "generic":map_wavs(os.path.join(SV,"data/data_for_test/denoised/noise_0db/enhanced_0194")),
              "tse":map_wavs(os.path.join(BASE,"AI/8.27/exp/tse_v3_ns05duraug_r5/enhanced_official/noise_0db"))},
       "m5db":{"noisy":map_wavs(os.path.join(SV,"data/data_for_test/noise_m5db/3s")),
               "generic":map_wavs(os.path.join(SV,"data/data_for_test/denoised/noise_m5db/enhanced_0194")),
               "tse":map_wavs(os.path.join(BASE,"AI/8.27/exp/tse_v3_ns05duraug_r5/enhanced_official/noise_m5db"))}}
rows=[]
for ss in ["0db","m5db"]:
    for u in ids:
        if u not in CLEAN or any(u not in paths[ss][c] for c in paths[ss]): continue
        s=load(CLEAN[u]); nz=load(paths[ss]["noisy"][u])
        for c in ["noisy","generic","tse"]:
            e=load(paths[ss][c][u]); snr,beta=snr_matched_to_noisy(e,s,nz)
            rows.append({"snr_set":ss,"test_id":u,"condition":c,"snr_global_db":snr,"beta":beta})
by={(r["snr_set"],r["test_id"],r["condition"]):r for r in rows}
for r in rows:
    r["delta_snr_db"]=r["snr_global_db"]-by[(r["snr_set"],r["test_id"],"noisy")]["snr_global_db"]
out=os.path.join(DATA,"global_snr_inputmatched.csv")
with open(out,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["snr_set","test_id","condition","snr_global_db","delta_snr_db","beta"]); w.writeheader(); w.writerows(rows)
summary={}
for ss in ["0db","m5db"]:
    summary[ss]={}
    for c in ["noisy","generic","tse"]:
        sub=[r for r in rows if r["snr_set"]==ss and r["condition"]==c]
        summary[ss][c]={k:{"mean":float(np.mean([r[k] for r in sub])),"median":float(np.median([r[k] for r in sub])),"std":float(np.std([r[k] for r in sub]))} for k in ["snr_global_db","delta_snr_db","beta"]}
        summary[ss][c]["n"]=len(sub)
with open(os.path.join(DATA,"global_snr_inputmatched_summary.json"),"w",encoding="utf-8") as f:
    json.dump(summary,f,ensure_ascii=False,indent=2)
for ss in ["0db","m5db"]:
    print("===",ss)
    for c in ["noisy","generic","tse"]:
        s=summary[ss][c]
        print("  %-8s globalSNR %6.2f dB  dSNR %+6.2f  beta %.3f"%(c,s["snr_global_db"]["mean"],s["delta_snr_db"]["mean"],s["beta"]["mean"]))
