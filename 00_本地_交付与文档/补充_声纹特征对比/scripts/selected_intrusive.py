import os, json, csv
import numpy as np, soundfile as sf
from scipy.signal import correlate
from pesq import pesq
from pystoi import stoi
BASE="/home/ligangrui/china_mobile"; SV=os.path.join(BASE,"code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa"); DATA=os.path.join(BASE,"AI/10.17/data")
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
def si_sdr(est,ref):
    est,ref=align(est,ref); est=est-est.mean(); ref=ref-ref.mean()
    a=float(np.dot(est,ref)/(np.dot(ref,ref)+1e-10)); t=a*ref; e=est-t
    return 10*np.log10((np.sum(t**2)+1e-10)/(np.sum(e**2)+1e-10))
def frame_energy(x,win=512,hop=256):
    if len(x)<win: x=np.pad(x,(0,win-len(x)))
    n=1+(len(x)-win)//hop; idx=np.arange(win)[None,:]+hop*np.arange(n)[:,None]
    return np.mean(x[idx]**2,axis=1)+1e-12
def seg_snr(est,ref):
    est,ref=align(est,ref); ec=frame_energy(ref); ee=frame_energy(est)
    n=min(len(ec),len(ee)); ec,ee=ec[:n],ee[:n]
    lo=np.percentile(ec,20); hi=np.percentile(ec,60); nm=ec<=lo; sm=ec>=hi
    if nm.sum()<2 or sm.sum()<2: return float("nan")
    return 10*np.log10((np.mean(ee[sm])+1e-12)/(np.mean(ee[nm])+1e-12))
def map_wavs(root):
    m={}
    for dp,_,fs in os.walk(root):
        for fn in fs:
            if fn.endswith(".wav"): m[os.path.splitext(fn)[0]]=os.path.join(dp,fn)
    return m
CLEAN=map_wavs(os.path.join(SV,"data/data_for_test/clean/3s")); NOISY=map_wavs(os.path.join(SV,"data/data_for_test/noise_0db/3s"))
GENERIC=map_wavs(os.path.join(SV,"data/data_for_test/denoised/noise_0db/enhanced_0194")); TSE=map_wavs(os.path.join(BASE,"AI/8.27/exp/tse_v3_ns05duraug_r5/enhanced_official/noise_0db"))
paths={"noisy":NOISY,"generic":GENERIC,"tse":TSE}
meta=json.load(open(os.path.join(DATA,"selected_meta.json"),encoding="utf-8"))
rows=[]
for x in meta["selected"]:
    u=x["test_id"]; ref=load(CLEAN[u]); vals={}
    for c in ["noisy","generic","tse"]:
        est=load(paths[c][u]); n=min(len(ref),len(est)); r,e=ref[:n],est[:n]
        vals[c]=(si_sdr(e,r),seg_snr(e,r),float(pesq(16000,r,e,"wb")),float(stoi(r,e,16000,extended=False)))
    for c in ["noisy","generic","tse"]:
        s,seg,p,st=vals[c]
        rows.append({"test_id":u,"condition":c,"si_sdr_db":s,"si_sdri_db":s-vals["noisy"][0],
                     "seg_snr_db":seg,"snri_db":seg-vals["noisy"][1],
                     "pesq_wb":p,"d_pesq":p-vals["noisy"][2],
                     "stoi":st,"d_stoi":st-vals["noisy"][3]})
out=os.path.join(DATA,"selected_intrusive_metrics.csv")
with open(out,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r["test_id"][:28],r["condition"],"SI-SDR %.2f (i%+.2f) SNR %.2f (i%+.2f) PESQ %.3f (d%+.3f) STOI %.3f (d%+.3f)"%(r["si_sdr_db"],r["si_sdri_db"],r["seg_snr_db"],r["snri_db"],r["pesq_wb"],r["d_pesq"],r["stoi"],r["d_stoi"]))
