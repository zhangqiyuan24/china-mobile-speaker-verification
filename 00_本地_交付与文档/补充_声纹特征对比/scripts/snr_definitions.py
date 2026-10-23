import os, csv, numpy as np, soundfile as sf
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
def global_snr(est,ref):
    est,ref=align(est,ref); return 10*np.log10((np.sum(ref**2)+1e-12)/(np.sum((est-ref)**2)+1e-12))
def si_sdr(est,ref):
    est,ref=align(est,ref); est=est-est.mean(); ref=ref-ref.mean()
    a=float(np.dot(est,ref)/(np.dot(ref,ref)+1e-10)); t=a*ref; e=est-t
    return 10*np.log10((np.sum(t**2)+1e-10)/(np.sum(e**2)+1e-10))
def frame_energy(x,win=512,hop=256):
    if len(x)<win: x=np.pad(x,(0,win-len(x)))
    n=1+(len(x)-win)//hop; idx=np.arange(win)[None,:]+hop*np.arange(n)[:,None]
    return np.mean(x[idx]**2,axis=1)+1e-12
def vad_snr(est,ref):
    est,ref=align(est,ref); ec=frame_energy(ref); ee=frame_energy(est)
    n=min(len(ec),len(ee)); ec,ee=ec[:n],ee[:n]
    lo=np.percentile(ec,20); hi=np.percentile(ec,60); nm=ec<=lo; sm=ec>=hi
    return 10*np.log10((np.mean(ee[sm])+1e-12)/(np.mean(ee[nm])+1e-12))
CLEAN=map_wavs(os.path.join(SV,"data/data_for_test/clean/3s")); N0=map_wavs(os.path.join(SV,"data/data_for_test/noise_0db/3s")); NM=map_wavs(os.path.join(SV,"data/data_for_test/noise_m5db/3s"))
ids=sorted({r["test_id"] for r in csv.DictReader(open(os.path.join(DATA,"intrusive_metrics.csv"),encoding="utf-8"))})
for name,M in [("0 dB",N0),("-5 dB",NM)]:
    gs=[]; ss=[]; vs=[]
    for u in ids:
        if u not in CLEAN or u not in M: continue
        s=load(CLEAN[u]); x=load(M[u])
        gs.append(global_snr(x,s)); ss.append(si_sdr(x,s)); vs.append(vad_snr(x,s))
    print("%s set: n=%d global SNR %.2f dB | SI-SDR %.2f dB | VAD-SNR %.2f dB"%(name,len(gs),np.mean(gs),np.mean(ss),np.mean(vs)))
