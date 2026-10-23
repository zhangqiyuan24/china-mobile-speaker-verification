import os, csv, numpy as np, soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pesq import pesq
from pystoi import stoi
BASE="/home/ligangrui/china_mobile"; SV=os.path.join(BASE,"code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa"); DATA=os.path.join(BASE,"AI/10.17/data"); FIGS=os.path.join(BASE,"AI/10.17/figs")
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
CLEAN=map_wavs(os.path.join(SV,"data/data_for_test/clean/3s")); NOISY=map_wavs(os.path.join(SV,"data/data_for_test/noise_0db/3s"))
ids=sorted({r["test_id"] for r in csv.DictReader(open(os.path.join(DATA,"intrusive_metrics.csv"),encoding="utf-8"))})[:20]
snrs=[20,10,5,0,-5]; rows=[]
for u in ids:
    s=load(CLEAN[u]); x=load(NOISY[u]); n=min(len(s),len(x)); s,x=s[:n],x[:n]
    noise=x-s
    ps=float(np.mean(s**2)+1e-12); pn=float(np.mean(noise**2)+1e-12)
    input_snr=10*np.log10(ps/pn)
    rows.append({"test_id":u,"target_snr_db":"actual","input_snr_db":input_snr,
                 "pesq_wb":float(pesq(16000,s,x,"wb")),"stoi":float(stoi(s,x,16000,extended=False))})
    for snr in snrs:
        g=np.sqrt(ps/(pn*10**(snr/10)))
        y=s+g*noise
        rows.append({"test_id":u,"target_snr_db":snr,"input_snr_db":10*np.log10(np.mean(s**2)/(np.mean((y-s)**2)+1e-12)),
                     "pesq_wb":float(pesq(16000,s,y,"wb")),"stoi":float(stoi(s,y,16000,extended=False))})
out=os.path.join(DATA,"pesq_snr_sweep.csv")
with open(out,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["test_id","target_snr_db","input_snr_db","pesq_wb","stoi"]); w.writeheader(); w.writerows(rows)
print("actual noisy input SNR mean %.2f dB"%np.mean([r["input_snr_db"] for r in rows if r["target_snr_db"]=="actual"]))
print("actual noisy PESQ %.3f STOI %.3f"%(np.mean([r["pesq_wb"] for r in rows if r["target_snr_db"]=="actual"]),np.mean([r["stoi"] for r in rows if r["target_snr_db"]=="actual"])))
for snr in snrs:
    sub=[r for r in rows if r["target_snr_db"]==snr]
    print("snr %+3d dB: PESQ %.3f STOI %.3f (n=%d)"%(snr,np.mean([r["pesq_wb"] for r in sub]),np.mean([r["stoi"] for r in sub]),len(sub)))
# plot
fig,ax1=plt.subplots(figsize=(7.2,4.2))
xs=[]; ps_=[]; ss_=[]
for snr in sorted(snrs):
    sub=[r for r in rows if r["target_snr_db"]==snr]; xs.append(snr); ps_.append(np.mean([r["pesq_wb"] for r in sub])); ss_.append(np.mean([r["stoi"] for r in sub]))
ax1.plot(xs,ps_,"o-",color="#d55e00",label="PESQ-WB")
ax1.set_xlabel("input SNR (dB)"); ax1.set_ylabel("PESQ-WB",color="#d55e00"); ax1.tick_params(axis="y",labelcolor="#d55e00")
ax1.grid(alpha=0.25)
ax2=ax1.twinx(); ax2.plot(xs,ss_,"s--",color="#0072b2",label="STOI"); ax2.set_ylabel("STOI",color="#0072b2"); ax2.tick_params(axis="y",labelcolor="#0072b2")
ax1.set_title("PESQ-WB / STOI vs input SNR (20 test utterances, actual noise residual)")
ax1.set_ylim(0.5,4.7); ax2.set_ylim(0,1.0)
fig.tight_layout(); fig.savefig(os.path.join(FIGS,"pesq_vs_snr.png"),bbox_inches="tight")
print("saved",os.path.join(FIGS,"pesq_vs_snr.png"))
