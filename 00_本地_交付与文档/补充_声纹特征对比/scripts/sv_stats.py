import os
import numpy as np
import librosa
import soundfile as sf

base = "/home/ligangrui/china_mobile/AI/10.17/wav"
rows = []
for u in sorted(os.listdir(base)):
    d = os.path.join(base, u)
    if not os.path.isdir(d):
        continue
    for cond, fn in [("clean", "clean.wav"), ("noisy", "noisy_0db.wav"),
                     ("generic", "fullsubnetplus.wav"), ("tse", "tse.wav")]:
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            continue
        x, sr = sf.read(p, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(1)
        rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
        peak = float(np.max(np.abs(x)))
        clip = float(np.mean(np.abs(x) >= 0.999))
        dc = float(np.mean(x))
        S = np.abs(librosa.stft(x, n_fft=512, hop_length=160)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
        e = S.sum(1) + 1e-12
        low = float(e[(freqs >= 20) & (freqs < 500)].sum() / e.sum())
        mid = float(e[(freqs >= 500) & (freqs < 2000)].sum() / e.sum())
        high = float(e[(freqs >= 2000) & (freqs <= 7600)].sum() / e.sum())
        m = (freqs >= 100) & (freqs <= 4000)
        slope = float(np.polyfit(np.log10(freqs[m]), 10 * np.log10(e[m] + 1e-12), 1)[0])
        rows.append((u.split("_Distance")[0], cond, 20 * np.log10(rms + 1e-12),
                     peak, clip, dc, low, mid, high, slope))

header = "{:28s} {:8s} {:>7s} {:>6s} {:>7s} {:>8s} {:>6s} {:>6s} {:>6s} {:>7s}".format(
    "utt", "cond", "RMSdB", "peak", "clip%", "DC", "low%", "mid%", "high%", "tilt")
print(header)
for r in rows:
    print("{:28s} {:8s} {:7.2f} {:6.3f} {:7.3f} {:+8.4f} {:6.1f} {:6.1f} {:6.1f} {:7.2f}".format(
        r[0], r[1], r[2], r[3], 100 * r[4], r[5], 100 * r[6], 100 * r[7], 100 * r[8], r[9]))
