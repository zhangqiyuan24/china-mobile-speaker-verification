# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
Extended augmentation for robust fine-tuning:
  - NoiseReverbCorrupter: supports negative SNR (-5~0 dB) and positive SNR (0~15 dB)
  - OverlapCorrupter:   overlap2 / overlap3 with controlled TIR
"""

import torch
import torchaudio
from scipy import signal
import numpy as np
import random

from speakerlab.utils.fileio import load_wav_scp


def addreverb(wav, rir_wav):
    wav = wav.numpy()
    rir_wav = rir_wav.numpy()
    wav_len = wav.shape[0]
    rir_wav = rir_wav / np.sqrt(np.sum(rir_wav**2))
    out_wav = signal.convolve(wav, rir_wav, mode='full')[:wav_len]
    out_wav = out_wav / (np.max(np.abs(out_wav)) + 1e-6)
    return torch.from_numpy(out_wav)


def addnoise(wav, noise=None, snr_high=15, snr_low=0):
    """Add noise at SNR uniformly sampled from [snr_low, snr_high]."""
    if noise is None:
        noise = torch.randn_like(wav)
    noise = noise.numpy()
    wav = wav.numpy()

    wav_len = wav.shape[0]
    noise_len = noise.shape[0]
    if noise_len >= wav_len:
        start = random.randint(0, noise_len - wav_len)
        noise = noise[start:start + wav_len]
    else:
        noise = noise.repeat(wav_len // noise_len + 1)
        noise = noise[:wav_len]

    wav_db = 10 * np.log10(np.mean(wav**2) + 1e-6)
    noise_db = 10 * np.log10(np.mean(noise**2) + 1e-6)
    noise_snr = random.uniform(snr_low, snr_high)
    noise = np.sqrt(10**((wav_db - noise_db - noise_snr) / 10)) * noise
    out_wav = wav + noise
    out_wav = out_wav / (np.max(np.abs(out_wav)) + 1e-6)
    return torch.from_numpy(out_wav)


def addoverlap(target_wav, interferer_wavs, tir_db=0.0):
    """
    Mix target with interferers at controlled aggregate TIR.

    target_wav:      [T,]
    interferer_wavs: list of [T,]
    tir_db:          aggregate TIR (target_power / sum(interferer_power))

    For 2 interferers, each gets (target_rms / sqrt(2)) * 10^(-tir/20)
    so that sum(int_power) = target_power * 10^(-tir/10).
    """
    target = target_wav.numpy()
    n_int = len(interferer_wavs)
    target_rms = np.sqrt(np.mean(target**2) + 1e-12)

    # Desired aggregate interferer RMS
    desired_total_int_rms = target_rms / (10**(tir_db / 20.0))

    # Each interferer contributes equal power
    per_int_rms = desired_total_int_rms / np.sqrt(n_int)

    mixed = target.copy().astype(np.float64)
    for int_wav in interferer_wavs:
        int_wav = int_wav.numpy()
        # Align length
        if len(int_wav) >= len(target):
            start = random.randint(0, len(int_wav) - len(target))
            int_seg = int_wav[start:start + len(target)]
        else:
            reps = int(np.ceil(len(target) / len(int_wav)))
            int_seg = np.tile(int_wav, reps)[:len(target)]

        # Scale to desired per-interferer RMS
        int_rms = np.sqrt(np.mean(int_seg**2) + 1e-12)
        scale = per_int_rms / int_rms
        mixed += int_seg * scale

    out = mixed / (np.max(np.abs(mixed)) + 1e-6)
    return torch.from_numpy(out.astype(np.float32))


class NoiseReverbCorrupter(object):
    """Original corrupter, extended to support negative SNR."""
    def __init__(
        self,
        noise_prob=0.0,
        reverb_prob=0.0,
        noise_file=None,
        reverb_file=None,
        noise_snr_low=0,
        noise_snr_high=15,
    ):
        if reverb_prob > 0.0:
            if reverb_file is None:
                raise ValueError('Reverb_file not be assigned.')
            self.add_reverb = addreverb
            self.reverb_data = load_wav_scp(reverb_file)
            self.reverb_data_keys = list(self.reverb_data.keys())

        if noise_prob > 0.0:
            if noise_file is None:
                raise ValueError('Noise_file not be assigned.')
            self.add_noise = addnoise
            self.noise_data = load_wav_scp(noise_file)
            self.noise_data_keys = list(self.noise_data.keys())

        self.reverb_prob = reverb_prob
        self.noise_prob = noise_prob
        self.noise_snr_low = noise_snr_low
        self.noise_snr_high = noise_snr_high

    def __call__(self, wav, fs=16000):
        if self.reverb_prob > random.random():
            reverb_path = self.reverb_data[random.choice(self.reverb_data_keys)]
            reverb, fs_rir = torchaudio.load(reverb_path)
            assert fs_rir == fs
            wav = self.add_reverb(wav, reverb[0])
        if self.noise_prob > random.random():
            noise_path = self.noise_data[random.choice(self.noise_data_keys)]
            noise, fs_noise = torchaudio.load(noise_path)
            assert fs_noise == fs
            wav = self.add_noise(
                wav, noise[0],
                snr_high=self.noise_snr_high,
                snr_low=self.noise_snr_low,
            )
        return wav


class OverlapCorrupter(object):
    """
    Online overlap augmentation using training-set interferer speakers.

    Arguments:
        train_data: dict {spk_id: [wav_path, ...]} — interferer pool
        spk_label:  speaker id of the current sample (to exclude self)
        num_interferers: 1 → overlap2,  2 → overlap3
        tir_db:     aggregate TIR (target / total interference), default 0 dB
    """
    def __init__(self, train_data, spk_label, num_interferers=1, tir_db=0.0):
        self.train_data = train_data
        self.spk_label = spk_label
        self.num_interferers = num_interferers
        self.tir_db = tir_db

        # Build speaker list excluding current speaker
        self.other_spks = [s for s in train_data.keys() if s != spk_label]
        if len(self.other_spks) < num_interferers:
            raise ValueError(
                f"Not enough other speakers ({len(self.other_spks)}) "
                f"for overlap{num_interferers+1} with speaker {spk_label}"
            )

    def __call__(self, wav, fs=16000):
        chosen_spks = random.sample(self.other_spks, self.num_interferers)
        int_wavs = []
        for spk in chosen_spks:
            int_path = random.choice(self.train_data[spk])
            int_wav, int_fs = torchaudio.load(int_path)
            if int_fs != fs:
                int_wav = torchaudio.functional.resample(int_wav, int_fs, fs)
            int_wavs.append(int_wav[0])

        return addoverlap(wav, int_wavs, self.tir_db)
