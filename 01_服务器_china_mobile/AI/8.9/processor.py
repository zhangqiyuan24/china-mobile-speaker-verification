# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import random
import pickle
import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi

from speakerlab.process.augmentation import NoiseReverbCorrupter
from speakerlab.utils.fileio import load_data_csv


class WavReader(object):
    def __init__(self,
        sample_rate = 16000,
        duration: float = 3.0,
        speed_pertub: bool = False,
        lm: bool = True,
    ):
        self.duration = duration
        self.sample_rate = sample_rate
        self.speed_pertub = speed_pertub
        self.lm = lm

    def __call__(self, wav_path):
        wav, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav[0]

        if self.speed_pertub and self.lm:
            speeds = [1.0, 0.9, 1.1]
            speed_idx = random.randint(0, 2)
            if speed_idx > 0:
                wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                    wav.unsqueeze(0), self.sample_rate, [['speed', str(speeds[speed_idx])], ['rate', str(self.sample_rate)]])
        else:
            speed_idx = 0

        wav = wav.squeeze(0)
        data_len = wav.shape[0]

        chunk_len = int(self.duration * self.sample_rate)
        if data_len >= chunk_len:
            start = random.randint(0, data_len - chunk_len)
            end = start + chunk_len
            wav = wav[start:end]
        else:
            wav = F.pad(wav, (0, chunk_len - data_len))

        return wav, speed_idx

class SpkLabelEncoder(object):
    def __init__(self, data_file):
        self.lab2ind = {}
        self.ind2lab = {}
        self.starting_index = -1
        self.load_from_csv(data_file)

    def __call__(self, spk, speed_idx=0):
        spkid = self.lab2ind[spk]
        spkid = spkid + len(self.lab2ind) * speed_idx
        return spkid

    def load_from_csv(self, path):
        self.data = load_data_csv(path)
        for key in self.data:
            self.add(self.data[key]['spk'])

    def add(self, label):
        if label in self.lab2ind:
            return
        index = self._next_index()
        self.lab2ind[label] = index
        self.ind2lab[index] = label

    def _next_index(self):
        self.starting_index += 1
        return self.starting_index

    def __len__(self):
        return len(self.lab2ind)

    def save(self, path, device=None):
        with open(path, 'wb') as f:
            pickle.dump(self.lab2ind, f)

    def load(self, path, device=None):
        self.lab2ind = {}
        self.ind2lab = {}
        with open(path, 'rb') as f:
            self.lab2ind = pickle.load(f)
        for label in self.lab2ind:
            self.ind2lab[self.lab2ind[label]] = label


class SpkVeriAug(object):
    def __init__(
        self,
        aug_prob: float = 0.0,
        noise_file: str = None,
        reverb_file: str = None,
    ):
        self.aug_prob = aug_prob
        if aug_prob > 0:
            self.add_noise = NoiseReverbCorrupter(
                noise_prob=1.0,
                noise_file=noise_file,
                )
            self.add_rir = NoiseReverbCorrupter(
                reverb_prob=1.0,
                reverb_file=reverb_file,
                )
            self.add_rir_noise = NoiseReverbCorrupter(
                noise_prob=1.0,
                reverb_prob=1.0,
                noise_file=noise_file,
                reverb_file=reverb_file,
                )

            self.augmentations = [self.add_noise, self.add_rir, self.add_rir_noise]

    def __call__(self, wav, spk_label=None):
        sample_rate = 16000
        if self.aug_prob > random.random():
            aug = random.choice(self.augmentations)
            wav = aug(wav, sample_rate)
        return wav


class RobustSpkAug(object):
    """
    Multi-condition augmentation for robust fine-tuning.

    Modes (configurable probability):
      - clean:        no augmentation
      - pos_noise:    SNR 0~15 dB (reuses original augmentation range)
      - neg_noise:    SNR -5~0 dB (new negative SNR range)
      - reverb:       RIR convolution only
      - noise_reverb: noise + reverb cascade
      - overlap2:     2-speaker overlap, aggregate TIR=0dB
      - overlap3:     3-speaker overlap, aggregate TIR=0dB

    The interferer pool for overlap is built from train_data_dir.
    """
    def __init__(
        self,
        aug_prob: float = 0.6,
        noise_file: str = None,
        reverb_file: str = None,
        # Mode probabilities (should sum to ~1.0)
        clean_prob: float = 0.15,
        pos_noise_prob: float = 0.20,
        neg_noise_prob: float = 0.30,
        reverb_prob: float = 0.10,
        noise_reverb_prob: float = 0.10,
        overlap2_prob: float = 0.10,
        overlap3_prob: float = 0.05,
        # SNR ranges
        pos_snr_low: float = 0.0,
        pos_snr_high: float = 15.0,
        neg_snr_low: float = -5.0,
        neg_snr_high: float = 0.0,
        # Overlap
        overlap_tir_db: float = 0.0,
        overlap_spk2utt: dict = None,
    ):
        from speakerlab.process.augmentation import (
            NoiseReverbCorrupter, OverlapCorrupter,
        )

        self.aug_prob = aug_prob
        self.overlap_spk2utt = overlap_spk2utt or {}

        # Build corrupter instances
        if noise_file and pos_noise_prob + neg_noise_prob + noise_reverb_prob > 0:
            self.pos_noise = NoiseReverbCorrupter(
                noise_prob=1.0, reverb_prob=0.0,
                noise_file=noise_file, noise_snr_low=pos_snr_low,
                noise_snr_high=pos_snr_high,
            )
            self.neg_noise = NoiseReverbCorrupter(
                noise_prob=1.0, reverb_prob=0.0,
                noise_file=noise_file, noise_snr_low=neg_snr_low,
                noise_snr_high=neg_snr_high,
            )
        else:
            self.pos_noise = None
            self.neg_noise = None

        if reverb_file and reverb_prob + noise_reverb_prob > 0:
            self.reverb_only = NoiseReverbCorrupter(
                noise_prob=0.0, reverb_prob=1.0, reverb_file=reverb_file,
            )
            self.noise_reverb = NoiseReverbCorrupter(
                noise_prob=1.0, reverb_prob=1.0,
                noise_file=noise_file, reverb_file=reverb_file,
                noise_snr_low=pos_snr_low, noise_snr_high=pos_snr_high,
            )
        else:
            self.reverb_only = None
            self.noise_reverb = None

        # Build mode sampler
        modes = []
        probs = []
        if clean_prob > 0:
            modes.append('clean'); probs.append(clean_prob)
        if pos_noise_prob > 0 and self.pos_noise:
            modes.append('pos_noise'); probs.append(pos_noise_prob)
        if neg_noise_prob > 0 and self.neg_noise:
            modes.append('neg_noise'); probs.append(neg_noise_prob)
        if reverb_prob > 0 and self.reverb_only:
            modes.append('reverb'); probs.append(reverb_prob)
        if noise_reverb_prob > 0 and self.noise_reverb:
            modes.append('noise_reverb'); probs.append(noise_reverb_prob)
        if overlap2_prob > 0 and self.overlap_spk2utt:
            modes.append('overlap2'); probs.append(overlap2_prob)
        if overlap3_prob > 0 and self.overlap_spk2utt:
            modes.append('overlap3'); probs.append(overlap3_prob)

        # Normalize
        total = sum(probs) if probs else 1.0
        self.modes = modes
        self.probs = [p / total for p in probs]
        self.overlap_tir_db = overlap_tir_db

    def __call__(self, wav, spk_label=None):
        """wav: [T,], spk_label: speaker id string for overlap exclusion"""
        import random
        sample_rate = 16000

        if self.aug_prob <= random.random():
            return wav  # no augmentation at all

        if not self.modes:
            return wav

        mode = random.choices(self.modes, weights=self.probs, k=1)[0]

        if mode == 'clean':
            return wav
        elif mode == 'pos_noise':
            return self.pos_noise(wav, sample_rate)
        elif mode == 'neg_noise':
            return self.neg_noise(wav, sample_rate)
        elif mode == 'reverb':
            return self.reverb_only(wav, sample_rate)
        elif mode == 'noise_reverb':
            return self.noise_reverb(wav, sample_rate)
        elif mode in ('overlap2', 'overlap3'):
            if spk_label and self.overlap_spk2utt:
                num_int = 1 if mode == 'overlap2' else 2
                from speakerlab.process.augmentation import OverlapCorrupter
                oc = OverlapCorrupter(
                    self.overlap_spk2utt, spk_label,
                    num_interferers=num_int, tir_db=self.overlap_tir_db,
                )
                return oc(wav, sample_rate)
            return wav
        else:
            return wav


class FBank(object):
    def __init__(self,
        n_mels,
        sample_rate,
        mean_nor: bool = False,
    ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=0):
        sr = 16000
        assert sr==self.sample_rate
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        # select single channel
        if wav.shape[0] > 1:
            wav = wav[0, :]
            wav = wav.unsqueeze(0)
        assert len(wav.shape) == 2 and wav.shape[0]==1
        feat = Kaldi.fbank(wav, num_mel_bins=self.n_mels,
            sample_frequency=sr, dither=dither)
        # feat: [T, N]
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat
