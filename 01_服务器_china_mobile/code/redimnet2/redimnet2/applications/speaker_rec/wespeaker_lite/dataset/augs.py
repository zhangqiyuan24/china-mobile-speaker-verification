"""Composable audio augmentors for speaker verification training.

Supports noise addition, reverberation, codec compression, and low-pass
filtering. Augmentors can be composed via Sequential and OneOf combinators,
and deserialized from YAML/dict configs via ``deserialize_augmentor()``.

Noise and RIR data can be loaded from either:
  - LMDB files (via ``lmdb_file`` parameter, uses LmdbData/LmdbDataMap)
  - Disk directories (via ``dataset_dir`` parameter, globs for audio files)

Usage::

    aug_config = {
        'aug_type': 'Sequential',
        'prob': 1.0,
        'augmentors': [
            {'aug_type': 'OneOf', 'prob': 0.65, 'augmentors': [
                {'aug_type': 'CustomNoise', 'lmdb_file': '/path/to/musan/lmdb',
                 'snr': [0, 15], 'prob': 1.0},
            ]},
            {'aug_type': 'Reverb', 'lmdb_file': '/path/to/rirs/lmdb',
             'prob': 0.25},
            {'aug_type': 'CodecAug', 'prob': 0.3},
        ]
    }
    augmentor = deserialize_augmentor(aug_config)
    # augmentor expects dict with 'wav' (torch Tensor [1,T]) and 'sample_rate'
"""

import io
import os
import glob
import random

import numpy as np
import torch
import soundfile as sf
from scipy import signal as sp_signal
from scipy.io import wavfile
from abc import abstractmethod
from typing import Union, List, Tuple


# ── helpers ──────────────────────────────────────────────────────────

def _random_int(val: Union[int, Tuple[int, int], List[int]]) -> int:
    if isinstance(val, (tuple, list)):
        lo, hi = val
        return lo if lo >= hi else np.random.randint(lo, hi)
    return int(val)


def _random_float(val: Union[float, Tuple[float, float], List[float]]) -> float:
    if isinstance(val, (tuple, list)):
        lo, hi = val
        return lo + np.random.rand() * (hi - lo)
    return float(val)


def _crop_or_pad(signal: np.ndarray, target_len: int) -> np.ndarray:
    if len(signal) > target_len:
        start = np.random.randint(0, len(signal) - target_len)
        return signal[start:start + target_len]
    elif len(signal) < target_len:
        return np.pad(signal, (0, target_len - len(signal)), mode='wrap')
    return signal


# ── audio source (LMDB or disk) ─────────────────────────────────────

class AudioSource:
    """Lazy-loading audio source from LMDB or a directory of audio files."""

    def __init__(self, lmdb_file=None, dataset_dir=None, file_pattern='**/*.wav',
                 categorized=False):
        self._lmdb = None
        self._disk_files = None
        self._categorized = categorized

        if lmdb_file is not None:
            self._lmdb_path = lmdb_file
            self._source_type = 'lmdb'
        elif dataset_dir is not None:
            self._dataset_dir = dataset_dir
            self._file_pattern = file_pattern
            self._source_type = 'disk'
        else:
            raise ValueError("Either lmdb_file or dataset_dir must be provided")

    def _ensure_loaded(self):
        if self._source_type == 'lmdb' and self._lmdb is None:
            if self._categorized:
                from wespeaker_lite.dataset.lmdb_data import LmdbDataMap
                self._lmdb = LmdbDataMap(self._lmdb_path)
            else:
                from wespeaker_lite.dataset.lmdb_data import LmdbData
                self._lmdb = LmdbData(self._lmdb_path)
        elif self._source_type == 'disk' and self._disk_files is None:
            pattern = os.path.join(self._dataset_dir, self._file_pattern)
            self._disk_files = sorted(glob.glob(pattern, recursive=True))
            if not self._disk_files:
                raise FileNotFoundError(
                    f"No files found matching {pattern}")

    def random_one(self, category=None, frames=None) -> np.ndarray:
        """Return a random audio signal as float32 numpy array.

        If ``frames`` is given, decode at most that many samples from a
        random offset (disk source only). MUSAN files are 5-15 MB each and
        full decode dominates CustomNoise cost (esp. babble: 3-7 reads/call);
        partial decode is ~15x faster with statistically identical sampling
        (uniform random crop, just moved from post-decode to seek-time).
        """
        self._ensure_loaded()

        if self._source_type == 'lmdb':
            _, data = self._lmdb.random_one(category)
            sr, audio = wavfile.read(io.BytesIO(data))
            audio = audio.astype(np.float32)
            if audio.dtype == np.int16 or np.max(np.abs(audio)) > 2.0:
                audio = audio / (1 << 15)
            return audio, sr

        # Retry on read failure: libsndfile/scipy can raise on malformed
        # files. Use context manager so file handles always close (avoids FD
        # leak across millions of opens per worker over an epoch).
        last_err = None
        for _ in range(5):
            path = random.choice(self._disk_files)
            try:
                if frames is not None:
                    with sf.SoundFile(path) as f:
                        sr, n = f.samplerate, f.frames
                        if n > frames:
                            start = np.random.randint(0, n - frames)
                            f.seek(start)
                            audio = f.read(frames, dtype='float32',
                                           always_2d=False)
                        else:
                            audio = f.read(dtype='float32', always_2d=False)
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    return audio, sr

                sr, audio = wavfile.read(path)
                audio = audio.astype(np.float32)
                if np.max(np.abs(audio)) > 2.0:
                    audio = audio / (1 << 15)
                return audio, sr
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(
            f"AudioSource.random_one: 5 consecutive read failures "
            f"(last path={path}, last_err={last_err})")


# ── base augmentor ───────────────────────────────────────────────────

class Augmentor:
    def __init__(self, prob=0.5, aug_type="Augmentor"):
        self.prob = prob

    @abstractmethod
    def __apply__(self, example: dict) -> dict:
        pass

    def __call__(self, example: dict) -> dict:
        if np.random.rand() <= self.prob:
            return self.__apply__(example)
        return example


# ── combinators ──────────────────────────────────────────────────────

class OneOf(Augmentor):
    def __init__(self, augmentors: List[Augmentor], prob=0.5, aug_type="OneOf"):
        super().__init__(prob, aug_type)
        self.augmentors = augmentors

    def __apply__(self, example: dict) -> dict:
        idx = np.random.randint(0, len(self.augmentors))
        return self.augmentors[idx](example)

    def __repr__(self):
        return f"OneOf([{', '.join(repr(a) for a in self.augmentors)}], prob={self.prob})"


class Sequential(Augmentor):
    def __init__(self, augmentors: List[Augmentor], prob=1.0, aug_type="Sequential"):
        super().__init__(prob, aug_type)
        self.augmentors = augmentors

    def __apply__(self, example: dict) -> dict:
        for aug in self.augmentors:
            example = aug(example)
        return example

    def __repr__(self):
        return f"Sequential([{', '.join(repr(a) for a in self.augmentors)}], prob={self.prob})"


# ── noise augmentor ──────────────────────────────────────────────────

class CustomNoise(Augmentor):
    """Add noise at random SNR. Supports babble (multi-noise averaging).

    Noise loaded from LMDB (``lmdb_file``) or disk directory (``dataset_dir``).
    """

    def __init__(self, snr=(0, 15), babble=False,
                 lmdb_file=None, dataset_dir=None, file_pattern='**/*.wav',
                 noise_category=None,
                 prob=0.5, eps=1e-5, aug_type="CustomNoise", **kwargs):
        super().__init__(prob, aug_type)
        self.snr = snr
        self.num_noises = [3, 7] if babble else [1, 1]
        self.eps = eps
        self.noise_category = noise_category
        self._source = AudioSource(
            lmdb_file=lmdb_file, dataset_dir=dataset_dir,
            file_pattern=file_pattern,
            categorized=(lmdb_file is not None and noise_category is not None))

    def __apply__(self, example: dict) -> dict:
        audio = example['wav'].numpy()[0]
        audio_len = audio.shape[0]
        sample_rate = example['sample_rate']

        num_noises = _random_int(self.num_noises)
        noises = []
        # Read with margin for resampling (assume <=2x SR mismatch); full
        # path falls back if SR differs after read.
        read_frames = audio_len * 2
        for _ in range(num_noises):
            noise, noise_sr = self._source.random_one(
                self.noise_category, frames=read_frames)
            if noise_sr != sample_rate:
                from scipy.signal import resample as sp_resample
                noise = sp_resample(noise, int(len(noise) / noise_sr * sample_rate))
            noise = _crop_or_pad(noise, audio_len)
            noises.append(noise)

        noise_audio = np.mean(noises, axis=0)
        snr_db = _random_float(self.snr)

        audio_power = np.mean(audio ** 2) + self.eps
        noise_power = np.mean(noise_audio ** 2) + self.eps
        scale = np.sqrt(10 ** ((10 * np.log10(audio_power) -
                                10 * np.log10(noise_power) - snr_db) / 10))

        example['wav'] = torch.from_numpy(audio + noise_audio * scale).unsqueeze(0).float()
        return example

    def __repr__(self):
        return f"CustomNoise(snr={self.snr}, num_noises={self.num_noises}, prob={self.prob})"


# ── reverb augmentor ─────────────────────────────────────────────────

class Reverb(Augmentor):
    """Convolve with a room impulse response.

    RIR loaded from LMDB (``lmdb_file``) or disk directory (``dataset_dir``).
    """

    def __init__(self, rir_length_sec=1.0,
                 lmdb_file=None, dataset_dir=None, file_pattern='**/*.wav',
                 prob=0.5, aug_type="Reverb", **kwargs):
        super().__init__(prob, aug_type)
        self.rir_length_sec = rir_length_sec
        self._source = AudioSource(
            lmdb_file=lmdb_file, dataset_dir=dataset_dir,
            file_pattern=file_pattern)

    def __apply__(self, example: dict) -> dict:
        audio = example['wav'].numpy()[0]
        audio_len = audio.shape[0]
        sample_rate = example['sample_rate']

        rir, rir_sr = self._source.random_one()
        if rir_sr != sample_rate:
            from scipy.signal import resample as sp_resample
            rir = sp_resample(rir, int(len(rir) / rir_sr * sample_rate))

        rir = rir[:int(self.rir_length_sec * sample_rate)]
        rir = rir / np.sqrt(np.sum(rir ** 2) + 1e-10)
        out = sp_signal.convolve(audio, rir, mode='full')[:audio_len]

        example['wav'] = torch.from_numpy(out).unsqueeze(0).float()
        return example

    def __repr__(self):
        return f"Reverb(rir_len={self.rir_length_sec}s, prob={self.prob})"


# ── codec augmentor ──────────────────────────────────────────────────

class CodecAug(Augmentor):
    """Apply random codec compression (mp3, opus, vorbis, g722, g711).

    Requires FFmpeg and torchaudio with AudioEffector support.
    """

    def __init__(self, sample_rate=16000, prob=0.5, aug_type="CodecAug", **kwargs):
        super().__init__(prob, aug_type)
        self.sample_rate = sample_rate
        from torchaudio.io import AudioEffector, CodecConfig
        self.effectors = {
            "mp3-lo":     AudioEffector(format="mp3", codec_config=CodecConfig(qscale=9)),
            "mp3-mid":    AudioEffector(format="mp3", codec_config=CodecConfig(qscale=5)),
            "mp3-hi":     AudioEffector(format="mp3", codec_config=CodecConfig(qscale=2)),
            "vorbis-lo":  AudioEffector(format="ogg", encoder="vorbis", codec_config=CodecConfig(qscale=1)),
            "vorbis-hi":  AudioEffector(format="ogg", encoder="vorbis", codec_config=CodecConfig(qscale=10)),
            "opus-11k":   AudioEffector(format="ogg", encoder="opus", codec_config=CodecConfig(bit_rate=11000)),
            "opus-16k":   AudioEffector(format="ogg", encoder="opus", codec_config=CodecConfig(bit_rate=16000)),
            "opus-32k":   AudioEffector(format="ogg", encoder="opus", codec_config=CodecConfig(bit_rate=32000)),
            "opus-64k":   AudioEffector(format="ogg", encoder="opus", codec_config=CodecConfig(bit_rate=64000)),
            "g722":       AudioEffector(format="g722"),
            "g711-mulaw": AudioEffector(format="wav", encoder="pcm_mulaw"),
            "g711-alaw":  AudioEffector(format="wav", encoder="pcm_alaw"),
        }

    def __apply__(self, example: dict) -> dict:
        audio = example['wav']  # (1, T) torch tensor
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        effect_name = random.choice(list(self.effectors.keys()))
        effect = self.effectors[effect_name]

        out = effect.apply(audio.T, self.sample_rate).T  # (1, T')
        # Codec may change length slightly — match original
        orig_len = example['wav'].shape[-1]
        if out.shape[-1] > orig_len:
            out = out[..., :orig_len]
        elif out.shape[-1] < orig_len:
            out = torch.nn.functional.pad(out, (0, orig_len - out.shape[-1]))

        example['wav'] = out.float()
        return example

    def __repr__(self):
        return f"CodecAug(codecs={len(self.effectors)}, prob={self.prob})"


# ── low-pass filter augmentor ────────────────────────────────────────

class LowPassAug(Augmentor):
    """Apply a smooth low-pass filter in the frequency domain."""

    def __init__(self, cutoff_ratio=0.5, transition_width=0.1,
                 prob=0.5, aug_type="LowPassAug", **kwargs):
        super().__init__(prob, aug_type)
        self.cutoff_ratio = cutoff_ratio
        self.transition_width = transition_width

    def __apply__(self, example: dict) -> dict:
        audio = example['wav'].numpy()[0]
        T = audio.shape[0]

        rfft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(T)

        cutoff = self.cutoff_ratio * freqs[-1]
        tw = self.transition_width * freqs[-1]

        mask = np.ones_like(freqs)
        mask[freqs > cutoff + tw / 2] = 0
        transition = np.logical_and(freqs >= cutoff - tw / 2,
                                     freqs <= cutoff + tw / 2)
        mask[transition] = 0.5 * (1 + np.cos(
            np.pi * (freqs[transition] - cutoff) / tw))

        filtered = np.fft.irfft(rfft * mask, n=T)

        example['wav'] = torch.from_numpy(filtered).unsqueeze(0).float()
        return example

    def __repr__(self):
        return f"LowPassAug(cutoff={self.cutoff_ratio}, prob={self.prob})"


# ── deserialization ──────────────────────────────────────────────────

_AUG_CLASSES = {
    'Sequential': Sequential,
    'OneOf': OneOf,
    'CustomNoise': CustomNoise,
    'Reverb': Reverb,
    'CodecAug': CodecAug,
    'LowPassAug': LowPassAug,
}


def deserialize_augmentor(config: dict) -> Augmentor:
    """Build an augmentor tree from a nested dict (YAML-friendly).

    Example config::

        {
            'aug_type': 'Sequential',
            'prob': 1.0,
            'augmentors': [
                {'aug_type': 'CustomNoise', 'lmdb_file': '...', 'snr': [0,15], 'prob': 0.65},
                {'aug_type': 'Reverb', 'lmdb_file': '...', 'prob': 0.25},
                {'aug_type': 'CodecAug', 'prob': 0.3},
            ]
        }
    """
    config = dict(config)  # shallow copy
    aug_type = config.pop('aug_type')

    if aug_type not in _AUG_CLASSES:
        raise ValueError(f"Unknown augmentor type: {aug_type}")

    cls = _AUG_CLASSES[aug_type]

    if aug_type in ('Sequential', 'OneOf'):
        children = [deserialize_augmentor(c) for c in config.pop('augmentors')]
        return cls(augmentors=children, **config)
    else:
        return cls(**config)
