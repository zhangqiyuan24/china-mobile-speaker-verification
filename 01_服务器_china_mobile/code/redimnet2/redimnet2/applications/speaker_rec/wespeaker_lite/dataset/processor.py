# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2022 Hongji Wang (jijijiang77@gmail.com)
#               2023 Zhengyang Chen (chenzhengyang117@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
# import kaldiio
import json
import resampy
import warnings
import logging
import random
import traceback
import tarfile
import soundfile as sf
from pathlib import Path
from subprocess import PIPE, Popen
from urllib.parse import urlparse
from contextlib import nullcontext

import numpy as np
from scipy import signal
from scipy.io import wavfile
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])







def read_audio_section(fp,
        samplerate,
        duration=3.0,
        min_duration=1.0,
        start_time=None,
        extend_to_duration=False,
    ):
    # No Path(fp).exists() pre-check: sf.SoundFile already raises on a missing
    # file (callers handle that identically), and the explicit stat() was a
    # redundant virtiofs round-trip (~0.65 ms/sample, ~2 stat calls/sample).
    f = sf.SoundFile(fp)

    tar_dur = duration

    can_seek = f.seekable() # True
    if not can_seek:
        raise ValueError("Not compatible with seeking")

    sr = f.samplerate
    ns = f.frames
    tot_dur = ns / sr

    if min_duration > tot_dur:
        return None, None

    if start_time is None:
        rnd_start_range = max(0,tot_dur-tar_dur)
        t = np.random.uniform()
        start_time = rnd_start_range*t
    start_frame = int(sr * start_time)
    # print(f"start_frame : {start_frame}, tot len : {ns}")

    if duration is not None:
        frames_to_read = int(sr * duration)
    else:
        frames_to_read = ns

    f.seek(start_frame)

    samples = f.read(frames_to_read)

    if sr != samplerate:
        # warnings.warn(f"Runtime resampling : {fp}")
        samples = resampy.resample(samples,sr,samplerate)

    frames_to_read = int(samplerate * duration)
    if extend_to_duration:
        shortage = frames_to_read - len(samples)
        if shortage > 0:
            samples = np.pad(samples, (0, shortage), 'wrap')

    samples = samples[:frames_to_read]

    return samples, sr

def parse_raw_fast_and_filter(
        data,
        chunk_len,
        data_type='shard/raw/feat',
        min_num_frames=100,
        max_num_frames=800,
        frame_shift=10,
    ):
    """ Parse key/wav/spk from json line

        Args:
            data: Iterable[str], str is a json line has key/wav/spk

        Returns:
            Iterable[{key, wav, spk, sample_rate}]
    """
    assert data_type == 'raw_fast'

    for sample in data:
        # Read example & audio
        assert 'src' in sample
        json_line = sample['src']
        try:
            obj = json.loads(json_line)
        except Exception as e:
            logging.warning(f"Failed to parse json line: {json_line}")
            raise e
        assert 'key' in obj
        assert 'wav' in obj
        assert 'spk' in obj
        key = obj['key']
        wav_file = obj['wav']
        spk = obj['spk']
        try:
            sample_rate = 16_000

            min_len = int(frame_shift / 1000 * min_num_frames * sample_rate)
            max_len = int(frame_shift / 1000 * max_num_frames * sample_rate)

            ext_spd_pert_dur = (chunk_len/sample_rate) * 1.1

            waveform, sample_rate = read_audio_section(wav_file,
                        samplerate=sample_rate,
                        min_duration=min_len/sample_rate,
                        duration=ext_spd_pert_dur, # Extended for speed perturb
                        start_time=None,
                        extend_to_duration=True)

            if waveform is None:
                print(f"Skipping waveform : {obj}")
                continue

            waveform = waveform[:int(ext_spd_pert_dur*sample_rate)]

            waveform = torch.from_numpy(waveform).unsqueeze(0).float()

            if 'vad' in obj:
                print(f"NO VAD TURNED ON!")
                raise NotImplemented()

            sample = dict(key=key,
                           spk=spk,
                           path=wav_file,
                           wav=waveform,
                           sample_rate=sample_rate)
            yield sample
        except Exception as ex:
            traceback.print_exc()
            logging.warning('Failed to read & filter {}'.format(str(obj)))


def shuffle(data, shuffle_size=2500):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, wav/feat, spk}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, wav/feat, spk}]
    """
    # Use a deterministic RNG seeded from the worker's initial random state.
    # This makes the local shuffle reproducible regardless of how many random
    # calls other pipeline stages have made.
    rng = random.Random(random.getrandbits(64))
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            rng.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    rng.shuffle(buf)
    for x in buf:
        yield x


def spk_to_id(data, spk2id):
    """ Parse spk id

        Args:
            data: Iterable[{key, wav/feat, spk}]
            spk2id: Dict[str, int]

        Returns:
            Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert 'spk' in sample
        if sample['spk'] in spk2id:
            label = spk2id[sample['spk']]
        else:
            label = -1
        sample['label'] = label
        yield sample


def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.
        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate
        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        yield sample


def speed_perturb(data, num_spks):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    speeds = [1.0, 0.9, 1.1]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed_idx = sample.pop('speed_idx', random.randint(0, 2))
        if speed_idx > 0:
            wav, _ = torchaudio.functional.speed(
                waveform, sample_rate, speeds[speed_idx])
            sample['wav'] = wav
            sample['label'] = sample['label'] + num_spks * speed_idx

        yield sample


def get_random_chunk(data, chunk_len):
    """ Get random chunk

        Args:
            data: torch.Tensor (random len)
            chunk_len: chunk length

        Returns:
            torch.Tensor (exactly chunk_len)
    """
    data_len = len(data)
    data_shape = data.shape
    # random chunk
    if data_len >= chunk_len:
        chunk_start = random.randint(0, data_len - chunk_len)
        data = data[chunk_start:chunk_start + chunk_len]
        # re-clone the data to avoid memory leakage
        if type(data) == torch.Tensor:
            data = data.clone()
        else:  # np.array
            data = data.copy()
    else:
        # padding
        repeat_factor = chunk_len // data_len + 1
        repeat_shape = repeat_factor if len(data_shape) == 1 else (
            repeat_factor, 1)
        if type(data) == torch.Tensor:
            data = data.repeat(repeat_shape)
        else:  # np.array
            data = np.tile(data, repeat_shape)
        data = data[:chunk_len]

    return data




def random_chunk(data, chunk_len, data_type='shard/raw/feat'):
    """ Random chunk the data into chunk_len

        Args:
            data: Iterable[{key, wav/feat, label}]
            chunk_len: chunk length for each sample

        Returns:
            Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert 'key' in sample

        if data_type == 'feat':
            assert 'feat' in sample
            feat = sample['feat']
            feat = get_random_chunk(feat, chunk_len)
            sample['feat'] = feat
        else:
            assert 'wav' in sample
            wav = sample['wav'][0]
            wav = get_random_chunk(wav, chunk_len)
            sample['wav'] = wav.unsqueeze(0)
        yield sample

def norm_rir(rir_audio):
    return rir_audio / np.sqrt(np.sum(rir_audio**2))



def apply_augmentor(data, augmentor, resample_rate=16000):
    """Apply a deserialized Augmentor (from dataset.augs) to each sample.

    Bridges the Sequential/OneOf augmentor tree (used in distillation) into
    the standard training data pipeline. The augmentor mutates sample['wav']
    in place via a {'wav', 'sample_rate'} dict view.
    """
    for sample in data:
        ex = {'wav': sample['wav'],
              'sample_rate': sample.get('sample_rate', resample_rate)}
        ex = augmentor(ex)
        sample['wav'] = ex['wav']
        yield sample







def add_reverb_noise_gain(data,
                          reverb_source,
                          noise_source,
                          resample_rate=16000,
                          aug_prob=0.6,
                          aug_setup='default',
                          gain_prob=1.0,
                          gain_db_range=(-18.0, 6.0),
                          speech_rms_mode='full',
                          peak_protect=True,
                          noise_categories=None,
                          eps=1e-12):
    """Reverb -> additive noise at target SNR -> random gain. No output RMS norm.

    Level-faithful augmentation chain for training without input z-norm
    (``spec_params.norm_signal: false``). The absolute level reaching the
    model is never renormalized: the training-level distribution is the
    natural utterance levels spread by the final random gain.

    Per sample, with probability ``aug_prob``, one of {reverb+noise, reverb,
    noise} is applied:

    1. Reverb: convolution with a unit-energy RIR. Only the *filter* is
       normalized; the convolved output is left as-is.
    2. Additive noise: scaled to a target SNR measured against the *current*
       speech (i.e. after reverb, matching what the model hears) and added.
       The mixture is not rescaled afterwards, so the sampled SNR
       distribution is preserved exactly.

    Independently of the above, with probability ``gain_prob`` a random gain
    (uniform in dB over ``gain_db_range``) is applied to every sample,
    augmented or not. A uniform scale moves the absolute level but not SNR.

    ``peak_protect`` rescales the final signal only if its peak exceeds 1.0
    (real capture cannot exceed full scale); uniform rescale preserves SNR.

    Args:
        data: Iterable[{key, wav, label, sample_rate}]
        reverb_source / noise_source: LMDB sources. noise_source must support
            category sampling (LmdbDataMap) so SNR ranges match noise types.
        aug_setup: SNR preset, 'default' | 'medium' | 'hard'.
        speech_rms_mode: 'full' = power over the whole chunk (same SNR
            semantics as the older presets); 'active' = power over 25 ms
            frames within 40 dB of the loudest frame (active-speech RMS).

    Returns:
        Iterable[{key, wav, label, sample_rate, applied_augs}]
    """
    noisesnr = {
        'default': {'noise': [0, 15], 'music': [5, 15], 'speech': [13, 20], 'bubble': [13, 20]},
        'medium':  {'noise': [0, 12], 'music': [3, 15], 'speech': [5, 18],  'bubble': [5, 18]},
        'hard':    {'noise': [0, 12], 'music': [1, 12], 'speech': [2, 12],  'bubble': [2, 12]},
    }
    # [reverb+noise, reverb-only, noise-only]
    aug_type_prob = {
        'default': [0.25, 0.1, 0.65],
        'medium':  [0.25, 0.1, 0.65],
        'hard':    [0.25, 0.1, 0.65],
    }
    numnoise = {'noise': [1, 1], 'speech': [1, 1], 'bubble': [3, 7], 'music': [1, 1]}

    if noise_categories is not None:
        allowed = set(noise_categories)
        noisesnr = {k: {c: v for c, v in cats.items() if c in allowed}
                    for k, cats in noisesnr.items()}
        numnoise = {c: v for c, v in numnoise.items() if c in allowed}

    assert aug_setup in noisesnr, f"Unknown preset: {aug_setup}"
    assert speech_rms_mode in ('full', 'active'), speech_rms_mode
    assert gain_db_range[0] <= gain_db_range[1], gain_db_range

    frame_len = max(1, int(0.025 * resample_rate))

    def _full_power(x):
        return float(np.dot(x, x) / max(x.shape[0], 1))

    def _speech_power(x):
        if speech_rms_mode != 'active':
            return _full_power(x)
        n = x.shape[0] // frame_len
        if n < 2:
            return _full_power(x)
        fp = np.mean(x[:n * frame_len].reshape(n, frame_len) ** 2, axis=1)
        pmax = float(fp.max())
        if pmax <= 0.0:
            return 0.0
        return float(fp[fp >= pmax * 1e-4].mean())



    for sample in iter(data):

        rnd_ctx = sample.pop('rnd_ctx', nullcontext())

        with rnd_ctx:
            assert 'wav' in sample
            assert 'key' in sample
            curr_aug_setup = []
            audio = sample['wav'].numpy()[0]
            audio_len = audio.shape[0]
            modified = False

            if aug_prob > random.random():
                aug_type = random.choices([0, 1, 2],
                                        weights=aug_type_prob[aug_setup], k=1)[0]

                if aug_type in [0, 1]:
                    # Reverb first: the additive noise below models noise at the
                    # receiver, so it must not pass through the room filter.
                    rir_key, rir_data = reverb_source.random_one()
                    rir_sr, rir_audio = wavfile.read(io.BytesIO(rir_data))
                    rir_audio = rir_audio.astype(np.float32)
                    if rir_sr != resample_rate:
                        rir_audio = signal.resample(
                            rir_audio,
                            int(len(rir_audio) / rir_sr * resample_rate))
                    rir_audio = rir_audio / np.sqrt(np.sum(rir_audio**2) + eps)
                    audio = signal.convolve(audio, rir_audio,
                                            mode='full')[:audio_len]
                    modified = True
                    curr_aug_setup.append(dict(aug='reverb', rir_key=rir_key))

                if aug_type in [0, 2]:
                    cat = random.choice(list(numnoise.keys()))
                    sample_cat = 'speech' if cat == 'bubble' else cat
                    min_num, max_num = numnoise[cat]
                    if min_num < max_num:
                        num_noises = np.random.randint(min_num, max_num)
                    else:
                        num_noises = max_num

                    full_noise = []
                    for _ in range(num_noises):
                        _, noise_data = noise_source.random_one(sample_cat)
                        noise_sr, noise_audio = wavfile.read(io.BytesIO(noise_data))
                        # Chunk in the file's native dtype BEFORE float conversion:
                        # decoding whole files (minutes of music) to float32 was
                        # the chain's dominant cost (profiling/profile_augs_v2.py).
                        if noise_sr != resample_rate:
                            noise_audio = get_random_chunk(
                                noise_audio,
                                int(audio_len / resample_rate * noise_sr))
                            noise_audio = noise_audio.astype(np.float32) / (1 << 15)
                            noise_audio = signal.resample(noise_audio, audio_len)
                        else:
                            noise_audio = get_random_chunk(noise_audio, audio_len)
                            noise_audio = noise_audio.astype(np.float32) / (1 << 15)
                        full_noise.append(noise_audio)
                    if num_noises > 1:
                        noise_audio = np.mean(np.array(full_noise), axis=0)
                    else:
                        noise_audio = full_noise[0]

                    snr_db = random.uniform(*noisesnr[aug_setup][cat])
                    speech_power = _speech_power(audio)
                    noise_power = _full_power(noise_audio)
                    if speech_power > 0.0 and noise_power > 0.0:
                        # float(): a np.float64 scalar would promote the mixture
                        # and everything downstream to float64
                        scale = float(np.sqrt(
                            10**(-snr_db / 10) * speech_power / noise_power))
                        audio = audio + scale * noise_audio
                        modified = True
                        curr_aug_setup.append(dict(aug='noise', noise_cat=cat,
                                                snr_db=float(snr_db),
                                                scale=scale))

            if gain_prob > random.random():
                gain_db = random.uniform(gain_db_range[0], gain_db_range[1])
                audio = audio * (10.0**(gain_db / 20.0))
                modified = True
                curr_aug_setup.append(dict(aug='gain', gain_db=float(gain_db)))

            if peak_protect and modified:
                peak = float(np.max(np.abs(audio)))
                if peak > 1.0:
                    audio = audio * (0.99 / peak)
                    curr_aug_setup.append(dict(aug='peak_protect',
                                            peak=float(peak)))

            if modified:
                sample['wav'] = torch.from_numpy(audio).unsqueeze(0).float()
            sample['applied_augs'] = json.dumps(curr_aug_setup)
            yield sample
