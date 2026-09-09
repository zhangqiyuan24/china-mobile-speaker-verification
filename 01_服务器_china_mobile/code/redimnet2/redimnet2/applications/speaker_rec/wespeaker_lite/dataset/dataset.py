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

import random
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from wespeaker_lite.utils.file_utils import read_lists
from wespeaker_lite.dataset.lmdb_data import LmdbData, LmdbDataMap
import wespeaker_lite.dataset.processor as processor
from wespeaker_lite.dataset.processor import read_audio_section


class Processor(IterableDataset):

    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        try:
            self.source.set_epoch(epoch)
        except:
            pass

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)

    def __len__(self):
        try:
            return len(self.source)
        except:
            return None


class DistributedSampler:

    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data


class DataList(IterableDataset):

    def __init__(self,
                 lists,
                 shuffle=True,
                 partition=True,
                 repeat_dataset=True):
        self.lists = lists
        self.repeat_dataset = repeat_dataset
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __len__(self):
        return len(self.lists)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        if not self.repeat_dataset:
            for index in indexes:
                data = dict(src=self.lists[index])
                data.update(sampler_info)
                yield data
        else:
            indexes_len = len(indexes)
            counter = 0
            # Base seed for deterministic re-seeding at each data pass.
            # Derived from the worker's initial random state (set by seed_worker).
            _base_seed = random.getrandbits(64)
            _pass = 0
            while True:
                if counter % indexes_len == 0:
                    # Re-seed random/np at each pass over the data so that
                    # augmentation/shuffle stay deterministic regardless of
                    # prefetch timing between workers.
                    random.seed(_base_seed + _pass)
                    np.random.seed((_base_seed + _pass) % (2**31))
                    _pass += 1
                index = indexes[counter % indexes_len]
                counter += 1
                data = dict(src=self.lists[index])
                data.update(sampler_info)
                yield data


def Dataset(data_type,
            data_list_file,
            configs,
            spk2id_dict,
            verbose=False,
            whole_utt=False,
            reverb_lmdb_file=None,
            noise_lmdb_file=None,
            repeat_dataset=True):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw/feat file level. The second is local shuffle
        at training samples level.

        Args:
            data_type(str): shard/raw/raw_fast
            data_list_file: data list file
            configs: dataset configs
            spk2id_dict: spk2id dict
            reverb_lmdb_file: reverb data source lmdb file
            noise_lmdb_file: noise data source lmdb file
            whole_utt: use whole utt or random chunk
    """
    assert data_type == 'raw_fast'
    # Global shuffle
    lists = read_lists(data_list_file)
    shuffle = configs.get('shuffle', False)
    dataset = DataList(lists, shuffle=shuffle, repeat_dataset=repeat_dataset)

    num_frms = configs.get('num_frms', 200)
    frame_shift = configs['fbank_args'].get('frame_shift', 10)
    frame_length = configs['fbank_args'].get('frame_length', 25)
    resample_rate = configs.get('resample_rate', 16000)
    chunk_len = ((num_frms - 1) * frame_shift +
                 frame_length) * resample_rate // 1000
    filter_conf = configs.get('filter_args', {})
    dataset = Processor(dataset, processor.parse_raw_fast_and_filter,
                        chunk_len, data_type, **filter_conf)

    # Local shuffle
    if shuffle:
        dataset = Processor(dataset, processor.shuffle,
                            **configs['shuffle_args'])

    # spk2id
    dataset = Processor(dataset, processor.spk_to_id, spk2id_dict)

    # resample
    resample_rate = configs.get('resample_rate', 16000)
    dataset = Processor(dataset, processor.resample, resample_rate)
    # speed perturb
    speed_perturb_flag = configs.get('speed_perturb', True)
    if speed_perturb_flag:
        dataset = Processor(dataset, processor.speed_perturb,
                            len(spk2id_dict))
    if (not whole_utt):
        # random chunk
        num_frms = configs.get('num_frms', 200)
        frame_shift = configs['fbank_args'].get('frame_shift', 10)
        frame_length = configs['fbank_args'].get('frame_length', 25)
        chunk_len = ((num_frms - 1) * frame_shift +
                     frame_length) * resample_rate // 1000
        dataset = Processor(dataset, processor.random_chunk, chunk_len,
                            data_type)

    # ── Standard augmentation ──
    if configs.get("aug_setup", None) is not None:
        aug_cfg = configs['aug_setup']
        # New format: composable Sequential/OneOf tree from dataset.augs.
        # Detected by presence of 'aug_type' at the root (vs legacy 'type').
        if 'aug_type' in aug_cfg:
            from wespeaker_lite.dataset.augs import deserialize_augmentor
            augmentor = deserialize_augmentor(aug_cfg)
            print(f"AUG_TYPE : augmentor_chain -> {repr(augmentor)}")
            dataset = Processor(dataset, processor.apply_augmentor,
                                augmentor, resample_rate)
            return dataset

        # Legacy string format: both 'wespeaker_augs' and 'wespeaker_augs_v2'
        # use the level-faithful chain (reverb -> noise@SNR -> random gain, no
        # RMS norm) via add_reverb_noise_gain with categorized LMDB sources.
        aug_type = aug_cfg['type']
        print(f"AUG_TYPE : {aug_type}")
        reverb_data = LmdbDataMap(aug_cfg['reverb_lmdb_file'])
        noise_data = LmdbDataMap(aug_cfg['noise_lmdb_file'])
        dataset = Processor(dataset, processor.add_reverb_noise_gain,
                            reverb_data, noise_data, resample_rate,
                            aug_cfg.get('aug_prob', 0.6),
                            aug_cfg.get('preset', 'default'),
                            gain_prob=aug_cfg.get('gain_prob', 1.0),
                            gain_db_range=tuple(
                                aug_cfg.get('gain_db_range', (-18.0, 6.0))),
                            speech_rms_mode=aug_cfg.get(
                                'speech_rms_mode', 'full'),
                            peak_protect=aug_cfg.get('peak_protect', True),
                            noise_categories=aug_cfg.get(
                                'noise_categories', None))

    return dataset

def collate_func_single(batch):
    assert len(batch) == 1
    (x0,x1),y = batch[0]
    return (x0,x1),y

class Vox1TestEvalDataset(torch.utils.data.Dataset):
    def __init__(self,
                    vox1_test_root,
                    wavs_subroot='wav',
                    vox1_veri_test2_list=None,
                    samplerate = 16000,
                    chunk_len_sec = 3.0,
                    num_chunks = 5
                ):
        self.vox1_test_root = Path(vox1_test_root)
        self.wavs_subroot = wavs_subroot

        if vox1_veri_test2_list is None:
            vox1_veri_test2_list = self.vox1_test_root/'veri_test2.txt'

        vox1_protocol = pd.read_csv(vox1_veri_test2_list,sep=" ",
                                         names=['label','enroll','verify'])

        trials = vox1_protocol['enroll'].values.tolist() + vox1_protocol['verify'].values.tolist()
        self.trials = sorted(list(set(trials)))
        assert all([(self.vox1_test_root/self.wavs_subroot/rlp).exists() for rlp in self.trials])
        trials_ind_map = {key:ind for ind, key in enumerate(self.trials)}

        vox1_protocol['enroll_ind'] = vox1_protocol.enroll.map(trials_ind_map)
        vox1_protocol['verify_ind'] = vox1_protocol.verify.map(trials_ind_map)
        self.vox1_protocol = vox1_protocol

        self.samplerate = samplerate
        self.chunk_len_sec = chunk_len_sec
        self.num_chunks = num_chunks

    def __len__(self):
        return len(self.trials)

    @staticmethod
    def slice_samples_into_chunks(
        samples,
        samplerate=16_000,
        chunk_len_sec=3.0,
        num_chunks=5
    ):
        chunk_len = int(chunk_len_sec*samplerate)
        chunked = []
        for shift in np.linspace(0, len(samples) - chunk_len, num_chunks).astype(int):
            chunked.append(samples[shift: shift + chunk_len])
        return np.stack(chunked)

    def __getitem__(self,index):
        path = self.vox1_test_root/self.wavs_subroot/self.trials[index]
        label = str(path.relative_to(self.vox1_test_root))

        samples, sr = sf.read(path)
        assert sr == self.samplerate
        chunks = self.slice_samples_into_chunks(samples,
                    samplerate=self.samplerate,
                    chunk_len_sec=self.chunk_len_sec,
                    num_chunks=self.num_chunks)

        x0 = torch.from_numpy(samples).float()[None,:]
        x1 = torch.from_numpy(chunks).float()
        y = torch.tensor(index).long().unsqueeze(0)

        return (x0,x1), y

def collate_func_mult(batch):
    tbatch = tuple(zip(*batch))
    return [torch.stack(xyz,dim=0) for xyz in tbatch]

class Vox1TestEvalDatasetSimple(torch.utils.data.Dataset):
    def __init__(self,
                    vox1_test_root,
                    wavs_subroot='wav',
                    vox1_veri_test2_list=None,
                    samplerate = 16000,
                    duration = 8.0
                ):
        self.vox1_test_root = Path(vox1_test_root)
        self.wavs_subroot = wavs_subroot
        if vox1_veri_test2_list is None:
            vox1_veri_test2_list = self.vox1_test_root/'veri_test2.txt'

        vox1_protocol = pd.read_csv(vox1_veri_test2_list,sep=" ",
                                         names=['label','enroll','verify'])

        trials = vox1_protocol['enroll'].values.tolist() + vox1_protocol['verify'].values.tolist()
        self.trials = sorted(list(set(trials)))
        assert all([(self.vox1_test_root/self.wavs_subroot/rlp).exists() for rlp in self.trials])
        trials_ind_map = {key:ind for ind, key in enumerate(self.trials)}

        vox1_protocol['enroll_ind'] = vox1_protocol.enroll.map(trials_ind_map)
        vox1_protocol['verify_ind'] = vox1_protocol.verify.map(trials_ind_map)
        self.vox1_protocol = vox1_protocol

        self.samplerate = samplerate
        self.duration = duration

    def __len__(self):
        return len(self.trials)

    def __getitem__(self,index):
        path = self.vox1_test_root/self.wavs_subroot/self.trials[index]
        label = str(path.relative_to(self.vox1_test_root))

        samples, sr = read_audio_section(path,
            samplerate=self.samplerate,
            duration=self.duration,
            min_duration=0.0,
            start_time=0.0,
            extend_to_duration=self.duration,
        )
        x = torch.from_numpy(samples).float()[None,:]
        y = torch.tensor(index).long().unsqueeze(0)

        return x, y
