# Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
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
import pickle

import lmdb
from collections import defaultdict
from pathlib import Path

class LmdbData:
    def __init__(self, lmdb_file):
        self.db = lmdb.open(lmdb_file,
                            readonly=True,
                            lock=False,
                            readahead=False)
        with self.db.begin(write=False) as txn:
            obj = txn.get(b'__keys__')
            assert obj is not None
            self.keys = pickle.loads(obj)
            assert isinstance(self.keys, list)

    def random_one(self, dummy=None):
        assert len(self.keys) > 0
        index = random.randint(0, len(self.keys) - 1)
        key = self.keys[index]
        with self.db.begin(write=False) as txn:
            value = txn.get(key.encode())
            assert value is not None
        return key, value

    def __del__(self):
        self.db.close()

class LmdbDataMap:
    def __init__(self, lmdb_file):
        self.db = lmdb.open(lmdb_file,
                            readonly=True,
                            lock=False,
                            readahead=False)
        with self.db.begin(write=False) as txn:
            obj = txn.get(b'__keys__')
            assert obj is not None
            self.keys = pickle.loads(obj)
            assert isinstance(self.keys, list)

            self.keys_cat_dict = defaultdict(list)
            for k in self.keys:
                self.keys_cat_dict[k.split("/")[0]].append(k)
            for k,v in self.keys_cat_dict.items():
                print(f"{k} : {len(v)} noises")

    def random_one(self, rnd_noise_cat=None):
        assert len(self.keys) > 0
        # index = random.randint(0, len(self.keys) - 1)
        # key = self.keys[index]
        # self.keys_cat_dict

        if rnd_noise_cat is None:
            rnd_noise_cat = random.choice(list(self.keys_cat_dict.keys()))
        else:
            assert rnd_noise_cat in self.keys_cat_dict
        rnd_noise_ind = random.randint(0, len(self.keys_cat_dict[rnd_noise_cat]) - 1)
        key = self.keys_cat_dict[rnd_noise_cat][rnd_noise_ind]
        with self.db.begin(write=False) as txn:
            value = txn.get(key.encode())
            assert value is not None
        return key, value

    def __del__(self):
        self.db.close()


if __name__ == '__main__':
    import sys
    db = LmdbData(sys.argv[1])
    key, _ = db.random_one()
    print(key)
    key, _ = db.random_one()
    print(key)
