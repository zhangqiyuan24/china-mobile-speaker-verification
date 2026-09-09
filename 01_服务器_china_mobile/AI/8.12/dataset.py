# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
Modified WavSVDataset for Teacher-Student training.

When preprocessor['teacher_student'] is True, __getitem__ returns:
    (feat_clean, feat_aug, spkid)
Otherwise original behavior:
    (feat, spkid)

Usage: Replace speakerlab/dataset/dataset.py
"""

from torch.utils.data import Dataset
from speakerlab.utils.fileio import load_data_csv


class BaseSVDataset(Dataset):
    def __init__(self, data_file: str, preprocessor: dict):
        self.data_points = self.read_file(data_file)
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.data_points)


class WavSVDataset(BaseSVDataset):

    def __getitem__(self, index):
        data = self.get_data(index)
        wav_path = data['path']
        spk = data['spk']

        # Step 1: Read wav (no speed_perturb)
        wav, speed_index = self.preprocessor['wav_reader'](wav_path)

        # Step 2: Encode speaker label
        spkid = self.preprocessor['label_encoder'](spk, speed_index)

        # Step 3: Teacher-Student mode or standard mode
        if self.preprocessor.get('teacher_student', False):
            # ---- Teacher-Student mode ----
            # Clean features (Teacher input)
            feat_clean = self.preprocessor['feature_extractor'](wav)

            # Augmented features (Student input)
            wav_aug = self.preprocessor['augmentations'](wav, spk_label=spk)
            feat_aug = self.preprocessor['feature_extractor'](wav_aug)

            return feat_clean, feat_aug, spkid
        else:
            # ---- Original mode ----
            wav = self.preprocessor['augmentations'](wav, spk_label=spk)
            feat = self.preprocessor['feature_extractor'](wav)

            return feat, spkid

    def get_data(self, index):
        if not hasattr(self, 'data_keys'):
            self.data_keys = list(self.data_points.keys())
        key = self.data_keys[index]
        return self.data_points[key]

    def read_file(self, data_file):
        return load_data_csv(data_file)
