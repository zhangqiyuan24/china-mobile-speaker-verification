# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn

import redimnet2

def get_speaker_model(model_name: str):
    if model_name.startswith("ReDimNet2"):
        return getattr(redimnet2, model_name)
    else:  # model_name error !!!
        print(model_name + " not found !!!")
        exit(1)

class SpeakerModel(nn.Module):
    def __init__(self,backbone,projection):
        super().__init__()
        self.backbone = backbone
        self.projection = projection

    def forward(self, x, **kwargs):
        return self.backbone(x, **kwargs)

    def extract_features(self, x):
        return self.backbone.extract_features(x)

    def extract_embedding(self, features, mask=None):
        return self.backbone.extract_embedding(features, mask=mask)
