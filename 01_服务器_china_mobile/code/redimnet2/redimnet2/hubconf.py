import os
import sys

sys.path.append(os.path.dirname(__file__))
from redimnet2 import load_custom

dependencies = ['torch', 'torchaudio']


def redimnet2(model_name='b0', train_type='lm', dataset='vox2', pretrained=True):
    """Load a ReDimNet2 speaker embedding model.

    Args:
        model_name: One of 'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6'.
        train_type: 'ptn' (pretraining), 'lm' (large-margin fine-tuning),
            or 'dis' (distilled; available for b6/vb2+vox2_v0).
        dataset: Training dataset, default 'vox2'. Other released values
            include 'vb2+vox2_v0' and 'vb2+vox2+cnc2_v0'.
        pretrained: If True, load pretrained weights.

    Returns:
        ReDimNet2Wrap model.
    """
    if pretrained:
        return load_custom(model_name, train_type=train_type, dataset=dataset)
    raise ValueError("Only pretrained=True is supported. Use ReDimNet2Wrap directly for custom models.")
