from typing import Dict

import segmentation_models_pytorch as smp
from torch import nn


def init_network(params: Dict) -> nn.Module:
    return smp.create_model(**params)
