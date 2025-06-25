# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.

from .registry import Registry, build_module
from .fake_dataset import FakeDataset
from .koala_dataset import KoalaDataset
from .dataset import ConcatDataset
from .clip_dataset import ClipDataset
from .variable_clip_dataset import VariableClipDataset
from .variable_image_dataset import VariableImageDataset
from .variable_mix_dataset import VariableMixDataset
from .tensor_dataset import TensorDataset

DATASETS = Registry()
DATASETS.register_module(FakeDataset)
DATASETS.register_module(KoalaDataset)
DATASETS.register_module(ConcatDataset)
DATASETS.register_module(ClipDataset)
DATASETS.register_module(VariableClipDataset)
DATASETS.register_module(VariableImageDataset)
DATASETS.register_module(VariableMixDataset)
DATASETS.register_module(TensorDataset)


def build_dataset(params_or_type, *args, **kwargs):
    return build_module(DATASETS, params_or_type, *args, **kwargs)