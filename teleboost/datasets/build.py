# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.

from .registry import Registry, build_module
import torch
import random
import logging
from teleboost.datasets.collators import DefaultCollator
from teleboost.utils import (
    print_rank_0,
    get_args,
    set_config,
)
# NOTE: `from teleboost.train.utils import get_train_valid_test_num_samples` and
# `from teleboost.core.parallel_state import get_transformer_model_group` are
# done lazily inside build_train_valid_test_datasets() to break a circular
# import. teleboost/train/__init__.py → trainer.py → dataloader.py → us, so
# importing them at module level here makes `import teleboost.datasets` fail
# with "partially initialized module" when `teleboost.datasets` is loaded
# before `teleboost.train` is fully resolved.

# FakeDataset / FakeDPODataset have no external dep — always available for
# smoke tests / OSS users without an internal data infra checkout.
from .fake_dataset import FakeDataset, FakeDPODataset

# Production datasets depend on `teleai_data_tool` (TeleAI internal data infra:
# lmdb_client, file_client, schema.Clip, etc.). OSS users without that package
# can still `import teleboost`, register their own DPODatasetBase subclass, and
# train via the FakeDataset path or a custom dataset. Production users that
# have teleai_data_tool on PYTHONPATH get the full set automatically.
_OPTIONAL_DATASET_MODULES = {
    "ClipDataset":         (".clip_dataset",     "ClipDataset"),
    "VariableClipDataset": (".variable_dataset", "VariableClipDataset"),
    "WanDPODataset":       (".dpo_dataset",      "WanDPODataset"),
}

DATASETS = Registry()
DATASETS.register_module(FakeDataset)
DATASETS.register_module(FakeDPODataset)

import importlib
_log = logging.getLogger(__name__)
for _name, (_modpath, _attr) in _OPTIONAL_DATASET_MODULES.items():
    try:
        _mod = importlib.import_module(_modpath, package=__package__)
        _cls = getattr(_mod, _attr)
        DATASETS.register_module(_cls)
        # Re-export at package level so existing `from teleboost.datasets.build
        # import WanDPODataset` callers keep working.
        globals()[_name] = _cls
    except ImportError as _e:
        _log.info(f"teleboost.datasets: {_name} unavailable ({_e}); "
                  f"this is expected on OSS installs without teleai_data_tool. "
                  f"FakeDataset and any user-registered datasets remain functional.")



def build_dataset(params_or_type, *args, **kwargs):
    return build_module(DATASETS, params_or_type, *args, **kwargs)

def build_train_valid_test_datasets(dp_rank=None, dp_size=None, shuffle=False):
    """Build pretraining datasets."""
    # Lazy imports — see top-of-file note re: circular import.
    from teleboost.train.utils import get_train_valid_test_num_samples  # noqa: F401
    from teleboost.core.parallel_state import get_transformer_model_group

    args = get_args()

    print_rank_0("> building train, validation, and test datasets for multimodal ...")

    global_config = set_config()
    transformer_group = get_transformer_model_group()

    if transformer_group is not None:
        return  None, None, None
    else:
        import os
        logger = logging.getLogger("teleboost")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        all_data_paths = global_config.dataset.data_path_list
        # shuffle
        if shuffle:
            random.seed(global_config.sampler.seed)
            random.shuffle(all_data_paths)
        num_samples = len(all_data_paths)
        base_samples = (num_samples + args.distributed_vae_world_size -1) // args.distributed_vae_world_size

        # 这段逻辑只负责对于config.dataset.data_path_list进行各个producer的划分
        big_producer_count = args.distributed_vae_world_size - (args.distributed_vae_world_size *  base_samples - num_samples)
        if global_rank < big_producer_count + args.dit_world_size:
            start_idx = (global_rank - args.dit_world_size) * base_samples
            end_idx = start_idx + base_samples
            local_data_paths = all_data_paths[start_idx: end_idx]
            extra_sample = None
        else:
            start_idx = big_producer_count * base_samples + (global_rank - args.dit_world_size - big_producer_count) * (base_samples - 1)
            end_idx = start_idx + base_samples -1
            local_data_paths = all_data_paths[start_idx: end_idx]
            extra_sample = random.choice(all_data_paths[0:big_producer_count * base_samples])
            local_data_paths.append(extra_sample)
            
        global_config.dataset.data_path_list = local_data_paths
        logger.info(
            "[DatasetSplit] rank=%s local_rank=%s world_size=%s data_len=%s "
            "total_paths=%s base_samples=%s range=[%s,%s) assigned=%s extra=%s",
            global_rank,
            local_rank,
            world_size,
            len(local_data_paths),
            num_samples,
            base_samples,
            start_idx,
            end_idx,
            local_data_paths,
            extra_sample,
        )

    train_ds_config = global_config
    eval_ds_config = global_config.get("eval", None)
    dataset = build_dataset(train_ds_config.dataset)
    if eval_ds_config is not None:
        eval_data_list = eval_ds_config.get("data_path_list", None) 
    else:
        eval_data_list = None
    if eval_data_list is not None and len(eval_data_list) > 0:
        train_ds_config.dataset.data_path_list = eval_data_list
        dataset_eval = build_dataset(train_ds_config.dataset)
    else:
        dataset_eval = None

    print("> finished creating multimodal datasets ...")

    return dataset, dataset_eval, None
