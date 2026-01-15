# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.

from .registry import Registry, build_module
from .clip_dataset import ClipDataset
from .fake_dataset import FakeDataset
from .variable_dataset import VariableClipDataset
from .dpo_dataset import WanDPODataset
import torch
import random
from teletron.datasets.collators import DefaultCollator
from teletron.utils import (
    print_rank_0,
    get_args,
    set_config,
)
from my_utils import get_global_logger
from teletron.train.utils import (
    get_train_valid_test_num_samples,
)
from teletron.core.parallel_state import get_transformer_model_group

DATASETS = Registry()
DATASETS.register_module(ClipDataset)
DATASETS.register_module(FakeDataset)
DATASETS.register_module(VariableClipDataset)
DATASETS.register_module(WanDPODataset)



def build_dataset(params_or_type, *args, **kwargs):
    return build_module(DATASETS, params_or_type, *args, **kwargs)

def build_train_valid_test_datasets(dp_rank=None, dp_size=None, shuffle=False):
    """Build pretraining datasets."""
    args = get_args()

    print_rank_0("> building train, validation, and test datasets for multimodal ...")

    global_config = set_config()
    transformer_group = get_transformer_model_group()

    if transformer_group is not None:
        return  None, None, None
    else:
        import os
        logger = get_global_logger()
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
