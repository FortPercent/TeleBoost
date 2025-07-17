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
import torch
from megatron.core import mpu
from .samplers import build_sampler 
from teletron.datasets.collators import DefaultCollator
from teletron.utils import (
    print_rank_0,
    get_args,
    set_config,
)
from teletron.train.utils import (
    get_train_valid_test_num_samples,
)
from teletron.core.parallel_state import get_transformer_model_group
from teletron.datasets.hunyuanvideo_dataset_builder import (
    HunyuanVideoDatasetBuilder,
    HunyuanVideoDatasetConfig,
)



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


def build_train_valid_test_datasets(dp_rank=None, dp_size=None):
    """Build pretraining datasets."""
    train_valid_test_num_samples = get_train_valid_test_num_samples()
    print_rank_0(' > datasets target sizes (minimum size):')
    print_rank_0('    train:      {}'.format(train_valid_test_num_samples[0]))
    print_rank_0('    validation: {}'.format(train_valid_test_num_samples[1]))
    print_rank_0('    test:       {}'.format(train_valid_test_num_samples[2]))
    args = get_args()

    print_rank_0("> building train, validation, and test datasets for multimodal ...")

    if args.dataset_type == "FakeDataset" or args.dataset_type == "KoalaDataset":
        train_ds = build_dataset(args.dataset_type)
        valid_ds = None
        test_ds = None
    elif args.dataset_type == "VastDataset": 
        global_config = set_config()
        transformer_group = get_transformer_model_group()
        if args.temp_accelerate:
            if transformer_group is not None:
                return  None, None, None
            else:
                import os
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                global_rank = int(os.environ.get("RANK", 0))
                world_size = int(os.environ.get("WORLD_SIZE", 1))
                all_data_paths = global_config.dataset.data_path_list
                num_samples = len(all_data_paths)
                samples_per_rank = num_samples // args.distributed_vae_world_size
                start_index = (global_rank - args.dit_world_size) * samples_per_rank
                if global_rank == world_size - 1:
                    end_index = num_samples
                else:
                    end_index = start_index + samples_per_rank
                local_data_paths = all_data_paths[start_index:end_index]
                global_config.dataset.data_path_list = local_data_paths
                print(f"rank:{global_rank}: {local_data_paths}")
        
        train_ds_config = global_config
        eval_ds_config = global_config.get("eval", None)
       
        ds_config = HunyuanVideoDatasetConfig(
            train_ds_config=train_ds_config,
            eval_ds_config=eval_ds_config
        )
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
            
        train_ds, valid_ds, test_ds = HunyuanVideoDatasetBuilder(
            dataset,
            dataset_eval,
            train_valid_test_num_samples,
            lambda: True,
            ds_config,
        ).build()
    elif args.dataset_type == "TensorDataset":
        global_config = set_config()
        train_ds_config = global_config.dataloaders.train
        train_ds = build_dataset(train_ds_config)
        valid_ds = None
        test_ds = None
        #     pass

    elif args.dataset_type == "BucketDataset": 
        global_config = set_config()
        assert global_config.sampler.type == "BucketVariableBatchSampler"
        assert args.dataloader_type == 'external', "BucketDataset use cumstomed dataloader"
        assert args.task_type == "wan_i2v_bucket", "BucketDataset is only supported for t2i_wanvae task"
        print_rank_0("Warning: The `args.micro_batch_size` and `seed` from dataset config will NOT BE USED when use BucketDataset.")

        dataset = build_dataset(global_config.dataset)
        if dp_rank is None or dp_size is None:
            sampler = build_sampler(
                global_config.sampler,
                dataset=dataset, 
                rank=mpu.get_data_parallel_rank(),
                num_replicas=mpu.get_data_parallel_world_size(),
                seed=args.seed
            )
        else:
            sampler = build_sampler(
                global_config.sampler,
                dataset=dataset, 
                rank=dp_rank,
                num_replicas=dp_size,
                seed=args.seed
            )
        # 使用 iteration 和 num_replicas / world size 计算，近似最后访问的索引
        if args.last_microbatch_size_index is None:
            sampler.last_micro_batch_access_index = args.iteration * sampler.num_replicas
        else:
            sampler.last_micro_batch_access_index = args.last_microbatch_size_index
        # else:
        # sampler.last_micro_batch_access_index = iteration * sampler.num_replicas

        collator = DefaultCollator(is_equal=True)
        train_ds = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=global_config.dataloader.num_workers
        )
        train_ds = iter(train_ds)
        valid_ds = None
        test_ds = None
    else:
        raise NotImplementedError

    print_rank_0("> finished creating multimodal datasets ...")

    return train_ds, valid_ds, test_ds