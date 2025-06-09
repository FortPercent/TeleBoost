import torch
from torch.utils.data import BatchSampler
from megatron.training import (
    get_args,
    print_rank_0
)
from megatron.core import mpu
from vast.datasets import DefaultCollator
from vast.train.configs.config import load_config
from vast.datasets.datasets.build import build_dataset as build_dataset_vast
from vast.train.samplers import build_sampler as build_sampler_vast
from teletron.datasets.build import build_dataset
from teletron.datasets.vast_dataset.hunyuan_dataset_config import HunyuanVideoDatasetConfig
from teletron.datasets.vast_dataset.hunyuanvideo_dataset_builder import HunyuanVideoDatasetBuilder



def load_config_vast():
    args = get_args()
    if args.task_type == "t2v":
        print("loading t2v config")
        from config.hunyuanvideo_t2v import config
    elif args.task_type == "i2v":
        print("loading i2v config")
        from config.hunyuanvideo_i2vhy import config 
    elif args.task_type == "i2v_multimask":
        print("loading i2v_multimask config")
        from config.hunyuanvideo_i2v_multimask import config
    elif args.task_type == "i2vhy_token_replace":
        print("loading i2vhy_token_replace config")
        from config.hunyuanvideo_i2vhy_token_replace import config
    elif args.task_type == "t2i_wanvae": 
        print("loading t2i_wanvae config")
        from config.hunyuanvideo_t2i_wanvae import config
    elif args.task_type == "wan_flf":
        from config.wan_flf import config
    elif args.task_type == "wan_i2v_prone":
        from config.prone10_lowerlr import config
    else:
        return None
    config_vast = load_config(config)
    return config_vast


def train_valid_test_datasets_provider(train_val_test_num_samples):

    args = get_args()

    print_rank_0("> building train, validation, and test datasets for multimodal ...")

    if args.dataset_type == "FakeDataset" or args.dataset_type == "KoalaDataset":
        train_ds = build_dataset(args.dataset_type)
        valid_ds = None
        test_ds = None
    elif args.dataset_type == "VastDataset": 
        global_config = load_config_vast()
        train_ds_config = global_config.dataloaders.train
        eval_ds_config = global_config.dataloaders.get("eval", None)
        ds_config = HunyuanVideoDatasetConfig(
            train_ds_config=train_ds_config,
            eval_ds_config=eval_ds_config
        )
        dataset = build_dataset_vast(train_ds_config.dataset)
        train_ds, valid_ds, test_ds = HunyuanVideoDatasetBuilder(
            dataset,
            train_val_test_num_samples,
            lambda: True,
            ds_config,
        ).build()

    elif args.dataset_type == "BucketDataset": 
        global_config = load_config_vast()
        assert global_config.sampler.type == "BucketVariableBatchSampler"
        assert args.dataloader_type == 'external', "BucketDataset use cumstomed dataloader"
        assert args.task_type == "t2i_wanvae", "BucketDataset is only supported for t2i_wanvae task"
        print_rank_0("Warning: The `args.micro_batch_size` and `seed` from vast dataset config will NOT BE USED when use BucketDataset.")

        dataset = build_dataset_vast(global_config.dataset)
        sampler = build_sampler_vast(
            global_config.sampler,
            dataset=dataset, 
            rank=mpu.get_data_parallel_rank(),
            num_replicas=mpu.get_data_parallel_world_size(),
            seed=args.seed
        )
        # 使用 iteration 和 num_replicas / world size 计算，近似最后访问的索引
        sampler.last_micro_batch_access_index = args.iteration * sampler.num_replicas

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
