from megatron.training import (
    get_args,
    print_rank_0
)
from vast.train.configs.config import load_config
from vast.datasets.datasets.build import build_dataset as build_dataset_vast
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
        eval_ds_config = global_config.dataloaders.eval
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
        train_ds_config = global_config
        ds_config = HunyuanVideoDatasetConfig(
            train_ds_config=train_ds_config,
        )
        dataset = build_dataset_vast(train_ds_config.dataset)

        train_ds, valid_ds, test_ds = HunyuanVideoDatasetBuilder(
            dataset,
            train_val_test_num_samples,
            lambda: True,
            ds_config,
        ).build()
    else:
        raise NotImplementedError

    print_rank_0("> finished creating multimodal datasets ...")

    return train_ds, valid_ds, test_ds
