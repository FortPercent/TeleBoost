# Copyright (c) 2025, TeleAI-infra Team and NVIDIA CORPORATION. All rights reserved.

"""General utilities."""

import torch

from megatron.training import (
    get_args,
    print_rank_0
)
from megatron.core import mpu
from typing import get_origin
from vast.train.configs.config import load_config
from vast.datasets.datasets.build import build_dataset as build_dataset_vast
from teletron.datasets.build import build_dataset
from teletron.datasets.vast_dataset.hunyuan_dataset_config import HunyuanVideoDatasetConfig
from teletron.datasets.vast_dataset.hunyuanvideo_dataset_builder import HunyuanVideoDatasetBuilder


def get_batch_on_this_tp_rank_vast(data_iterator):
    args = get_args()
    data_dict = ['images', 'prompt_embeds', 'prompt_masks', 'clip_text_embed', 'ref_mask', 'ref_images']

    def _broadcast(item):
        if item is not None:
           torch.distributed.broadcast(item, mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:
        if data_iterator is not None:
           data = next(data_iterator)
        else:
           data = None
           
        batch = {}
        data_dict = [d for d in data.keys()]
        for param in data_dict:
            if isinstance(data[param], list): # prompt is list
                pass
            elif isinstance(data[param], torch.Tensor):
                batch.update({param: data[param].cuda(non_blocking = True)})
            else:
                raise NotImplementedError(f"Unsupported data type: {type(data[param])}")


        # Step 1: 保存每部分的大小信息（只在 Rank 0 执行）
        sizes_info = {key: tensor.size() if tensor is not None else None for key, tensor in batch.items()}
        # Step 2: 广播大小信息
        sizes_info = torch.distributed.broadcast_object_list([sizes_info],mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())

        for param in batch.keys():
            _broadcast(batch[param])

    else:
        sizes_info = None 
        sizes_info_list = [sizes_info]
        torch.distributed.broadcast_object_list(sizes_info_list,mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())

        batch = {}
        data_dict = [d for d in sizes_info_list[0].keys()]
        for param in data_dict:
            if param == "prompt_masks":
                batch. update({param: torch.empty(sizes_info_list[0][param], dtype=torch.int64, device=torch.cuda.current_device())})
            else:
                batch. update({param: torch.empty(sizes_info_list[0][param], dtype=torch.float32, device=torch.cuda.current_device())})
        for param in batch.keys():
            _broadcast(batch[param])

    return batch


def get_batch_on_this_tp_cp_rank_vast(data_iterator):
    args = get_args()
    data_dict = {
        'images': torch.float32,
        'prompt_embeds': torch.float32,
        'prompt_masks': torch.int64,
        'clip_text_embed': torch.float32,
        'ref_mask': torch.float32,
        'ref_images': torch.float32,
        'struct_prompt': list[str],
        'short_prompt': list[str],
        'dense_prompt': list[str],
        'prompt': list[str],
        'first_ref_image': torch.float32,
        'latents': torch.bfloat16
    }

    def _broadcast(item):
        if item is not None:
           torch.distributed.broadcast(item, mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())

    def _broadcast_object_list(item):
        if item is not None:
           torch.distributed.broadcast_object_list(item, mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
    
    if mpu.get_tensor_context_parallel_rank() == 0:
        if data_iterator is not None:
           data = next(data_iterator)
        else:
           data = None
        assert all(key in data_dict for key in data.keys()), f"Not all keys from input valid: {set(data.keys()) - set(data_dict)}"
        
        batch = {}
        for param in data.keys():
            dtype = data_dict[param]
            if get_origin(dtype) is list:
                assert isinstance(data[param], list), f"{param} is not list"
                # 字符串列表更新内容本身
                batch.update({param: data[param]})
            elif isinstance(dtype, torch.dtype):
                assert data[param].dtype == dtype, f"{param} is not of type {dtype}"
                # torch Tensor更新前先传输到CUDA
                batch.update({param: data[param].cuda(non_blocking = True)})
            else:
                raise NotImplementedError(f"Unsupported data type: {type(data[param]), dtype}")

        # Step 1: 保存每部分的大小信息（只在 Rank 0 执行）
        sizes_info = {key: tensor.size() if (tensor is not None and isinstance(tensor, torch.Tensor)) else None for key, tensor in batch.items()}
        # Step 2: 广播大小信息
        _broadcast_object_list([sizes_info])
        for param in data.keys():
            dtype = data_dict[param]
            if get_origin(dtype) is list: 
                # 字符串以object list的形式广播
                _broadcast_object_list(batch[param])
            elif isinstance(dtype, torch.dtype):
                _broadcast(batch[param])

    else:
        # TODO check 
        sizes_info = None 
        sizes_info_list = [sizes_info]
        _broadcast_object_list(sizes_info_list)

        batch = {}
        for param in sizes_info_list[0].keys():
            dtype = data_dict[param]
            if get_origin(dtype) is list: 
                # 需要注意，列表需要有一个默认值占位，否则广播失败
                batch.update({param: ['']})
            elif isinstance(dtype, torch.dtype):
                batch.update({param: torch.empty(sizes_info_list[0][param], dtype=dtype, device=torch.cuda.current_device())})

        for param in batch.keys():
            dtype = data_dict[param]
            if get_origin(dtype) is list: 
                # 以object list的形式广播字符串列表
                _broadcast_object_list(batch[param])
            elif isinstance(dtype, torch.dtype):
                _broadcast(batch[param])

    return batch



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
