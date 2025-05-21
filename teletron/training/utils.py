# Copyright (c) 2025, TeleAI-infra Team and NVIDIA CORPORATION. All rights reserved.

"""General utilities."""

import torch

from megatron.training import (
    get_args,
)
from megatron.core import mpu


def get_batch_on_this_tp_rank_vast(data_iterator):
    args = get_args()

    def _broadcast(item):
        if item is not None:
           torch.distributed.broadcast(item, mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:
        if data_iterator is not None:
           data = next(data_iterator)
        else:
           data = None
           
        batch = {}

        for param in data.keys():
            if isinstance(data[param], torch.Tensor):
                batch.update({param: data[param].cuda(non_blocking = True)})

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
        for param in sizes_info_list[0].keys():
            batch.update({param: torch.empty(sizes_info_list[0][param], dtype=torch.float32, device = torch.cuda.current_device())})
        for param in batch.keys():
            _broadcast(batch[param])

    return batch


def get_batch_on_this_tp_cp_rank_vast(data_iterator):
    args = get_args()

    def _broadcast(item):
        if item is not None:
           torch.distributed.broadcast(item, mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
    
    if mpu.get_tensor_context_parallel_rank() == 0:
        if data_iterator is not None:
           data = next(data_iterator)
        else:
           data = None
        batch = {}
        for param in data.keys():
            if isinstance(data[param], torch.Tensor):
                batch.update({param: data[param].cuda(non_blocking = True)})

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
        for param in sizes_info_list[0].keys():
            batch.update({param: torch.empty(sizes_info_list[0][param], dtype=torch.float32, device = torch.cuda.current_device())})
        for param in batch.keys():
            _broadcast(batch[param])

    return batch