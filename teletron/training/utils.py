# Copyright (c) 2025, TeleAI-infra Team and NVIDIA CORPORATION. All rights reserved.

"""General utilities."""

import torch

from megatron.training import (
    get_args
)
from megatron.core import mpu
from typing import get_origin


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
    def _broadcast(item):
        if item is not None:
            import torch.distributed as dist
            rank = dist.get_rank()
            torch.distributed.broadcast(item, mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
    
    if mpu.get_tensor_context_parallel_rank() == 0:
        print("begin data iterator")
        if data_iterator is not None:
           data = next(data_iterator)
        else:
           data = None
        
        sizes_info = {}
        type_info = {}
        batch=dict(data)
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                batch[key] = tensor.to(torch.cuda.current_device())
        for key, tensor in batch.items():
            sizes_info[key] = tensor.size() if tensor is not None and isinstance(tensor, torch.Tensor)  else None
            type_info[key] = tensor.dtype if tensor is not None and isinstance(tensor, torch.Tensor) else type(tensor)

        # Step 2: 广播大小信息
        sizes_info = torch.distributed.broadcast_object_list([sizes_info],mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
        type_info = torch.distributed.broadcast_object_list([type_info],mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())

        for key, tensor in batch.items():
            if isinstance(tensor, list):
                torch.distributed.broadcast_object_list(tensor, mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
            else:
                _broadcast(tensor)

    else:
        sizes_info_list = [None]
        torch.distributed.broadcast_object_list(sizes_info_list,mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
        type_info_list =[None]
        torch.distributed.broadcast_object_list(type_info_list,mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
        
        batch = {}
        for key, value in sizes_info_list[0].items():
            dtype = type_info_list[0][key]
            
            if isinstance(dtype, torch.dtype):  # dtype 是 torch.float32 这种
                tensor = torch.empty(value, dtype=dtype, device=torch.cuda.current_device())
                _broadcast(tensor)
                batch[key] = tensor
            
            else:  # 表示这是个 list 类型对象
                tensor = [None]
                torch.distributed.broadcast_object_list(
                    tensor,
                    src=mpu.get_tensor_context_parallel_src_rank(),
                    group=mpu.get_tensor_context_parallel_group()
                )
                batch[key] = tensor

    return batch