import os
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from megatron.core import mpu, tensor_parallel
from teletron.utils import (get_args,)
from teletron.utils.debug_utils import dump_object_summary
from teletron.core.parallel_state import get_comm_pair
from teletron.models.wan.encoder.wan_encoder import WanVideoEncoder
from teletron.models.teleai.teleai_encoder import TeleaiEncoder, PROPERTY_DIMS
from teletron.utils import set_config

def unpack_tensors(packed_tensor, intervals, producer_tensors=None):
    features = [packed_tensor[intervals[i-1]:intervals[i]] for i in range(1, len(intervals))]
    if producer_tensors is not None:
        assert len(producer_tensors) == len(features)
    return features

class BaseBatchLoader(ABC):
    """
    """
    def __init__(self, data_iterator):
        self.data_iterator = data_iterator
        self.rank = mpu.get_tensor_context_parallel_rank()
        self.src_rank = mpu.get_tensor_context_parallel_src_rank()
        self.group = mpu.get_tensor_context_parallel_group()
        self.iteration = 0
        
        if self.rank == self.src_rank and self.data_iterator is None:
            print("Warning: data_iterator is None on the source rank.")

    def _get_debug_dump_path(self):
        base_dir = getattr(get_args(), "profile_path", None) or "."
        return os.path.join(base_dir, f"consumer/batch_debug_rank_{self.rank}.jsonl")

    def _broadcast_tensor(self, tensor):
        if tensor is not None:
            dist.broadcast(tensor.contiguous(), self.src_rank, group=self.group)

    def _broadcast_object(self, obj_list):
        dist.broadcast_object_list(obj_list, self.src_rank, group=self.group)

    @abstractmethod
    def _prepare_batch_on_rank_zero(self):
        pass

    def __iter__(self):
        return self


    def __next__(self):
        device = torch.cuda.current_device()
        from .utils import build_meta_tree, allocate_from_meta, broadcast_tensor_tree, recv_tensor_tree
        if self.rank == 0:
            # =========================
            # Rank 0: 准备 batch
            # =========================
            batch = self._prepare_batch_on_rank_zero()

            # dump_object_summary(
            #     batch,
            #     self._get_debug_dump_path(),
            #     meta={
            #         "rank": self.rank,
            #         "iteration": self.iteration,
            #         "stage": "consumer_batch_prepare",
            #     },
            # )

            if batch is None:
                self._broadcast_object([None])
                raise StopIteration

            # 1. 构造 meta tree
            meta_tree = build_meta_tree(batch)

            # 2. broadcast meta
            self._broadcast_object([meta_tree])

            # 3. 递归 broadcast Tensor
            broadcast_tensor_tree(batch, self._broadcast_tensor)

            # dump_object_summary(
            #     batch,
            #     self._get_debug_dump_path(),
            #     meta={
            #         "rank": self.rank,
            #         "iteration": self.iteration,
            #         "stage": "consumer_batch",
            #     },
            # )

            self.iteration += 1
            return batch

        else:
            # =========================
            # Non-rank0: 接收 batch
            # =========================
            meta_list = [None]
            self._broadcast_object(meta_list)
            meta_tree = meta_list[0]

            if meta_tree is None:
                raise StopIteration

            # 1. 根据 meta 构造空 batch
            batch = allocate_from_meta(meta_tree, device)

            # 2. 递归接收 Tensor
            recv_tensor_tree(batch, self._broadcast_tensor)

            # dump_object_summary(
            #     batch,
            #     self._get_debug_dump_path(),
            #     meta={
            #         "rank": self.rank,
            #         "iteration": self.iteration,
            #         "stage": "consumer_batch",
            #     },
            # )

            self.iteration += 1
            return batch

    # def __next__(self):
    #     if self.rank == 0:
    #         batch = self._prepare_batch_on_rank_zero()
    #         dump_object_summary(
    #             batch,
    #             self._get_debug_dump_path(),
    #             meta={
    #                 "rank": self.rank,
    #                 "iteration": self.iteration,
    #                 "stage": "consumer_batch_prepare",
    #             },
    #         )
    #         if batch is None: 
    #             self._broadcast_object([None])
    #             raise StopIteration

    #         meta_info = {}
    #         for key, value in batch.items():
    #             if isinstance(value, torch.Tensor):
    #                 meta_info[key] = {'shape': value.shape, 'dtype': value.dtype}
    #             elif isinstance(value, list):
    #                 meta_info[key] = {'shape': len(value), 'dtype': list}
    #             else:
    #                 raise TypeError(f"Unsupported type {type(value)} for broadcasting in batch.")
            
    #         self._broadcast_object([meta_info])

    #         for key, value in batch.items():
    #             if isinstance(value, torch.Tensor):
    #                 self._broadcast_tensor(value)
    #             elif isinstance(value, list):
    #                 self._broadcast_object(value)

    #         dump_object_summary(
    #             batch,
    #             self._get_debug_dump_path(),
    #             meta={
    #                 "rank": self.rank,
    #                 "iteration": self.iteration,
    #                 "stage": "consumer_batch",
    #             },
    #         )
    #         self.iteration += 1
    #         return batch
    #     else:
    #         meta_info_list = [None]
    #         self._broadcast_object(meta_info_list)
    #         meta_info = meta_info_list[0]

    #         if meta_info is None:
    #             raise StopIteration

    #         batch = {}
    #         for key, info in meta_info.items():
    #             dtype = info['dtype']
    #             shape = info['shape']
    #             if dtype is list:
    #                 batch[key] = [None] * shape
    #             else:
    #                 batch[key] = torch.empty(shape, dtype=dtype, device=torch.cuda.current_device())
            
    #         # 3. 接收广播的数据填充容器
    #         for key, value in batch.items():
    #             if isinstance(value, torch.Tensor):
    #                 self._broadcast_tensor(value)
    #             elif isinstance(value, list):
    #                 self._broadcast_object(value)
    #         dump_object_summary(
    #             batch,
    #             self._get_debug_dump_path(),
    #             meta={
    #                 "rank": self.rank,
    #                 "iteration": self.iteration,
    #                 "stage": "consumer_batch",
    #             },
    #         )
    #         self.iteration += 1
    #         return batch

def _unflatten_tensor_tree(paths, tensors):
    """
    paths: List[str], e.g. ["chosen/latents", "context"]
    tensors: List[Tensor], same order
    return: nested dict
    """
    root = {}
    for path, t in zip(paths, tensors):
        cur = root
        parts = path.split("/")
        for p in parts[:-1]:
            if p not in cur:
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = t
    return root



class VastDistBatchLoader(BaseBatchLoader):

    # def _prepare_batch_on_rank_zero(self):
    #     # if self.data_iterator is None:
    #     #     return None
        
    #     # 1. 从数据迭代器获取原始数据（如果需要的话）
    #     # data = next(self.data_iterator)
        
    #     # 2. 从 producer rank 接收 Tensors
    #     # breakpoint()
    #     comm_pair = get_comm_pair()
    #     args = get_args()

    #     meta_info = [None]
    #     dist.recv_object_list(meta_info, comm_pair.producer)
    #     meta_info = meta_info[0]


    #     batch = {}
    #     # unpack
    #     if args.distributed_vae:
    #         intervals = [0]
            
    #         for data_to_get in TeleaiEncoder.get_output_schema():
    #             data_size = 1
    #             for dim in meta_info[data_to_get]:
    #                 data_size *= dim 
    #             intervals.append(intervals[-1] + data_size)
            
    #         total_size = intervals[-1]
    #         recv_tensor = torch.empty((total_size), device=torch.cuda.current_device(), dtype=torch.bfloat16)
    #         dist.recv(recv_tensor, comm_pair.producer, tag=0)
    #         unpacked_data = unpack_tensors(recv_tensor, intervals, TeleaiEncoder.get_output_schema())

    #         for i, data_to_get in enumerate(TeleaiEncoder.get_output_schema()):
    #             tensor_shape = meta_info[data_to_get]
    #             reshaped_data = unpacked_data[i].view(*tensor_shape)
    #             batch[data_to_get] = reshaped_data
    #     else:
    #         # 如果 distributed_vae 为 False，需要定义相应的行为
    #         # 例如，返回空的或默认的 tensors
    #         raise NotImplementedError("distributed_vae=False case not implemented in this refactoring.")

    #     return batch

    def _prepare_batch_on_rank_zero(self):
        comm_pair = get_comm_pair()
        args = get_args()

        # ========== 1️⃣ recv meta ==========
        meta_info = [None]
        dist.recv_object_list(meta_info, comm_pair.producer)
        meta_info = meta_info[0]

        paths = meta_info["paths"]
        shapes = meta_info["shapes"]

        # ========== 2️⃣ 计算 flat intervals ==========
        intervals = [0]
        for p in paths:
            size = 1
            for d in shapes[p]:
                size *= d
            intervals.append(intervals[-1] + size)

        total_size = intervals[-1]

        # ========== 3️⃣ recv packed tensor ==========
        recv_tensor = torch.empty(
            (total_size,),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,   # 和 producer 保持一致
        )
        dist.recv(recv_tensor, comm_pair.producer, tag=0)

        # ========== 4️⃣ unpack flat tensors ==========
        flat_tensors = unpack_tensors(recv_tensor, intervals)

        # reshape
        flat_named = []
        for p, t in zip(paths, flat_tensors):
            flat_named.append(t.view(*shapes[p]))

        # ========== 5️⃣ unflatten to nested batch ==========
        batch = _unflatten_tensor_tree(paths, flat_named)

        return batch

class WanDistBatchLoader(BaseBatchLoader):

    def _prepare_batch_on_rank_zero(self):
        if self.data_iterator is None:
            return None
        
        # 1. 从数据迭代器获取原始数据（如果需要的话）
        # data = next(self.data_iterator)
        
        # 2. 从 producer rank 接收 Tensors
        comm_pair = get_comm_pair()
        args = get_args()
        info_size  = sum([PROPERTY_DIMS[data_to_get] for data_to_get in WanVideoEncoder.get_output_schema()])
        tensors_info = torch.ones((info_size), device=torch.cuda.current_device(), dtype=torch.int32)
        req = dist.irecv(tensors_info, comm_pair.producer)
        req.wait()

        batch = {}
        # unpack
        if args.distributed_vae:
            start_dim = 0
            intervals = [0]
            
            for data_to_get in WanVideoEncoder.get_output_schema():
                dims = PROPERTY_DIMS[data_to_get]
                data_size = 1
                for dim in tensors_info[start_dim:start_dim + dims].tolist():
                    data_size *= dim 
                start_dim += dims
                intervals.append(intervals[-1] + data_size)
            
            total_size = intervals[-1]
            recv_tensor = torch.empty((total_size), device=torch.cuda.current_device(), dtype=torch.bfloat16)
            req = dist.irecv(recv_tensor, comm_pair.producer, tag=0)
            req.wait()
            
            unpacked_data = unpack_tensors(recv_tensor, intervals, WanVideoEncoder.get_output_schema())
            start_dim = 0
            for i, data_to_get in enumerate(WanVideoEncoder.get_output_schema()):
                dims = PROPERTY_DIMS[data_to_get]
                tensor_shape = tensors_info[start_dim:start_dim + dims].tolist()
                reshaped_data = unpacked_data[i].view(*tensor_shape)
                batch[data_to_get] = reshaped_data
                start_dim += dims
        else:
            # 如果 distributed_vae 为 False，需要定义相应的行为
            # 例如，返回空的或默认的 tensors
            raise NotImplementedError("distributed_vae=False case not implemented in this refactoring.")

        
        return batch

class CausalWanOriginalBatchLoader(BaseBatchLoader):
    def _prepare_batch_on_rank_zero(self):
        if self.data_iterator is None:
            return None
        
        try:
            data = next(self.data_iterator)
        except StopIteration:
            raise NotImplementedError("CausalWanModel")
            return None # 返回 None 以向基类发出迭代结束的信号

        batch = {
            'latents': data["latents"].cuda(non_blocking=True),
            'prompt_emb': data["prompt_emb"]['context'].cuda(non_blocking=True),
            # 'image_emb': data["image_emb"], 
        }
        return batch

class CausalWanBatchLoader(BaseBatchLoader):
    def _prepare_batch_on_rank_zero(self):
        if self.data_iterator is None:
            return None
        
        try:
            data = next(self.data_iterator)
        except StopIteration:
            raise NotImplementedError("CausalWanModel")
            return None # 返回 None 以向基类发出迭代结束的信号

        batch = {
            'latents': data["latents"].cuda(non_blocking=True),
            'prompt_emb': data["prompt_emb"].cuda(non_blocking=True),
            'unprompt_emb': data["unprompt_emb"].cuda(non_blocking=True),
        }
        return batch

class CausalDistBatchLoader(BaseBatchLoader):

    def _prepare_batch_on_rank_zero(self):

        comm_pair = get_comm_pair()
        args = get_args()

        meta_info_list = [None]
        dist.recv_object_list(meta_info_list, comm_pair.producer)
        meta_info = meta_info_list[0]

        batch = {}
        
        if not args.distributed_vae:
            raise NotImplementedError("CausalDistBatchLoader requires distributed_vae=True.")

        intervals = [0]

        data_keys_to_receive = TeleaiEncoder.get_output_schema()

        for key in data_keys_to_receive:
            shape = meta_info[key]
            data_size = 1
            for dim in shape:
                data_size *= dim
            intervals.append(intervals[-1] + data_size)

        total_size = intervals[-1]

        recv_tensor = torch.empty((total_size), device=torch.cuda.current_device(), dtype=torch.bfloat16)
        dist.recv(recv_tensor, comm_pair.producer, tag=0)

        unpacked_data = unpack_tensors(recv_tensor, intervals, data_keys_to_receive)

        for i, key in enumerate(data_keys_to_receive):
            tensor_shape = meta_info[key]
            reshaped_data = unpacked_data[i].view(*tensor_shape)
            batch[key] = reshaped_data

        return batch

def create_batch_loader(args, data_iterator):
    model_name_lower = set_config().model_config.dit.type.lower()
    is_distributed_vae = args.distributed_vae

    if 'teleai' in model_name_lower:
        if is_distributed_vae:
            print("Info: Creating VastDistBatchLoader.")
            return VastDistBatchLoader(data_iterator)
        else:
            raise NotImplementedError("A non-distributed VAE loader for VastModel is not implemented.")
    elif 'wan' in model_name_lower:
        if is_distributed_vae:
            print("Info: Creating VastDistBatchLoader.")
            return WanDistBatchLoader(data_iterator)
        else:
            raise NotImplementedError("A non-distributed VAE loader for VastModel is not implemented.")        
    elif 'causal' in model_name_lower:
        if is_distributed_vae:
            print("Info: Creating CausalWanBatchLoader.")
            return CausalDistBatchLoader(data_iterator)
        else:
            print("Info: Creating CausalWanOriginalBatchLoader.")
            return CausalWanOriginalBatchLoader(data_iterator)
    else:
        raise ValueError(f"Unknown model name '{args.model_name}' for batch loader creation.")
