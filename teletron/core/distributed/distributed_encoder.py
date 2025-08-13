import os
import torch
import torch.distributed as dist
import collections
import time
import copy
from typing import Callable, Any, Dict, List, Tuple,Literal
import json
from teletron.train.checkpoint.utils import (
    # read_metadata, # 移除未使用的导入
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
)
from teletron.core.parallel_state import get_comm_pair, get_world_group, CommPair
from teletron.utils import get_args
from teletron.train.checkpoint import ensure_directory_exists
from teletron.models.encoder_registry import get_encoder


NUM_ITEMS_PER_CONSUMER = 100000
MAX_QUEUE_PER_CONSUMER_ON_PRODUCER = 2
MAX_OUTSTANDING_SENDS_PER_CONSUMER = 1


TRAIN_MODE = 'train'
VALID_MODE = 'valid'

def cleanup_dist():
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: 销毁进程组。")
        dist.destroy_process_group()

def merge_commpairs(commpairs: list) -> Dict[int, CommPair]:
    merge_dict = {}
    for cp in commpairs:
        key = (cp.producer, cp.dp_rank, cp.dp_size)
        if key not in merge_dict:
            merge_dict[key] = []
        
        # 统一处理单个或多个 consumer 的情况
        consumers = cp.consumer if isinstance(cp.consumer, list) else [cp.consumer]
        merge_dict[key].extend(consumers)
    
    merged_list = {}
    for idx, (key, consumers_list) in enumerate(merge_dict.items()):
        new_cp = CommPair(
            producer=key[0],
            consumer=consumers_list,
            dp_rank=key[1],
            dp_size=key[2]
        )
        merged_list[idx] = new_cp
    return merged_list


class DistDataProducer:
    def __init__(
        self,
        rank: int,
        encoder_name: str,
        device,
        build_train_valid_test_data_iterators: Callable,
        train_ds: Any = None,
        valid_ds: Any = None,
    ):
        self.args = get_args()
        self.rank = rank
        self.device = device
        self.encoder = get_encoder(name=encoder_name, device=self.device)
        self.build_data_iterators_fn = build_train_valid_test_data_iterators
        self.train_ds_preloaded = train_ds
        self.valid_ds_preloaded = valid_ds
        self.step = 0
        self.modes = [TRAIN_MODE]
        if self.args.eval_iters > 0:
            self.modes.append(VALID_MODE)
        if self.args.producer_profile:
            self.batch_size = 1
        else:
            self.batch_size = self.args.producer_batch_size
        
        self.encoder.setup()
        self.comm_pairs = get_comm_pair()
        self.merged_comm_pairs = merge_commpairs(self.comm_pairs)

        # 2. 初始化 Consumer 状态
        self._initialize_consumer_state()
        
        # 3. 创建数据迭代器
        self._create_data_iterators()
        
        # 4. 初始化队列和请求跟踪器
        self._initialize_queues_and_trackers()

        # 5. 设置性能分析器 (Profiler)
        self._setup_profiler()

    def _initialize_consumer_state(self):
        """
        与 Consumers 同步初始状态。
        """
        print(f"Producer Rank {self.rank}: 从 Consumer 获取初始状态。")
        consumers_data = torch.zeros((len(self.comm_pairs), 3), dtype=torch.int64, device=self.device)
        req_queue = [dist.irecv(tensor=consumers_data[i], src=cp.consumer) for i, cp in enumerate(self.comm_pairs)]
        for req in req_queue:
            req.wait()
        
        # 假设所有 consumer 的初始状态是一致的
        self.args.iteration = consumers_data[0][0].item()
        self.args.consumed_train_samples = consumers_data[0][1].item() // self.args.distributed_vae_world_size
        self.args.consumed_valid_samples = consumers_data[0][2].item() // self.args.distributed_vae_world_size
        print(f"Producer Rank {self.rank}: 同步完成。Iteration: {self.args.iteration}, Consumed Train: {self.args.consumed_train_samples}")

    def _create_data_iterators(self):
        self.data_iterators = {mode: {} for mode in self.modes}
        self.same_data_group = {}
        
        train_ds_current = self.train_ds_preloaded
        valid_ds_current = self.valid_ds_preloaded
        
        for idx, mcp in self.merged_comm_pairs.items():
            dp_rank = idx
            dp_size = len(self.merged_comm_pairs)

            train_iter, valid_iter, _, train_ds_current, valid_ds_current = self.build_data_iterators_fn(
                is_tp_first=True, dp_rank=dp_rank, dp_size=dp_size,
                train_ds_prev=train_ds_current, valid_ds_prev=valid_ds_current, return_ds=True
            )
            self.data_iterators[TRAIN_MODE][idx] = train_iter
            if VALID_MODE in self.modes:
                self.data_iterators[VALID_MODE][idx] = valid_iter

            first_consumer = mcp.consumer[0]
            self.same_data_group[first_consumer] = mcp.consumer
        

    def _initialize_queues_and_trackers(self):
        all_consumer_ranks = [cp.consumer for cp in self.comm_pairs]
        self.data_queues = {}
        self.size_queues = {}
        self.sended_count = {}
        self.received_count = {}
        self.size_reqs = {}
        self.data_reqs = {}

        for mode in self.modes:
            self.data_queues[mode] = {rank: collections.deque() for rank in all_consumer_ranks}
            self.size_queues[mode] = {rank: collections.deque() for rank in all_consumer_ranks}
            self.sended_count[mode] = {rank: 0 for rank in all_consumer_ranks}
            self.received_count[mode] = {rank: 0 for rank in all_consumer_ranks}
            self.size_reqs[mode] = []
            self.data_reqs[mode] = []

    def _setup_profiler(self):
        """如果配置中启用，则设置PyTorch Profiler。"""
        self.profiler = None
        if self.args.producer_profile:
            prof_save_path = os.path.join(self.args.profile_path, f"producer/rank_{dist.get_rank()}.json")
            ensure_directory_exists(prof_save_path)
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                with_stack=True,
                on_trace_ready=lambda p: p.export_chrome_trace(prof_save_path),
                record_shapes=True
            )
    
    def _cleanup_completed_sends(self, mode: str):
        new_size_reqs, new_data_reqs = [], []
        for r_size, r_data in zip(self.size_reqs[mode], self.data_reqs[mode]):
            if r_size[0].is_completed() and r_data[0].is_completed():
                consumer_rank = r_size[1]
                self.received_count[mode][consumer_rank] += 1
            else:
                new_size_reqs.append(r_size)
                new_data_reqs.append(r_data)
        
        self.size_reqs[mode] = new_size_reqs
        self.data_reqs[mode] = new_data_reqs
    
    def _produce_and_enqueue_data(self, idx: int, mcp: CommPair, mode: str):
        first_consumer = mcp.consumer[0]
        
        if len(self.data_queues[mode][first_consumer]) < MAX_QUEUE_PER_CONSUMER_ON_PRODUCER:
            try:
                raw_batch = [next(self.data_iterators[mode][idx]) for i in range(self.batch_size)]
            except StopIteration:
                # 统一处理迭代器耗尽的情况
                print(f"信息: {mode} 模式的数据迭代器 {idx} 已耗尽。")
                return

            tensors_to_send = self.encoder.encode(raw_batch)

            for item in tensors_to_send:
                size_info_tensor = self.encoder._get_tensors_size(item, device=self.device)
                packed_tensor = self.encoder._pack_tensors(item)
                
                for consumer_rank in self.same_data_group[first_consumer]:
                    self.size_queues[mode][consumer_rank].append(size_info_tensor)
                    self.data_queues[mode][consumer_rank].append(packed_tensor)

            if mode == TRAIN_MODE:
                self.step += self.batch_size
    
    def _initiate_new_sends(self, cp: CommPair, mode: str):
        consumer_rank = cp.consumer
        
        outstanding_sends = sum(1 for _, c, _ in self.size_reqs[mode] if c == consumer_rank)
        if self.size_queues[mode][consumer_rank] and outstanding_sends < MAX_OUTSTANDING_SENDS_PER_CONSUMER:
            size_to_send = self.size_queues[mode][consumer_rank].popleft()
            req_size = dist.isend(tensor=size_to_send, dst=consumer_rank)
            self.size_reqs[mode].append((req_size, consumer_rank, size_to_send))
            tensor_to_send = self.data_queues[mode][consumer_rank].popleft()
            req_data = dist.isend(tensor=tensor_to_send, dst=consumer_rank)
            self.data_reqs[mode].append((req_data, consumer_rank, tensor_to_send))
            
            self.sended_count[mode][consumer_rank] += 1
        
    def _wait_all_reqs_end(self):
        """等待所有模式下挂起的请求完成。"""
        print(f"Rank {dist.get_rank()}: 所有数据项已启动发送，等待最终完成...")
        for mode in self.modes:
            for r_size, r_data in zip(self.size_reqs[mode], self.data_reqs[mode]):
                r_size[0].wait()
                r_data[0].wait()
    
    def _main_produce_and_send(self):
        # 阶段 A: 清理所有模式下已完成的发送
        for mode in self.modes:
            self._cleanup_completed_sends(mode)
        
        # 阶段 B: 根据模式和需求生产和发送数据
        if VALID_MODE in self.modes:
            # 训练和验证交替进行
            train_data_count = self.args.eval_interval
            valid_data_count = self.args.eval_iters
            
            # 假设每个 consumer 的进度大致相同，以第一个为基准
            first_consumer = self.comm_pairs[0].consumer
            num_dispatched_in_cycle = self.sended_count[TRAIN_MODE][first_consumer] % train_data_count
            
            if num_dispatched_in_cycle < train_data_count:
                mode_to_process = TRAIN_MODE
                
            else:
                mode_to_process = VALID_MODE
            
            # 为当前模式生产和发送数据
            for idx, mcp in self.merged_comm_pairs.items():
                self._produce_and_enqueue_data(idx, mcp, mode_to_process)
            for cp in self.comm_pairs:
                self._initiate_new_sends(cp, mode_to_process)
        else:
            # 只处理训练数据
            for idx, mcp in self.merged_comm_pairs.items():
                self._produce_and_enqueue_data(idx, mcp, TRAIN_MODE)
            for cp in self.comm_pairs:
                self._initiate_new_sends(cp, TRAIN_MODE)

    def run(self):
        try:
            # 启动性能分析器
            if self.profiler and self.step == self.args.profile_step_start:
                self.profiler.start()

            # 主循环
            while any(self.sended_count[TRAIN_MODE][cp.consumer] < NUM_ITEMS_PER_CONSUMER for cp in self.comm_pairs):
                self._main_produce_and_send()
                
                # 检查是否停止 profiler
                if self.profiler and self.step == self.args.profile_step_end:
                    self.profiler.stop()
                    print(f"Rank {dist.get_rank()}: Profiler data saved.")
                
                
                time.sleep(0.01)  # 短暂休眠以避免CPU空转

            # 等待所有挂起的通信完成
            self._wait_all_reqs_end()
            
            # 全局屏障，确保所有进程都完成了它们的工作
            dist.barrier(group=get_world_group())

        except Exception as e:
            import traceback
            print(f"Rank {dist.get_rank()} 发生异常:")
            traceback.print_exc()
            dist.abort(group=get_world_group())
        finally:
            cleanup_dist()