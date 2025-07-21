import os
import torch
import torch.distributed as dist
import collections
import time
import copy
from typing import Callable, Any, Dict, List, Tuple
import json
from teletron.train.checkpoint.utils import (
    # read_metadata,
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
        if isinstance(cp.consumer, int):
            merge_dict[key].append([cp.consumer])
        else: 
            merge_dict[key].append(cp.consumer)
    
    merged_list = {idx: None for idx in range(len(merge_dict))}

    idx=0
    for key, consumers_list in merge_dict.items():
        flat_consumers = []
        for sublist in consumers_list:
            if isinstance(sublist, list):
                flat_consumers.extend(sublist)
            else:
                flat_consumers.append(sublist)
        new_cp = CommPair(
            producer=key[0],
            consumer=flat_consumers,
            dp_rank=key[1],
            dp_size=key[2]
        )
        merged_list[idx] = new_cp
        idx+=1
    return merged_list

def save_producer_checkpoint(args, rank, iteration, consumed_train_samples, consumed_valid_samples):
    """保存producer的状态到文件。"""
    # checkpoint_path = os.path.join(args.save, "producer_checkpoints", f"rank_{rank}_checkpoint.json") if args.save is not None else None
    checkpoint_path = get_checkpoint_name(args.save, iteration, return_base_dir=True) if args.save is not None else None
    
    if checkpoint_path is None:
        return
    checkpoint_file = os.path.join(checkpoint_path, "producer_checkpoints", f"rank_{rank}_checkpoint.json")
    ensure_directory_exists(checkpoint_file)

    state = {
        'iteration': iteration,
        'consumed_train_samples': consumed_train_samples,
        'consumed_valid_samples': consumed_valid_samples,
    }

    with open(checkpoint_file, 'w') as f:
        json.dump(state, f, indent=4)
    print(f"Producer Rank {rank}: 已在第 {iteration} 步保存检查点于 {checkpoint_file}", flush=True)

def read_metadata(tracker_filename):
    # Read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration = 0
    release = False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
    return iteration, release

def load_producer_checkpoint(args, rank):
    """从文件加载producer的状态。"""
    # print(f"in load !!!"*5, flush=True)
    if not args.load:
        return None
    tracker_filename = get_checkpoint_tracker_filename(args.load)
    print(f"before read_metadata: {tracker_filename}", flush=True)
    iteration, release = read_metadata(tracker_filename)
    if release is True:
        return None
    # print(f"before checkpoint_name"*10, flush=True)
    checkpoint_path = get_checkpoint_name(args.load, iteration, return_base_dir=True)
    checkpoint_name = os.path.join(checkpoint_path, "producer_checkpoints", f"rank_{rank}_checkpoint.json")
    print(f"checkpoint_name: {checkpoint_name}"*10, flush=True)
    if checkpoint_name and os.path.exists(checkpoint_name):
        with open(checkpoint_name, 'r') as f:
            state = json.load(f)
        print(f"Producer Rank {rank}: 成功从 {checkpoint_name} 加载检查点。", flush=True)
        return state
    return None

class DistDataProducer:
    def __init__(
        self,
        rank:int,
        encoder_name: str,
        device,
        build_train_valid_test_data_iterators: Callable,
        train_ds: Any = None,
        valid_ds: Any = None,
    ):
        args = get_args()
        self.args = args
        self.do_valid = args.eval_iters > 0
        self.rank = rank
        self.device = device
        self.build_data_iterators_fn = build_train_valid_test_data_iterators
        self.train_ds_preloaded = train_ds
        self.valid_ds_preloaded = valid_ds
        
        # 1. 设置编码器
        self.encoder = get_encoder(name=encoder_name, device=self.device)
        self.encoder.setup()

        # 2. 初始化通信状态
        self.comm_pairs = get_comm_pair()
        self.merged_comm_pairs = merge_commpairs(self.comm_pairs)
        self._initialize_consumer_state()
        
        # 3. 合并通信对并创建数据迭代器
        
        self._create_data_iterators()
        
        # 4. 初始化数据队列和发送请求跟踪器
        self._initialize_queues()

        # 5. 设置性能分析器 (如果启用)
        self._setup_profiler()

    
    def _initialize_consumer_state(self):
        args = get_args()
        # loaded_state = load_producer_checkpoint(args, self.rank)
        # if loaded_state:
        #     # 如果成功加载，直接使用文件中的状态
        #     args.iteration = loaded_state['iteration']
        #     args.consumed_train_samples = loaded_state['consumed_train_samples']
        #     args.consumed_valid_samples = loaded_state['consumed_valid_samples']
        #     print(f"Producer Rank {self.rank}: 从检查点恢复状态。Iteration: {args.iteration}", flush=True)


        #     consumers_data = torch.zeros((len(self.comm_pairs), 3), dtype=torch.int64, device=self.device)
        #     req_queue = [dist.irecv(tensor=consumers_data[i], src=cp.consumer) for i, cp in enumerate(self.comm_pairs)]
        #     for req in req_queue:
        #         req.wait()
        #     print(f"Producer Rank {self.rank}: 已与 Consumer 同步，将使用自己加载的状态。", flush=True)
        # else:
        #     # 如果没有检查点文件，则执行原来的逻辑，从 consumer 获取初始状态
        #     print(f"Producer Rank {self.rank}: 未找到检查点，从 Consumer 获取初始状态。", flush=True)
        consumers_data = torch.zeros((len(self.comm_pairs), 3), dtype=torch.int64, device=self.device)
        req_queue = [dist.irecv(tensor=consumers_data[i], src=cp.consumer) for i, cp in enumerate(self.comm_pairs)]
        for req in req_queue:
            req.wait()
        # 假设所有consumer的初始状态是一致的
        args.iteration = consumers_data[0][0].item()
        args.consumed_train_samples = consumers_data[0][1].item() // args.distributed_vae_world_size
        args.consumed_valid_samples = consumers_data[0][2].item() // args.distributed_vae_world_size
        print(f"iteration: {args.iteration}", flush=True)
        print(f"consumed_train_samples: {args.consumed_train_samples}", flush=True)
        print(f"consumed_valid_samples: {args.consumed_valid_samples}", flush=True)
    
    def  _create_data_iterators(self):
        # args = get_args()
        
        self.train_iterators = {}
        self.valid_iterators = {}
        self.same_data_group = {}
        
        train_ds_current = self.train_ds_preloaded
        valid_ds_current = self.valid_ds_preloaded
        
        for idx, mcp in self.merged_comm_pairs.items():

            dp_rank = idx if self.args.temp_accelerate else mcp.dp_rank
            dp_size = len(self.merged_comm_pairs) if self.args.temp_accelerate else mcp.dp_size

            train_iter, valid_iter, _, train_ds_current, valid_ds_current = self.build_data_iterators_fn(
                is_tp_first=True,
                dp_rank=dp_rank,
                dp_size=dp_size,
                train_ds_prev=train_ds_current,
                valid_ds_prev=valid_ds_current,
                return_ds=True
            )
            self.train_iterators[idx] = train_iter
            self.valid_iterators[idx] = valid_iter

            # 映射第一个消费者到所有需要相同数据的消费者列表
            first_consumer = mcp.consumer[0]
            self.same_data_group[first_consumer] = mcp.consumer
            if not self.do_valid:
                self.valid_iterators = None
        
        self.train_ds_preloaded = valid_ds_current
        self.valid_ds_preloaded = valid_ds_current
    
    def _initialize_queues(self):
        all_consumer_ranks = [cp.consumer for cp in self.comm_pairs]
        
        self.train_data_queues = {rank: collections.deque() for rank in all_consumer_ranks}
        self.train_size_queues = {rank: collections.deque() for rank in all_consumer_ranks}
        self.train_sended_count = {rank: 0 for rank in all_consumer_ranks}
        self.train_received_count = {rank: 0 for rank in all_consumer_ranks}
        
        self.train_size_reqs: List[Tuple[dist.Work, int, torch.Tensor]] = []
        self.train_data_reqs: List[Tuple[dist.Work, int, torch.Tensor]] = []
        
        if self.do_valid:
            self.valid_data_queues = {rank: collections.deque() for rank in all_consumer_ranks}
            self.valid_size_queues = {rank: collections.deque() for rank in all_consumer_ranks}
            self.valid_sended_count = {rank: 0 for rank in all_consumer_ranks}
            self.valid_received_count = {rank: 0 for rank in all_consumer_ranks}
            
            self.valid_size_reqs: List[Tuple[dist.Work, int, torch.Tensor]] = []
            self.valid_data_reqs: List[Tuple[dist.Work, int, torch.Tensor]] = []

    def _setup_profiler(self):
        """如果配置中启用，则设置PyTorch Profiler。"""
        self.profiler = None
        if self.args.producer_profile:
            prof_save_path = os.path.join(self.args.profile_path, f"producer/rank_{dist.get_rank()}.json")
            ensure_directory_exists(prof_save_path)
            
            def trace_handler(p):
                p.export_chrome_trace(prof_save_path)

            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                with_stack=True,
                on_trace_ready=trace_handler,
                record_shapes=True
            )
    
    def _cleanup_completed_sends(self):
        """清理已完成的异步发送请求。"""
        new_train_size_reqs = []
        new_train_data_reqs = []
        for r1,r2 in zip(self.train_size_reqs, self.train_data_reqs):
            if r1[0].is_completed()and r2[0].is_completed():
                self.train_received_count[r1[1]] += 1
                # del r1[2]
                # del r2[2]
            else:
                new_train_size_reqs.append(r1)
                new_train_data_reqs.append(r2)
        
        self.train_size_reqs = new_train_size_reqs
        self.train_data_reqs = new_train_data_reqs
            
        if self.do_valid:
            new_valid_size_reqs = []
            new_valid_data_reqs = []
            for r1,r2 in zip(self.valid_size_reqs, self.valid_data_reqs):
                if r1[0].is_completed() and r2[0].is_completed():
                    self.valid_received_count[r1[1]] += 1
                    # del r1[2]
                    # del r2[2]
                else:
                    new_valid_size_reqs.append(r1)
                    new_valid_data_reqs.append(r2)
            
            self.valid_size_reqs = new_valid_size_reqs
            self.valid_data_reqs = new_valid_data_reqs
            
    def _produce_and_enqueue_data(self, idx, mcp,mission:str = 'train'):
        """从数据迭代器生成新数据，编码后放入队列。"""
        # for idx, mcp in self.merged_comm_pairs.items():
        first_consumer = mcp.consumer[0]
        
        # 如果该组消费者的队列未满，则生成新数据
        if mission == 'train':
            if len(self.train_data_queues[first_consumer]) < MAX_QUEUE_PER_CONSUMER_ON_PRODUCER:
                try:
                    raw_batch = next(self.train_iterators[idx])
                except StopIteration:
                    print(f"警告: 数据迭代器 {idx} 已耗尽。")
                    return
                
                # 编码数据
                tensors_to_send, size_info_tensor = self.encoder.encode(raw_batch)
                packed_tensor = self.encoder._pack_tensors(tensors_to_send)

                # 将数据分发给所有需要它的消费者队列
                for consumer_rank in self.same_data_group[first_consumer]:
                    self.train_size_queues[consumer_rank].append(size_info_tensor)
                    self.train_data_queues[consumer_rank].append(packed_tensor)
                    
        elif mission == 'valid':
            if len(self.valid_data_queues[first_consumer]) < MAX_QUEUE_PER_CONSUMER_ON_PRODUCER:
                try:
                    raw_batch = next(self.valid_iterators[idx])
                except StopIteration:
                    print(f"信息: 数据迭代器 {idx} 已耗尽")
                    # print(f"信息: 数据迭代器 {idx} 已耗尽，正在重新创建...")
                    # dp_rank = idx if self.args.temp_accelerate else mcp.dp_rank
                    # dp_size = len(self.merged_comm_pairs) if self.args.temp_accelerate else mcp.dp_size

                    # _, valid_iter, _, train_ds_current, valid_ds_current = self.build_data_iterators_fn(
                    #     is_tp_first=True,
                    #     dp_rank=dp_rank,
                    #     dp_size=dp_size,
                    #     train_ds_prev=self.train_ds_preloaded,
                    #     valid_ds_prev=self.valid_ds_preloaded,
                    #     return_ds=True
                    # )
                    # self.valid_iterators[idx] = valid_iter
                    # raw_batch = next(self.valid_iterators[idx])
                    return
                
                # 编码数据
                tensors_to_send, size_info_tensor = self.encoder.encode(raw_batch)
                packed_tensor = self.encoder._pack_tensors(tensors_to_send)

                # 将数据分发给所有需要它的消费者队列
                for consumer_rank in self.same_data_group[first_consumer]:
                    self.valid_size_queues[consumer_rank].append(size_info_tensor)
                    self.valid_data_queues[consumer_rank].append(packed_tensor)
                    
        else:
            raise ValueError(f"produce mission error")
    
    def _initiate_new_sends(self,cp,mission:str='train'):
        """从队列中取出数据，并启动新的异步发送操作。"""
        # for cp in self.comm_pairs:
        cr = cp.consumer
        
        # 检查是否有待发送数据，并且未完成的发送请求数未达上限
        if mission == 'train':
            outstanding_sends = sum(1 for _, c, _ in self.train_size_reqs if c == cr)
            if self.train_size_queues[cr] and outstanding_sends < MAX_OUTSTANDING_SENDS_PER_CONSUMER:
                
                # 1. 发送尺寸信息
                size_to_send = self.train_size_queues[cr].popleft()
                req_size = dist.isend(tensor=size_to_send, dst=cr)
                self.train_size_reqs.append((req_size, cr, size_to_send))

                # 2. 发送数据本身
                tensor_to_send = self.train_data_queues[cr].popleft()
                req_data = dist.isend(tensor=tensor_to_send, dst=cr)
                self.train_sended_count[cr] += 1
                self.train_data_reqs.append((req_data, cr, tensor_to_send))
        elif mission == 'valid':
            outstanding_sends = sum(1 for _, c, _ in self.valid_size_reqs if c == cr)
            if self.valid_size_queues[cr] and outstanding_sends < MAX_OUTSTANDING_SENDS_PER_CONSUMER:
                
                # 1. 发送尺寸信息
                size_to_send = self.valid_size_queues[cr].popleft()
                req_size = dist.isend(tensor=size_to_send, dst=cr)
                self.valid_size_reqs.append((req_size, cr, size_to_send))

                # 2. 发送数据本身
                tensor_to_send = self.valid_data_queues[cr].popleft()
                req_data = dist.isend(tensor=tensor_to_send, dst=cr)
                self.valid_sended_count[cr] += 1
                self.valid_data_reqs.append((req_data, cr, tensor_to_send))
        else:
            raise ValueError(f"data send mission error")
        
    def _wait_all_reqs_end(self):
        print(f"Rank {dist.get_rank()}: 所有数据项已启动发送，等待最终完成...")
        for req1, req2 in zip(self.train_size_reqs,self.train_data_reqs):
            if not req1[0].is_completed():
                req1[0].wait()
            if not req2[0].is_completed():
                req2[0].wait()
        
        if self.do_valid:
            for req1, req2 in zip(self.valid_size_reqs,self.valid_data_reqs):
                if not req1[0].is_completed():
                    req1[0].wait()
                if not req2[0].is_completed():
                    req2[0].wait()
    
    def _produce_and_send_with_valid(self):
        args = get_args()
        train_data_count = args.eval_interval
        valid_data_count = args.eval_iters
        
        old_train_sended_count = copy.deepcopy(self.train_sended_count)
        old_valid_sended_count = copy.deepcopy(self.valid_sended_count)
        
        # train data
        while any(self.train_sended_count[cp.consumer] - old_train_sended_count[cp.consumer] < train_data_count for cp in self.comm_pairs):
            # clean data
            self._cleanup_completed_sends()
            # produce data
            for idx, mcp in self.merged_comm_pairs.items():
                self._produce_and_enqueue_data(idx, mcp,'train')
            # send data
            for cp in self.comm_pairs:
                if self.train_sended_count[cp.consumer] - old_train_sended_count[cp.consumer] < train_data_count:
                    self._initiate_new_sends(cp, 'train')
                    
        # valid data
        while any(self.valid_sended_count[cp.consumer] - old_valid_sended_count[cp.consumer] < valid_data_count for cp in self.comm_pairs):
            # clean data
            self._cleanup_completed_sends()
            # produce data
            for idx, mcp in self.merged_comm_pairs.items():
                self._produce_and_enqueue_data(idx, mcp,'valid')
            # send data
            for cp in self.comm_pairs:
                if self.valid_sended_count[cp.consumer] - old_valid_sended_count[cp.consumer] < valid_data_count:
                    self._initiate_new_sends(cp, 'valid')
                
    def _produce_and_send_without_valid(self):
        for idx, mcp in self.merged_comm_pairs.items():
            self._produce_and_enqueue_data(idx, mcp)
        for cp in self.comm_pairs:
            self._initiate_new_sends(cp)
            
    def run(self):
        step=0
        # args = get_args()
        try:
            while any(self.train_sended_count[cp.consumer] < NUM_ITEMS_PER_CONSUMER for cp in self.comm_pairs):
                # 启动性能分析器
                if self.profiler and step == self.args.profile_step_start:
                    self.profiler.start()

                # 阶段 A: 清理已完成的发送
                self._cleanup_completed_sends()

                if self.do_valid:
                    self._produce_and_send_with_valid()
                else:
                    self._produce_and_send_without_valid()
                
                # current_train_step_approx = self.train_sended_count[self.comm_pairs[0].consumer]
                # current_valid_step_approx = self.valid_sended_count[self.comm_pairs[0].consumer]
                
                # if args.save and args.save_interval and current_train_step_approx % args.save_interval == 0 and current_train_step_approx > 0:
                #     save_producer_checkpoint(args, self.rank, args.iteration + current_train_step_approx, 
                #                              args.consumed_train_samples + current_train_step_approx * len(self.merged_comm_pairs),
                #                              args.consumed_valid_samples + current_valid_step_approx * len(self.merged_comm_pairs))
                # time.sleep(0.01)
                
                # 停止性能分析器
                if self.profiler and step == self.args.profile_step_end:
                    self.profiler.stop()
                    print(f"Rank {dist.get_rank()}: Profiler data saved.")
                
                
                step += 1
                time.sleep(0.01) # 短暂休眠以避免CPU空转

            # 等待所有挂起的通信完成
            self._wait_all_reqs_end()
            
            # 全局屏障，确保所有进程都完成了它们的工作
            dist.barrier(group=get_world_group())

        except Exception as e:
            import traceback
            print(f"Rank {dist.get_rank()} 发生异常:")
            traceback.print_exc()
            # 强制中止，避免部分进程挂起
            dist.abort(group=get_world_group())
        finally:
            cleanup_dist()
            
        