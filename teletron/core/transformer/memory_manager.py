import torch
from collections import defaultdict, deque
from teletron.utils import get_args

class MemoryManager:
    """
    管理锁页内存缓冲区和专用的CUDA数据传输流。
    这对于实现高效的异步数据传输至关重要。
    """
    def __init__(self):
        self.device = torch.cuda.current_device
        # self.data_stream = torch.cuda.Stream(device=self.device)
        # 预先分配一个锁页内存池，以减少运行时开销
        self.memory_pool = defaultdict(deque)
        # self.event_pool = deque()
        args = get_args()
        self.num_layers = args.num_layers
        self._warmup()

    def get_event(self):

        return torch.cuda.Event()

    def return_event(self, event):
        """将用完的 Event 返回池中"""
        pass
        

    def _warmup(self):
        # print("Warming up PinnedMemoryManager...")
        # for _ in range(self.pool_size):
        #     # 创建一个小的占位符张量并固定它
        #     try:
        #         placeholder = torch.empty(1, dtype=torch.float32).pin_memory()
        #         self.pinned_tensors_pool.append(placeholder)
        #     except RuntimeError as e:
        #         print(f"Warning: Could not pin memory. Async offload might be slow. Error: {e}")
        #         # 如果pin_memory失败（例如在不支持的环境），则退回到非固定内存
        #         self.pinned_tensors_pool.append(torch.empty(1, dtype=torch.float32))
        pass

    def get_buffer(self,  shape, dtype):
        """从池中获取一个足够大的内存。"""
        key = (shape, dtype)
        if self.memory_pool[key]:
            # 从池中弹出一个可用的缓冲区
            return self.memory_pool[key].popleft()
        else:
            return torch.empty(shape, dtype=dtype)
        
    def return_buffer(self, buffer):
        """
        将使用完毕的锁页内存缓冲区归还到池中。
        """
        key = (buffer.shape, buffer.dtype)
        self.memory_pool[key].append(buffer)
    # def get_data_stream(self):
    #     """返回通信流。"""
    #     return self.data_stream
    # def get_prefetch_events_queue(self):
    #     """返回预取事件队列。"""
    #     return self.prefetch_events
    # def __str__(self):
    #     pool_stats = {str(k): len(v) for k, v in self.pinned_memory_pool.items()}
    #     return f"<Manager: CommStream={self.comm_stream.cuda_stream}, PoolStats={pool_stats}>"


# 全局管理器实例
_memory_manager = None

def get_memory_manager():
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager