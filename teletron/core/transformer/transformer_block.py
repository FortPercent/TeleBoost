import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import List, Union
from teletron.core.transformer.memory_manager import get_memory_manager
from torch.autograd.graph import saved_tensors_hooks
from typing import Tuple
from functools import partial


class save_on_cpu(saved_tensors_hooks):

    def __init__(self, pin_memory: bool = False, device_type: str = "cuda") -> None:
        device_module = getattr(torch, device_type, torch.cuda)

        def pack_to_cpu(tensor: torch.Tensor) -> Tuple[torch.device, torch.Tensor]:
            
            if not pin_memory:
                manager = get_memory_manager()
                tensor_buffer = manager.get_buffer(tensor.size(), tensor.dtype)
                tensor_buffer.copy_(tensor, non_blocking=False)
                return (tensor.device, tensor_buffer)
            packed = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=(device_module.is_available() and not tensor.is_sparse),
            )
            packed.copy_(tensor)
            return (tensor.device, packed)
 
        def unpack_from_cpu(packed: Tuple[torch.device, torch.Tensor]) -> torch.Tensor:
            device, tensor = packed
            manager = get_memory_manager()
            reloaded_t = torch.empty(tensor.size(), dtype=tensor.dtype, device=device)
            reloaded_t.copy_(tensor, non_blocking=pin_memory)
            manager.return_buffer(tensor)
            return tensor.to(device, non_blocking=pin_memory)

        super().__init__(pack_to_cpu, unpack_from_cpu)

# class CheckpointWrapper(nn.Module):
#     """
#     一个包裹模块，用于对单个 Transformer Block 应用激活重计算 (Activation Checkpointing)。
#     它的 `forward` 方法会通过 `torch.utils.checkpoint.checkpoint` 来调用被包裹的模块。
#     """
#     def __init__(self, module_to_wrap: nn.Module):
#         super().__init__()
#         self.module_to_wrap = module_to_wrap

#     def forward(self, *args, **kwargs):
#         """
#         通过 checkpoint 调用被包裹的模块。
#         - checkpoint 会在前向传播时不保存中间激活值，在反向传播时重新计算它们 [citation: 3][citation: 10]。
#         - use_reentrant=False 是推荐的现代用法，通常效率更高。
#         """
#         return checkpoint(self.module_to_wrap, *args, **kwargs, use_reentrant=False)

# class OffloadWrapper(nn.Module):
#     """
#     一个包裹模块，用于将激活值卸载 (Offload) 到 CPU。
#     它利用 `torch.autograd.graph.save_on_cpu` 上下文管理器来实现。
#     """
#     def __init__(self, module_to_wrap: nn.Module):
#         super().__init__()
#         self.module_to_wrap = module_to_wrap

#     def forward(self, *args, **kwargs):
#         with save_on_cpu():
#             return self.module_to_wrap(*args, **kwargs)
        
def offload(forward_func):

    def wrapped_forward(self, *args, **kwargs):
        with save_on_cpu():
            return forward_func(self, *args, **kwargs)
    return wrapped_forward




# --- 步骤 2: 重构 TransformerGeneralMixin ---

class TransformerGeneralMixin:
    """
    一个提供高级内存优化功能的 Mixin 类。
    它采用模块化包裹的方式来启用激活重计算和卸载，避免了直接修改方法。
    """

    def enable_activation_optimizations(
        self,
        blocks: nn.ModuleList,
        enable_checkpointing: bool = True,
        enable_offloading: bool = False
    ):
        """
        统一的入口函数，用于启用激活优化。

        Args:
            blocks (nn.ModuleList): 包含所有 Transformer 层的 ModuleList。
            enable_checkpointing (bool): 是否启用激活重计算。
            enable_offloading (bool): 是否启用激活卸载。
        """
        # if not (enable_checkpointing or enable_offloading):
        #     return
        enable_offloading = False
        # 从配置中获取详细参数
        from teletron.utils import get_args
        args = get_args()

        # Checkpointing 相关配置
        recompute_method = getattr(args, 'recompute_method', 'block')
        recompute_num_layers = getattr(args, 'recompute_num_layers', 0) if enable_checkpointing else 0

        print("Applying activation optimizations...")
        if enable_checkpointing:
            print(f"  - Checkpointing enabled: method='{recompute_method}', num_layers={recompute_num_layers}")
        if enable_offloading:
            print("  - Offloading enabled for all layers.")

        for i in range(len(blocks)):
            module_to_wrap = blocks[i]
            should_checkpoint_this_layer = False
            if enable_checkpointing and recompute_num_layers > 0:
                if recompute_method == 'block':
                    if i < recompute_num_layers:
                        should_checkpoint_this_layer = True
                elif recompute_method == 'uniform':
                    should_checkpoint_this_layer = True
                else:
                    raise ValueError(f"Invalid activation recompute method {recompute_method}.")
            if should_checkpoint_this_layer:
                if enable_offloading :
                    module_to_wrap.forward = partial(checkpoint, module_to_wrap.forward, use_reentrant=False)
                    module_to_wrap.forward = offload(module_to_wrap.forward)

                else:
                    module_to_wrap.forward = partial(checkpoint, module_to_wrap.forward, use_reentrant=False)
            
            if module_to_wrap is not blocks[i]:
                blocks[i] = module_to_wrap

    def set_input_tensor(self, x):
        return None
