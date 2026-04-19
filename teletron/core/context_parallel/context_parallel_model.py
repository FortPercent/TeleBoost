import torch
import torch.nn as nn 
from megatron.core import mpu
from teletron.core.context_parallel.mappings import split_forward_gather_backward,\
        gather_forward_split_backward


class ContextParallelModelManager():
    def __init__(self, split_dim=1, gather_dim=1):
        self.cp_size = mpu.get_context_parallel_world_size()
        self.cp_group = mpu.get_context_parallel_group()
        self.split_dim = split_dim
        self.gather_dim = gather_dim
        # Stack so nested / sequential split→gather pairs (e.g. DPO chosen+rejected
        # with differing sequence lengths) don't clobber each other's pad state.
        self._pad_stack = []

    def split_input(self, x):
        # assert x is not parallel
        origin_length = x.shape[self.split_dim]
        if origin_length % self.cp_size != 0:
            padded_length = origin_length + self.cp_size - (origin_length % self.cp_size)
            x = self._pad(x, origin_length, padded_length)
            self._pad_stack.append(origin_length)
        else:
            self._pad_stack.append(None)

        x = split_forward_gather_backward(x, self.cp_group, dim=self.split_dim, grad_scale="none")
        return x

    def gather_output(self, output):
        output = gather_forward_split_backward(output, self.cp_group, dim=self.gather_dim, grad_scale="none")
        origin_length = self._pad_stack.pop() if self._pad_stack else None
        if origin_length is not None:
            output = output.narrow(self.gather_dim, 0, origin_length)
        return output

    def _pad(self, tensor, origin_length, padded_length):
        pad_size = int(padded_length - origin_length)
        if pad_size <= 0:
            return tensor
        pad = [0] * (2 * tensor.dim())
        pad[-(2 * self.split_dim + 1)] = pad_size
        return torch.nn.functional.pad(tensor, pad)

    # def context_parallel_forward_transformer_blocks(self, forward_func):
    #     @wraps(forward_func)
    #     def cp_forward_func(cp_args=[], cp_kwargs={}, regular_args=[], regular_kwargs={}):

            

