
from abc import abstractmethod
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class TransformerGeneralMixin:
    def enable_activation_checkpointing(self, blocks):
        blocks.forward = self.checkpointed_forward_transformer_blocks

    # todo: kwargs are not updated
    def checkpointed_forward_transformer_blocks(self, *args, **kwargs):
        if self.recompute_granularity == "full"  and self.training:
            output = self._checkpointed_forward(*args)
        else:
            for block in self.blocks:
                output = block(*args)
                args = self._update_args(output, args)
        return output

    def _get_block(self, layer_number: int):
        return self.blocks[layer_number]    

    def _update_args(self, output, args):
        if isinstance(output, tuple):
            return output + args[len(output):]
        else:
            return (output,) + args[1:]

    def _checkpointed_forward(self, *args):
        recompute_num_layers = self.recompute_num_layers

        def create_custom_forward(start, end):
            def custom_forward(*args):
                for index in range(start, end):
                    block = self._get_block(index)
                    output = block(*args)
                    args = self._update_args(output, args)
                return output
            return custom_forward

        if self.activation_recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and
            # checkpoint the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            _layer_num = 0
            while _layer_num < self.num_layers:
                output = checkpoint(
                    create_custom_forward(_layer_num, _layer_num + recompute_num_layers),
                    *args,
                    use_reentrant=False,
                )
                args = self._update_args(output, args)
                _layer_num += recompute_num_layers

        elif self.activation_recompute_method == "block":
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for _layer_num in range(self.num_layers):
                if _layer_num < recompute_num_layers:
                    output = checkpoint(
                        create_custom_forward(_layer_num, _layer_num + 1),
                        *args,
                        use_reentrant=False,
                    )
                else:
                    block = self._get_block(_layer_num)
                    output = block(*args)
                args = self._update_args(output, args)
        else:
            raise ValueError(f"Invalid activation recompute method {self.recompute_method}.")

        return output 
