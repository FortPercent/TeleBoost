
from abc import abstractmethod
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class TransformerModel(nn.Module):
    """
    Base TeleTron transformer model definition.

    Args:
        config: Transformer config
    """
    def __init__(self, config):
        """
        Initialize transformer model.
        """
        self.config = config
        self.num_layers = config.num_layers
        self.blocks = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.hidden_size) for _ in range(self.num_layers)]
        )
        self.activation_recompute_method = config.activation_recompute_method
        self.activation_offload_method = config.activation_offload_method

    def _get_block(self, layer_number: int):
        return self.blocks[layer_number]    

    def _updata_args(self, output, args):
        if isinstance(output, tuple):
            return output + args[len(output):]
        else:
            return (output,) + args[1:]

    def _checkpointed_forward(self, *args):
        recompute_num_layers = self.config.recompute_num_layers

        def create_custom_forward(start, end):
            def custom_forward(*args):
                for index in range(start, end):
                    block = self._get_block(index)
                    output = block(*args)
                    args = self._updata_args(output, args)
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
                args = self._updata_args(output, args)
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
                args = self._updata_args(output, args)
        else:
            raise ValueError(f"Invalid activation recompute method {self.recompute_method}.")

        return output 

    def forward_transformer_blocks(self, *args, **kwargs):

        if self.config.recompute_granularity == "full"  and self.training:
            output = self._checkpointed_forward(*args)
        else:
            for block in self.blocks:
                output = block(*args)
                args = self._updata_args(output, args)
        return output


    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def pre_forward_transformer_blocks(self):
        """
        Process all input befor transformer blocks
        """
        raise NotImplementedError("pre_forward_transformer_blocks function not implemented.")
        
    @abstractmethod
    def post_forward_transformer_blocks(self):
        """
        Process all output after transformer blocks
        """
        raise NotImplementedError("post_forward_transformer_blocks function not implemented.")