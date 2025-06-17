

import torch
import torch.nn as nn
from teletron.core.transformer import TransformerModel

class TestParams:
    hidden_size = 1024
    num_layers = 4
    recompute_granularity = "full"
    activation_recompute_method = "block"
    activation_offload_method = "test"
    recompute_num_layers = 4


def test_transformer_block_recompute():
    config = TestParams()
    model = TransformerModel(config)
    model.blocks = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.hidden_size) for _ in range(model.num_layers)]
        )
    model.train()
    x = torch.rand(16,1024)
    output_shape = x.shape
    with torch.autograd.profiler.profile(with_stack=True, use_cuda=False) as prof:
        output = model.forward_transformer_blocks(x)
        loss = output.sum()
        loss.backward()
    
    linear_events = [e for e in prof.key_averages() if "aten::linear" in e.key]
    real_linear_calls = sum(e.count for e in linear_events)
    if config.activation_recompute_method == "uniform":
        total_linear_calls = config.num_layers * 2
    elif config.activation_recompute_method == "full":
        total_linear_calls = config.num_layers + min(config.num_layers, config.recompute_num_layers)

    assert output.shape == output_shape
    assert real_linear_calls == total_linear_calls