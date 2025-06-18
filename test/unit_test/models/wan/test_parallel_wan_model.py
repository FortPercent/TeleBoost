import pytest 
import torch.nn.functional as F
from typing import Tuple, Optional, Callable
from unittest import TestCase
from unittest.mock import patch, Mock
# import torch.multiprocessing as mp 
from multiprocessing import Process
import multiprocessing as mp 
import argparse
from unit_test.test_utils import spawn
import logging
from teletron.models.wan import ParallelWanModel, WanModel
from teletron.core import initialize_model_parallel

# Configure logging
logging.basicConfig(level=logging.DEBUG,
format='%(asctime)s - %(levelname)s - %(message)s')

class WanParams:
    patch_size: Tuple[int] = (1, 2, 2)
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    activation_func: Callable = F.gelu
    in_channels: int = 36
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 1
    cross_attn_norm: bool = True
    qk_norm: Optional[str] = "rms_norm_across_heads"
    eps: float = 1e-6
    image_dim: int = 1280
    added_kv_proj_dim: int = 5120
    rope_max_seq_len: int = 1024
    has_image_input: bool = True
    has_image_pos_emb: bool = False


WAN_MODEL_SUCCESS = "Parallel Wan model forward test success"
WAN_MODEL_FAIL = "Parallel Wan model forward test fail"


@patch("teletron.get_args")
def parallel_wan_model_testing(mock_get_args, rank, world_size, q):
    args = Mock()
    args.recompute_method = "block"
    args.recompute_granularity = "full"
    args.recompute_num_layers = 1
    mock_get_args.return_value = args

    cp_size = world_size
    torch.distributed.init_process_group(world_size=world_size, rank=cp_rank)
    initialize_model_parallel(
            tensor_model_parallel_size = 1,
            pipeline_model_parallel_size = 1,
            virtual_pipeline_model_parallel_size = None,
            pipeline_model_parallel_split_rank = None,
            use_sharp = False,
            context_parallel_size = cp_size,
            expert_model_parallel_size = 1,
            nccl_communicator_config_path = None,
            distributed_timeout_minutes = 30,
        )
    wanConfig = WanParams()
    wan_model = WanModel(
            dim=wanConfig.num_attention_heads * wanConfig.attention_head_dim,
            in_dim=wanConfig.in_channels,
            ffn_dim=wanConfig.ffn_dim,
            out_dim=wanConfig.out_channels,
            text_dim=wanConfig.text_dim,
            freq_dim=wanConfig.freq_dim,
            eps=wanConfig.eps,
            patch_size=wanConfig.patch_size,
            num_heads=wanConfig.num_attention_heads,
            num_layers=wanConfig.num_layers,
            has_image_input=wanConfig.has_image_input,
            has_image_pos_emb=wanConfig.has_image_pos_emb
        )
    parallel_wan_model = ParallelWanModel(
            dim=wanConfig.num_attention_heads * wanConfig.attention_head_dim,
            in_dim=wanConfig.in_channels,
            ffn_dim=wanConfig.ffn_dim,
            out_dim=wanConfig.out_channels,
            text_dim=wanConfig.text_dim,
            freq_dim=wanConfig.freq_dim,
            eps=wanConfig.eps,
            patch_size=wanConfig.patch_size,
            num_heads=wanConfig.num_attention_heads,
            num_layers=wanConfig.num_layers,
            has_image_input=wanConfig.has_image_input,
            has_image_pos_emb=wanConfig.has_image_pos_emb
        )
    parallel_wan_model.load_state_dict(wan_model.state_dict())

    input_dict = torch.load("test_data/transformer_input.pt")
    wan_model_output = wan_model(**input_dict)

    input_dict = torch.load("test_data/transformer_input.pt")
    parallel_wan_model_output = parallel_wan_model(**input_dict)

    if torch.allclose(wan_model_output, parallel_wan_model_output):
        q.put("{WAN_MODEL_SUCCESS} rank{rank}")
    else:
        q.put("{WAN_MODEL_FAIL} rank{rank}")
    



class testParallelWanModel(TestCase):
    def test_forward(self):
        cp_size = 4
        spawn(cp_size, parallel_wan_model_testing)