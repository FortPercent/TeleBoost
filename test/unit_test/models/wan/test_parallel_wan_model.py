import pytest 
import os 
import torch
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
import teletron

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


WAN_MODEL_FWD_SUCCESS = "Parallel Wan model forward test success"
WAN_MODEL_FWD_FAIL = "Parallel Wan model forward test fail"
WAN_MODEL_BWD_SUCCESS = "Parallel Wan model backward test success"
WAN_MODEL_BWD_FAIL = "Parallel Wan model backward test fail"

@patch("teletron.get_args")
def parallel_wan_model_testing(rank, world_size, q, mock_teletron):
    from teletron.models.wan import ParallelWanModel, WanModel
    from teletron.core import initialize_model_parallel
    args = Mock()
    args.recompute_method = "block"
    args.recompute_granularity = "full"
    args.recompute_num_layers = 1
    args.num_layers = 1 
    args.num_attention_heads = 40
    mock_teletron.return_value = args


    cp_size = world_size
    torch.distributed.init_process_group(world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
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
        ).cuda().to(torch.bfloat16)
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
        ).cuda().to(torch.bfloat16)
    parallel_wan_model.load_state_dict(wan_model.state_dict())

    input_dict = torch.load("/nvfile-heatstorage/teleai-infra/litian/teletron-refactor/test/test_data/transformer_inputs.pt", map_location=f"cuda:{rank}")
    wan_model_output = wan_model(**input_dict)

    input_dict = torch.load("/nvfile-heatstorage/teleai-infra/litian/teletron-refactor/test/test_data/transformer_inputs.pt", map_location=f"cuda:{rank}")
    parallel_wan_model_output = parallel_wan_model(**input_dict)
    if is_close_by_normalized_euclid_dist(wan_model_output, parallel_wan_model_output):
        q.put(f"{WAN_MODEL_FWD_SUCCESS} rank{rank}")
    else:
        q.put(f"{WAN_MODEL_FWD_FAIL} rank{rank}")
    #TODO: test backward
    # test backward
    wan_model_output.backward(torch.ones_like(wan_model_output))
    parallel_wan_model_output.backward(torch.ones_like(parallel_wan_model_output))

    model_grads = {name: param.grad for name, param in wan_model.named_parameters() if param.grad is not None}
    parallel_moedl_grads = {name: param.grad for name, param in parallel_wan_model.named_parameters() if param.grad is not None}
    for name in model_grads:
        if is_close_by_normalized_euclid_dist(model_grads[name], parallel_moedl_grads[name], True):
            grad_flag = True
        else:
            grad_flag = False
    

def is_close_by_normalized_euclid_dist(output, parallel_output):
    wan_norm = output.norm().item()
    parallel_norm = parallel_output.norm().item()
    euclid_dist = torch.norm(output - parallel_output)
    normalized_euclid_dist = 0.5 * euclid_dist / (wan_norm + parallel_norm)
    if normalized_euclid_dist < 0.001:
        return True 
    else:
        return False 



class testParallelWanModel(TestCase):
    def test_forward_backward(self):
        cp_size = 2
        os.environ['WORLD_SIZE'] = str(cp_size)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12445'
        q = spawn(cp_size, parallel_wan_model_testing)
        correct_responses = [f"{WAN_MODEL_FWD_SUCCESS} rank{rank}" for rank in range(cp_size)]
        responses = []
        while not q.empty():
            res = q.get()
            responses.append(res)
        self.assertEqual(sorted(responses), correct_responses)
        #TODO: test backward

