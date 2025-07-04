import os 
import torch
from typing import Tuple
from unittest import TestCase
from unittest.mock import patch, Mock
from unit_tests.test_utils import spawn
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG,
format='%(asctime)s - %(levelname)s - %(message)s')

class TeleaiParams:
    hidden_size: int = 5120
    in_channels: int = 36
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    eps: float = 1e-6
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    num_attention_heads: int = 40
    num_layers: int = 3
    has_image_input: bool = True
    has_image_pos_emb: bool = False


TELEAI_MODEL_FWD_SUCCESS = "Parallel Wan model forward test success"
TELEAI_MODEL_FWD_FAIL = "Parallel Wan model forward test fail"
TELEAI_MODEL_BWD_SUCCESS = "Parallel Wan model backward test success"
TELEAI_MODEL_BWD_FAIL = "Parallel Wan model backward test fail"

# @patch("teletron.utils.set_args")
@patch("teletron.utils.get_args")
def parallel_teleai_model_testing(rank, world_size, q, mock_teletron):
    from teletron.models.teleai import ParallelTeleaiModel,TeleaiModel
    from teletron.core.parallel_state import initialize_model_parallel_base 
    args = Mock()
    args.recompute_method = "block"
    args.recompute_granularity = "full"
    args.recompute_num_layers = 1
    args.activation_offload = True
    args.num_layers = 1 
    args.num_attention_heads = 40
    args.distributed_vae = False
    args.consumer_models_num = 1
    mock_teletron.return_value = args


    cp_size = world_size
    torch.distributed.init_process_group(world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    
    initialize_model_parallel_base(
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
    teleaiConfig = TeleaiParams()
    torch.manual_seed(1234)
    teleai_model = TeleaiModel(teleaiConfig).cuda().to(torch.bfloat16)
    torch.manual_seed(1234)
    parallel_teleai_model = ParallelTeleaiModel(teleaiConfig).cuda().to(torch.bfloat16)

    parallel_teleai_model.load_state_dict(teleai_model.state_dict())
    # teleai_params = dict(teleai_model.named_parameters())
    # teleai_parallel_params = dict(parallel_teleai_model.named_parameters())

    # from tensorwatch import watch_module_forward_backward, TensorWatch
    # watch_module_forward_backward(parallel_teleai_model)

    input_dict = torch.load("/nvfile-heatstorage/teleai-infra/litian/teletron-refactor/test/test_data/transformer_inputs.pt", map_location=f"cuda:{rank}")
    teleai_model_output = teleai_model(**input_dict)
    input_dict = torch.load("/nvfile-heatstorage/teleai-infra/litian/teletron-refactor/test/test_data/transformer_inputs.pt", map_location=f"cuda:{rank}")
    parallel_teleai_model_output = parallel_teleai_model(**input_dict)
    if is_close_by_normalized_euclid_dist(teleai_model_output, parallel_teleai_model_output):
        q.put(f"{TELEAI_MODEL_FWD_SUCCESS} rank{rank}")
    else:
        q.put(f"{TELEAI_MODEL_FWD_FAIL} rank{rank}")
    #TODO: test backward
    # test backward
    teleai_model_output.backward(torch.ones_like(teleai_model_output))
    parallel_teleai_model_output.backward(torch.ones_like(parallel_teleai_model_output))
    # TensorWatch.step()
    model_grads = {name: param.grad for name, param in teleai_model.named_parameters() if param.grad is not None}
    parallel_model_grads = {name: param.grad for name, param in parallel_teleai_model.named_parameters() if param.grad is not None}
    grad_allclose = True
    for name in model_grads:
        norm_euclid_dist = normalized_euclid_dist(model_grads[name], parallel_model_grads[name])
        if norm_euclid_dist < 0.02:
            continue
        else:
            logging.info(f"{name}: {norm_euclid_dist} {model_grads[name].norm()} {parallel_model_grads[name].norm()} rank{rank}")
            grad_allclose = False
    if grad_allclose:
        q.put(f"{TELEAI_MODEL_BWD_SUCCESS} rank{rank}")
    else:
        q.put(f"{TELEAI_MODEL_BWD_FAIL} rank{rank}")
    


def normalized_euclid_dist(output, parallel_output):
    teleai_norm = output.norm().item()
    parallel_norm = parallel_output.norm().item()
    euclid_dist = torch.norm(output - parallel_output)
    normalized_euclid_dist = 0.5 * euclid_dist / (teleai_norm + parallel_norm)
    return normalized_euclid_dist

def is_close_by_normalized_euclid_dist(output, parallel_output):
    teleai_norm = output.norm().item()
    parallel_norm = parallel_output.norm().item()
    euclid_dist = torch.norm(output - parallel_output)
    normalized_euclid_dist = 0.5 * euclid_dist / (teleai_norm + parallel_norm)
    if normalized_euclid_dist < 0.001:
        return True 
    else:
        return False 



class testParallelWanModel(TestCase):
    def test_forward_backward(self):
        cp_size = 2

        # os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        # from teletron.utils import set_args,get_args,validate_args
        # from teletron.train.arguments import parse_args
        # args = parse_args()
        # validate_args(args)
        # set_args(args)
        os.environ['WORLD_SIZE'] = str(cp_size)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12445'
        q = spawn(cp_size, parallel_teleai_model_testing)


        correct_responses = [f"{TELEAI_MODEL_BWD_SUCCESS} rank{rank}" for rank in range(cp_size)]
        correct_responses += [f"{TELEAI_MODEL_FWD_SUCCESS} rank{rank}" for rank in range(cp_size)]
        responses = []
        while not q.empty():
            res = q.get()
            responses.append(res)
        self.assertEqual(sorted(responses), correct_responses)
        #TODO: test backward



# if __name__ == '__main__':
#     import unittest
#     unittest.main()
