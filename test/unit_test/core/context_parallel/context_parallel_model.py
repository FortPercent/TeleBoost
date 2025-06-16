from unittest import TestCase
from teletron.core.context_parallel import ContextParallelModelManager
from teletron.core import initialize_model_parallel

def testContextParallelModelManager(q, cp_size, cp_rank):
    torch.distributed.init_process_group(world_size=cp_size, rank=cp_rank)
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
    
    cp_manager = ContextParallelModelManager()


def test_split_input(cp_manager, cp_size, cp_rank, q):
    with torch.no_grad():
        x_list = []
        for i in range(cp_size):
            x = torch.zeros((1, 100, 128)) + i
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
    x_split = cp_manager.split_input(x)
    if torch.all(x_split == x_list[cp_rank]):
        q.put("success")
    else:
        q.put("fail")
    



class testContextParallelModelManager(TestCase):
    def setUp(self):
        cp_size = 4
