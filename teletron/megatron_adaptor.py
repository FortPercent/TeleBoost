
import megatron.training
import megatron.core
from teletron.core.parallel_state import initialize_model_parallel_decorators
from teletron.core.training import setup_model_and_optimizer_decorators
from teletron.core.distributed.distributed_data_parallel import DistributedDataParallel
from teletron.core.distributed.param_and_grad_buffer import start_grad_sync

def exe_adaptation():
    megatron.core.parallel_state.initialize_model_parallel = initialize_model_parallel_decorators(
        megatron.core.parallel_state.initialize_model_parallel
    )
    megatron.core.mpu = megatron.core.parallel_state

    megatron.training.training.setup_model_and_optimizer = setup_model_and_optimizer_decorators(
        megatron.training.training.setup_model_and_optimizer
    )

    megatron.core.distributed.DistributedDataParallel = DistributedDataParallel
    megatron.core.distributed.param_and_grad_buffer.ParamAndGradBuffer.start_grad_sync = start_grad_sync
    
exe_adaptation()