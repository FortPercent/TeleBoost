
import megatron.training
import megatron.core
from teletron.core.parallel_state import initialize_model_parallel_decorators
from teletron.core.training import setup_model_and_optimizer_decorators

def exe_adaptation():
    megatron.core.parallel_state.initialize_model_parallel = initialize_model_parallel_decorators(
        megatron.core.parallel_state.initialize_model_parallel
    )
    megatron.core.mpu = megatron.core.parallel_state

    megatron.training.training.setup_model_and_optimizer = setup_model_and_optimizer_decorators(
        megatron.training.training.setup_model_and_optimizer
    )

exe_adaptation()