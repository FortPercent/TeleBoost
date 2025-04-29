
import megatron.training
import megatron.core
import teletron.core.parallel_state

def exe_adaptation():
    from .core.parallel_state import initialize_model_parallel_decorators
    megatron.core.parallel_state.initialize_model_parallel = initialize_model_parallel_decorators(
        megatron.core.parallel_state.initialize_model_parallel
    )
    megatron.core.mpu = megatron.core.parallel_state


    from .core.training import  setup_model_and_optimizer_decorators
    megatron.training.training.setup_model_and_optimizer = setup_model_and_optimizer_decorators(
        megatron.training.training.setup_model_and_optimizer
    )

exe_adaptation()