from .registry import Registry
from .teleai.parallel_teleai_model import ParallelTeleaiModel
from .wan.parallel_wan_model import ParallelWanModel
from .causwan import CausalDiffusion
from .wan.dpo_wan_model import WanTrainingModule


registor = Registry("model")
registor.register(ParallelTeleaiModel)
registor.register(ParallelWanModel)
registor.register(CausalDiffusion)
registor.register(WanTrainingModule)
def build_model(name,config=None):
    if config is None:
        return registor.build(name)
    else:
        return registor.build(name,config)


