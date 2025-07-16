from .registry import Registry
from .teleai.parallel_teleai_model import ParallelTeleaiModel
from .wan.parallel_wan_model import ParallelWanModel



registor = Registry("model")
registor.register(ParallelTeleaiModel)
registor.register(ParallelWanModel)

def build_model(name,config=None):
    if config is None:
        return registor.build(name)
    else:
        return registor.build(name,config)


