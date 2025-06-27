from .registry import Registry
from .vast.parallel_vast_model import ParallelVastModel
from .wan.parallel_wan_model import ParallelWanModel



registor = Registry("model")
registor.register(ParallelVastModel)
registor.register(ParallelWanModel)


def build_model(name,config=None):
    if config is None:
        return registor.build(name)
    else:
        return registor.build(name,config)


