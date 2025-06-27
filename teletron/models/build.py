from .registry import Registry
from .vast.parallel_vast_model import ParallelVastModel



registor = Registry("model")
registor.register(ParallelVastModel)


def build_model(name,config=None):
    if config is None:
        return registor.build(name)
    else:
        return registor.build(name,config)


