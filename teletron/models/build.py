from .registry import Registry
from .teleai.parallel_teleai_model import ParallelTeleaiModel



registor = Registry("model")
registor.register(ParallelTeleaiModel)


def build_model(name,config=None):
    if config is None:
        return registor.build(name)
    else:
        return registor.build(name,config)


