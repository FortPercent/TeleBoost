from .default_sampler import DefaultSampler
from teletron.datasets.registry import Registry, build_module

SAMPLERS = Registry()

SAMPLERS.register_module(DefaultSampler)

def build_sampler(params_or_type, *args, **kwargs):
    return build_module(SAMPLERS, params_or_type, *args, **kwargs)