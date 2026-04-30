"""Inject TeleBoost-only symbols into upstream verl namespaces.

Upstream verl@v0.4.0 doesn't ship a few APIs that recipe/dancegrpo/* uses:
  - verl.utils.debug: marked_timer, simple_timer, ProfilerConfig, WorkerProfiler,
                      WorkerProfilerExtension
  - verl.utils.device: get_device_id
  - verl.workers.reward_manager: register

Rather than rewrite every recipe import, we patch the upstream modules at
runtime so the symbols exist. teleboost-internal modules use the
teleboost.* paths directly.
"""
from __future__ import annotations


def apply() -> None:
    # verl.utils.debug
    import verl.utils.debug as _vud
    from teleboost.utils.debug import (
        ProfilerConfig,
        WorkerProfiler,
        WorkerProfilerExtension,
        marked_timer,
        simple_timer,
    )
    for name, value in [
        ("marked_timer", marked_timer),
        ("simple_timer", simple_timer),
        ("ProfilerConfig", ProfilerConfig),
        ("WorkerProfiler", WorkerProfiler),
        ("WorkerProfilerExtension", WorkerProfilerExtension),
    ]:
        if not hasattr(_vud, name):
            setattr(_vud, name, value)

    # verl.utils.device
    import verl.utils.device as _vudev
    from teleboost.utils.device import get_device_id, get_nccl_backend
    for name, value in [("get_device_id", get_device_id), ("get_nccl_backend", get_nccl_backend)]:
        if not hasattr(_vudev, name):
            setattr(_vudev, name, value)

    # verl.utils.model
    import verl.utils.model as _vum
    from teleboost.utils.model_extras import convert_weight_keys
    if not hasattr(_vum, "convert_weight_keys"):
        _vum.convert_weight_keys = convert_weight_keys

    # verl.workers.reward_manager
    import verl.workers.reward_manager as _vrm
    from teleboost.workers.reward_manager.registry import register, get_reward_manager_cls
    if not hasattr(_vrm, "register"):
        _vrm.register = register
    if not hasattr(_vrm, "get_reward_manager_cls"):
        _vrm.get_reward_manager_cls = get_reward_manager_cls

    # verl.utils.dataset.rl_dataset (wan-specific extras)
    import verl.utils.dataset.rl_dataset as _vrl_ds
    from teleboost.utils.dataset._wan_collate import wan_preprocessed_collate_function
    if not hasattr(_vrl_ds, "wan_preprocessed_collate_function"):
        _vrl_ds.wan_preprocessed_collate_function = wan_preprocessed_collate_function
