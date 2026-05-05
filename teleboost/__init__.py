"""TeleBoost: video-generation RL training stack on top of upstream verl.

Importing this package implicitly applies every TeleBoost patch over verl
(via `teleboost.patches.apply()`). Recipe entrypoints (e.g.
recipe.teleboost.main_teleboost) import this module before importing any
verl symbol so that the patched verl namespace is in place.
"""
from teleboost import patches as _patches

_patches.apply()

# Force-import every TeleBoost reward manager so its `@register("name")`
# decorator runs and populates teleboost.workers.reward_manager.registry's
# REWARD_MANAGER_REGISTRY dict. Otherwise downstream callers of
# `get_reward_manager_cls("dancegrpo")` would hit "Unknown reward manager".
from teleboost.workers.reward_manager import dancegrpo as _dancegrpo  # noqa: F401
