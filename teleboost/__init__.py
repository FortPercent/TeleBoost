"""TeleBoost: video-generation RL training stack on top of upstream verl.

Importing this package implicitly applies every TeleBoost patch over verl
(via `teleboost.patches.apply()`). Recipe entrypoints (e.g.
recipe.dancegrpo.main_dancegrpo) import this module before importing any
verl symbol so that the patched verl namespace is in place.
"""
from teleboost import patches as _patches

_patches.apply()
