"""Inject TeleBoost-only debug helpers into ``verl.utils.debug`` namespace.

Recipe code uses ``from verl.utils.debug import marked_timer``, but upstream
verl@v0.4.0 doesn't define it. Rather than rewrite every recipe import, we
patch the verl module so the symbol exists.
"""
from __future__ import annotations


def apply() -> None:
    import verl.utils.debug as _vud

    if not hasattr(_vud, "marked_timer"):
        from teleboost.utils.debug._marked_timer import marked_timer
        _vud.marked_timer = marked_timer
