"""TeleBoost-only context manager for timing inside training loops.

Upstream verl v0.4.0 doesn't ship `verl.utils.debug.marked_timer`; it was
introduced in a later upstream release as a unified API replacing the older
`_timer` context manager. teleboost_ray_trainer.py uses it. We backport the
simple non-NVTX variant here and patch it into `verl.utils.debug` at import
time via teleboost.patches.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Optional


@contextmanager
def marked_timer(
    name: str,
    timing_raw: Dict[str, float],
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
):
    """Time the wrapped block and accumulate elapsed seconds into ``timing_raw[name]``.

    The color/domain/category args are accepted for upstream-API parity; this
    fallback impl ignores them (the NVTX variant uses them for marker labels).
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timing_raw[name] = timing_raw.get(name, 0.0) + elapsed
