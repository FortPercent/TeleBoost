"""TeleBoost backport of `simple_timer` (not in upstream verl@v0.4.0)."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict


@contextmanager
def simple_timer(name: str, timing_raw: Dict[str, float]):
    """Time the wrapped block and accumulate elapsed seconds into ``timing_raw[name]``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timing_raw[name] = timing_raw.get(name, 0.0) + elapsed
