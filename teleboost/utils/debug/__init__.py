"""TeleBoost extensions to verl.utils.debug.

Re-exports the backport symbols that upstream verl@v0.4.0 doesn't ship but
TeleBoost code (and recipe code via teleboost.patches injection) needs.
"""
from teleboost.utils.debug._marked_timer import marked_timer
from teleboost.utils.debug._profile import (
    ProfilerConfig,
    WorkerProfiler,
    WorkerProfilerExtension,
)
from teleboost.utils.debug._simple_timer import simple_timer

__all__ = [
    "marked_timer",
    "simple_timer",
    "ProfilerConfig",
    "WorkerProfiler",
    "WorkerProfilerExtension",
]
