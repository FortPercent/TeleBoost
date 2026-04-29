"""TeleBoost backport of profiler classes (not in upstream verl@v0.4.0).

Minimal no-op implementation: WorkerProfiler is a placeholder so the
profiler-extension code paths can be wired without actually profiling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ProfilerConfig:
    """Worker profiler config (Nsight-style). Unused by default."""

    all_ranks: bool = False
    ranks: Optional[list] = None
    discrete: bool = False

    def union(self, other: "ProfilerConfig") -> "ProfilerConfig":
        return ProfilerConfig(
            all_ranks=self.all_ranks or other.all_ranks,
            ranks=list(set(self.ranks or []) | set(other.ranks or [])),
            discrete=self.discrete or other.discrete,
        )

    def intersect(self, other: "ProfilerConfig") -> "ProfilerConfig":
        return ProfilerConfig(
            all_ranks=self.all_ranks and other.all_ranks,
            ranks=list(set(self.ranks or []) & set(other.ranks or [])),
            discrete=self.discrete and other.discrete,
        )


class WorkerProfiler:
    """No-op profiler stub. Subclass for real profiling."""

    def __init__(self, rank: int, config: Optional[ProfilerConfig] = None):
        self.rank = rank
        self.config = config or ProfilerConfig()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    @staticmethod
    def annotate(
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Callable:
        def decorator(func):
            return func

        return decorator


class WorkerProfilerExtension:
    """Mixin that exposes start_profile/stop_profile RPCs on a worker.

    The decorator is imported at runtime so we don't have a hard dep on
    verl's dispatch enums at module-load time.
    """

    def __init__(self, profiler: WorkerProfiler):
        self.profiler = profiler

    def start_profile(self) -> None:
        self.profiler.start()

    def stop_profile(self) -> None:
        self.profiler.stop()
