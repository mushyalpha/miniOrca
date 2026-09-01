from __future__ import annotations

from driver import ENGINES
from model_loader import LoadedModel
from runner import FlatRunner
from scheduler import IterationLevelScheduler, KVDeadlock  # noqa: F401 -- re-exported

__all__ = ["OrcaEngine", "KVDeadlock"]


def OrcaEngine(lm: LoadedModel, **kwargs) -> IterationLevelScheduler:
    return IterationLevelScheduler(FlatRunner(lm), **kwargs)


ENGINES["orca"] = OrcaEngine
ENGINES["orca_no_reserve"] = lambda lm, **kw: OrcaEngine(lm, reserve=False, **kw)
ENGINES["orca_greedy_select"] = lambda lm, **kw: OrcaEngine(lm, fcfs_break=False, **kw)
