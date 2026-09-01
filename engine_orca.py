"""Orca: iteration-level scheduling (§3 S1) + selective batching (§3 S2).

This file is now just a composition: `IterationLevelScheduler` (Algorithm 1,
scheduler.py) paired with `FlatRunner` (selective batching, runner.py).
Nothing scheduling-related lives here anymore -- see scheduler.py for the
one and only implementation of Select(), shared with engine_ablations.py's
iter_naive and iter_padded, which are the same scheduler with a different
Runner. That sharing is the entire point of the split: previously
iter_naive would have needed to be a ~150-line copy of this file with the
forward call swapped out; now it's one line in engine_ablations.py.

pad_tokens and finished_tokens are 0 by construction for every engine built
from IterationLevelScheduler -- that is the comparison against engine_static,
independent of which Runner is underneath.
"""
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
