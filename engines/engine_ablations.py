from __future__ import annotations

from driver import ENGINES
from model_loader import LoadedModel
from runner import PaddedRunner, SingleRunner
from scheduler import IterationLevelScheduler


def IterNaiveEngine(lm: LoadedModel, **kwargs) -> IterationLevelScheduler:
    return IterationLevelScheduler(SingleRunner(lm), **kwargs)


def IterPaddedEngine(lm: LoadedModel, **kwargs) -> IterationLevelScheduler:
    return IterationLevelScheduler(PaddedRunner(lm), **kwargs)


ENGINES["iter_naive"] = IterNaiveEngine
ENGINES["iter_padded"] = IterPaddedEngine
