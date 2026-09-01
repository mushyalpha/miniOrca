"""C1-vs-C2 isolation. Same scheduler (IterationLevelScheduler = Algorithm 1)
as engine_orca.py -- identical admission policy, identical K/V budget,
identical FCFS invariant -- paired with a different Runner:

  iter_naive  : SingleRunner  -- no batching of any kind (C1 on, C2 off).
                Each request in the selected batch gets its own separate
                HF forward call, looped in Python. Isolates the win from
                *not padding/holding finished rows* from the win from
                *batched GEMMs*.
  iter_padded : PaddedRunner  -- dense [B, L, H] forward, request-scoped
                K/V reassembled into a padded batch every iteration (C1 on,
                C2 off, but with real cross-request batching -- the "naive
                continuous batching" pattern that predates selective
                batching). A batch that mixes an INITIATION request with
                INCREMENT requests pays a real padding cost here that
                FlatRunner (orca) does not; that gap *is* the empirical
                case for selective batching, not an artifact.

Neither pads/holds finished rows the way engine_static.py does -- that's
C1, and it's identical across orca/iter_naive/iter_padded by construction,
since all three share the one scheduler.
"""
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
