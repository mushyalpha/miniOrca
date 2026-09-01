"""
Reference engine: strict single-flight, batch size always 1.

This conforms to driver.Engine so it runs through the same outer loop as
every other engine, and its output is the correctness oracle — any other
engine's generated_ids must match this one exactly under greedy decoding
for the same trace.

Uses HF's own incremental cache directly (no flatten/split needed, since
there's never more than one request in flight) — the model math here is
untouched `model(...)`, only the outer loop differs from the other
engines.
"""
from __future__ import annotations

import torch

from driver import ENGINES, IterationResult
from metrics import IterationRecord
from model_loader import LoadedModel
from request import RequestState, State


class NoBatchEngine:
    def __init__(self, lm: LoadedModel, **kwargs):
        self.lm = lm
        self.pool: list[RequestState] = []
        self._current: RequestState | None = None

    def has_work(self) -> bool:
        return self._current is not None or any(r.is_selectable for r in self.pool)

    def _pick_next(self) -> RequestState:
        candidates = [r for r in self.pool if r.is_selectable]
        return min(candidates, key=lambda r: r.arrival_time)

    def step(self, now: float, iter_index: int) -> IterationResult:
        if self._current is None:
            self._current = self._pick_next()
            self._current.state = State.RUNNING

        req = self._current
        tokens = req.tokens_this_iter()
        is_initiation = len(tokens) > 1

        ids = torch.tensor([tokens], device=self.lm.device)
        with torch.inference_mode():
            out = self.lm.model(input_ids=ids, past_key_values=req.kv_cache, use_cache=True)
        req.kv_cache = out.past_key_values
        next_id = int(torch.argmax(out.logits[0, -1, :]))
        req.append_token(next_id, now)

        rec = IterationRecord(
            index=iter_index, start=now, end=now, batch_size=1,
            useful_tokens=len(tokens),
            n_initiation=1 if is_initiation else 0,
            n_increment=0 if is_initiation else 1,
        )

        returned: list[RequestState] = []
        if req.is_finished:
            self.pool.remove(req)
            returned.append(req)
            self._current = None

        return IterationResult(record=rec, returned=returned)


ENGINES["no_batch"] = NoBatchEngine
