"""Static (request-level) batching — the Figure 3 / FasterTransformer baseline.

Three properties this file exists to reproduce, in order of importance:
  1. The batch is CLOSED at admission. A request arriving one iteration later
     waits for the entire batch to drain (§3 C1: "the newly arrived request
     wait[s] until all requests in the current batch have finished").
  2. Rows that finish early keep getting computed. We feed them a pad token
     and throw away their logits, but the forward pass still runs on them --
     Figure 3's "-" entries. Shrinking the batch here would silently turn
     this into continuous batching and void the comparison.
  3. No early return. Every request in the batch is released at the same
     wall-clock moment, when the last row finishes.

Admission policy follows §6.2's description of the baseline scheduler:
"taking at most max batch size requests from the request queue". So we do
*not* block waiting for a full batch -- under low load static degenerates to
bs=1, which is exactly Figure 10a. The head-of-line blocking comes from the
batch being closed, not from waiting to fill it.
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import DynamicCache

from driver import ENGINES, IterationResult
from metrics import IterationRecord
from model_loader import LoadedModel
from request import Phase, RequestState, State


class StaticBatchingEngine:
    def __init__(self, lm: LoadedModel, max_bs: int = 8, n_slots: int = 0,
                 max_seq_len: int = 2048, slot_policy: str = "max_seq_len",
                 **_ignored):
        self.lm = lm
        self.max_bs = max_bs
        self.n_slots = n_slots            # 0 = unlimited
        self.max_seq_len = max_seq_len
        assert slot_policy in ("max_seq_len", "max_tokens")
        # §6.1: FasterTransformer preallocates per-request K/V "in proportion
        # to the max sequence length of the model (2048)", which is why it
        # OOMs at batch sizes Orca handles fine. Orca instead reserves
        # req.max_tokens (§4.2). Keep the default faithful to the baseline.
        self.slot_policy = slot_policy

        self.pool: list[RequestState] = []       # driver appends arrivals here
        self.batch: list[RequestState] = []      # closed batch, in flight
        self.cache: Optional[DynamicCache] = None
        self.attn_mask: Optional[torch.Tensor] = None
        self.next_tokens: Optional[torch.Tensor] = None   # [B]
        self.slots_held = 0

    # ------------------------------------------------------------ Engine API
    def has_work(self) -> bool:
        return bool(self.batch) or bool(self.pool)

    def step(self, now: float, iter_index: int) -> IterationResult:
        if not self.batch:
            self._admit(now)
        assert self.batch, "step() with nothing to run"

        if self.batch[0].phase is Phase.INITIATION:
            rec = self._initiation_iteration(now, iter_index)
        else:
            rec = self._increment_iteration(now, iter_index)
        rec.kv_slots_used = self.slots_held

        returned = self._maybe_close(now)
        return IterationResult(record=rec, returned=returned)

    # ------------------------------------------------------------- admission
    def _admit(self, now: float) -> None:
        """Take at most max_bs requests, FCFS. Once taken, they leave the
        pool -- for static batching the request genuinely leaves the
        scheduler's control until the engine hands the whole batch back."""
        self.pool.sort(key=lambda r: r.arrival_time)
        candidates = self.pool[: self.max_bs]

        # K/V budget. Only binds if you deliberately set n_slots low.
        if self.n_slots:
            while candidates and self._slots_for(candidates) > self.n_slots:
                candidates = candidates[:-1]
            if not candidates:
                raise RuntimeError(
                    f"n_slots={self.n_slots} too small for even one request "
                    f"under slot_policy={self.slot_policy} "
                    f"(needs {self._slots_for(self.pool[:1])})")

        self.batch = candidates
        self.pool = self.pool[len(candidates):]
        self.slots_held = self._slots_for(self.batch)
        for r in self.batch:
            r.state = State.RUNNING          # in flight for the whole batch
        self.cache = DynamicCache()

    def _slots_for(self, batch: list[RequestState]) -> int:
        if self.slot_policy == "max_seq_len":
            return len(batch) * self.max_seq_len
        return sum(r.max_tokens for r in batch)

    # ------------------------------------------------------------ iterations
    @torch.inference_mode()
    def _initiation_iteration(self, now: float, i: int) -> IterationRecord:
        """One forward over the whole padded prompt block: [B, L] -> [B, L, H].
        This is §2's initiation phase and §3's "canonical batching"."""
        lm, batch = self.lm, self.batch
        B = len(batch)
        lens = [r.n_input_tokens for r in batch]
        L = max(lens)
        pad_id = lm.tokenizer.pad_token_id

        ids = torch.full((B, L), pad_id, dtype=torch.long, device=lm.device)
        mask = torch.zeros((B, L), dtype=torch.long, device=lm.device)
        for row, r in enumerate(batch):
            # LEFT padding. With right padding, generated tokens would land
            # after the pad block and RoPE positions would be wrong. Left
            # padding also makes logits[:, -1] the true last token of every
            # row, which keeps the decode path trivial.
            ids[row, L - lens[row]:] = torch.tensor(r.prompt_ids, device=lm.device)
            mask[row, L - lens[row]:] = 1

        # Must be explicit: HF would otherwise use arange(L) and shift every
        # padded row's RoPE positions by its pad count, silently diverging
        # from the no_batch oracle.
        position_ids = (mask.cumsum(-1) - 1).clamp(min=0)

        out = lm.model(input_ids=ids, attention_mask=mask,
                       position_ids=position_ids,
                       past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.attn_mask = mask

        greedy = out.logits[:, -1, :].argmax(dim=-1)          # [B]
        for row, r in enumerate(batch):
            r.phase = Phase.INCREMENT
            r.append_token(int(greedy[row]), now)
        self.next_tokens = greedy

        useful = sum(lens)
        return IterationRecord(
            index=i, start=now, end=now, batch_size=B,
            useful_tokens=useful, pad_tokens=B * L - useful,
            finished_tokens=0, n_initiation=B, n_increment=0,
        )

    @torch.inference_mode()
    def _increment_iteration(self, now: float, i: int) -> IterationRecord:
        """One token for every row -- including rows that finished ten
        iterations ago. That is the whole point of this engine."""
        lm, batch = self.lm, self.batch
        B = len(batch)
        pad_id = lm.tokenizer.pad_token_id

        active = [not r.is_finished for r in batch]
        ids = self.next_tokens.clone()
        for row, alive in enumerate(active):
            if not alive:
                ids[row] = pad_id            # Figure 3's "-" input
        ids = ids.unsqueeze(1)               # [B, 1]

        # Mask the dead rows' new column off so their garbage K/V is never
        # attended to. We still pay for the QKV/MLP/attention compute --
        # we are removing the corruption, not the waste.
        new_col = torch.tensor([[1 if a else 0] for a in active],
                               dtype=torch.long, device=lm.device)
        self.attn_mask = torch.cat([self.attn_mask, new_col], dim=1)
        position_ids = (self.attn_mask.sum(-1, keepdim=True) - 1)

        out = lm.model(input_ids=ids, attention_mask=self.attn_mask,
                       position_ids=position_ids,
                       past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        greedy = out.logits[:, -1, :].argmax(dim=-1)

        for row, r in enumerate(batch):
            if active[row]:
                r.append_token(int(greedy[row]), now)
        self.next_tokens = greedy

        n_active = sum(active)
        return IterationRecord(
            index=i, start=now, end=now, batch_size=B,
            useful_tokens=n_active, pad_tokens=0,
            finished_tokens=B - n_active,           # <-- the Figure 3 metric
            n_initiation=0, n_increment=n_active,
        )

    # --------------------------------------------------------------- closing
    def _maybe_close(self, now: float) -> list[RequestState]:
        """No request is released until every row is done (§3 C1). The gap
        between req.completion_time and req.return_time is the latency
        penalty this engine exists to demonstrate."""
        if not all(r.is_finished for r in self.batch):
            return []
        returned = self.batch
        self.batch, self.cache, self.attn_mask, self.next_tokens = [], None, None, None
        self.slots_held = 0
        for r in returned:
            r.state = State.DONE
        return returned


ENGINES["static"] = StaticBatchingEngine
