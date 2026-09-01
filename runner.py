"""Runner: the *mechanism* half of an engine. Turns a list of RequestState
into next-token logits, given per-request K/V storage it owns. All
scheduling *policy* (who's in the batch, the memory budget) lives in
scheduler.py; a Runner never decides who runs, only how.

`alloc`/`free` are called exactly once per request, at admission and at
completion -- Algorithm 1's own boundary (Select() reserves at line 8,
frees at line 14-15), not a runner-specific batch lifecycle. This is why
none of the three runners below need a begin_batch/end_batch pair: there
is no batch-scoped state anywhere, only request-scoped state, some of
which happens to get temporarily reassembled into a dense tensor
(PaddedRunner) or never leaves its own buffer at all (FlatRunner).
"""
from __future__ import annotations

from typing import Optional, Protocol

import torch

from kv_manager import RequestKV
from model_loader import LoadedModel
from request import RequestState
from selective_batching import selective_batching_forward


class Runner(Protocol):
    def alloc(self, req: RequestState, capacity: Optional[int]) -> None: ...
    def run(self, batch: list[RequestState]) -> torch.Tensor: ...   # [len(batch), vocab]
    def free(self, req: RequestState) -> None: ...


class SingleRunner:
    """No batching at all, C1 and C2 both off. HF's own incremental cache,
    one request per forward call -- but `run()` still accepts an
    arbitrary-size batch and loops, so it can sit under iteration-level
    scheduling (iter_naive) and not just request-level (no_batch). The loop
    is the whole point: whatever wall-clock cost shows up here relative to
    FlatRunner/PaddedRunner *is* C2's contribution, with C1 held constant."""

    def __init__(self, lm: LoadedModel):
        self.lm = lm
        self.cache: dict[int, object] = {}

    def alloc(self, req: RequestState, capacity: Optional[int] = None) -> None:
        self.cache[req.rid] = None   # HF DynamicCache, built on first run()

    @torch.inference_mode()
    def run(self, batch: list[RequestState]) -> torch.Tensor:
        rows = []
        for req in batch:
            ids = torch.tensor([req.tokens_this_iter()], device=self.lm.device)
            out = self.lm.model(input_ids=ids, past_key_values=self.cache[req.rid],
                                use_cache=True)
            self.cache[req.rid] = out.past_key_values
            rows.append(out.logits[0, -1, :])
        return torch.stack(rows, dim=0)

    def free(self, req: RequestState) -> None:
        del self.cache[req.rid]


class FlatRunner:
    """Selective batching (§3 S2). Request-scoped K/V (RequestKV), one flat
    forward for the whole batch via selective_batching_forward. This is
    what engine_orca.py used directly before the scheduler/runner split;
    behavior is unchanged, just relocated."""

    def __init__(self, lm: LoadedModel):
        self.lm = lm
        self.kv: dict[int, RequestKV] = {}

    def alloc(self, req: RequestState, capacity: Optional[int] = None) -> None:
        self.kv[req.rid] = RequestKV(
            self.lm.n_layers, self.lm.n_kv_heads, self.lm.head_dim,
            self.lm.device, self.lm.dtype, capacity=capacity)

    def run(self, batch: list[RequestState]) -> torch.Tensor:
        return selective_batching_forward(self.lm, batch, self.kv)

    def free(self, req: RequestState) -> None:
        self.kv.pop(req.rid).free()


class PaddedRunner:
    """C1 on, C2 off: iteration-level scheduling with a dense [B, L, H]
    forward instead of selective batching -- the "naive continuous
    batching" pattern that predates it (per-request K/V, temporarily
    reassembled into a padded tensor and scattered back out every call).

    Two padding axes, independently, because a batch under Algorithm 1 can
    mix an INITIATION request with INCREMENT requests (§3 C2 case 3) and a
    closed static batch never can:
      - the NEW-token block is right-padded (rows contribute different
        numbers of real new tokens: a whole prompt vs. one token);
      - the PAST block is left-padded (so a row's true last new-token
        logit is always at a fixed offset, matching engine_static.py's
        convention).
    An explicit 4D additive attention mask is built by hand rather than
    relying on HF's 2D->4D auto-expansion, which assumes one scalar past
    length for the whole batch when drawing the causal boundary -- exactly
    wrong here, where left-padded past blocks have a different *real*
    length per row. Getting that wrong wouldn't show up as wasted compute,
    it would silently leak information across the causal boundary; see
    the standalone validation this was checked against before being wired
    in here (heterogeneous past+new lengths, compared logit-for-logit
    against each request run alone through HF's own forward).

    Cost, on purpose, not hidden: gathering/scattering is O(B x L) tensor
    copies every iteration, and a mixed batch pads every decode row out to
    the longest prefill in it. Both are real, and both are exactly the
    cost selective batching exists to eliminate -- that contrast is the
    point of this runner existing at all.
    """

    def __init__(self, lm: LoadedModel):
        self.lm = lm
        self.kv: dict[int, RequestKV] = {}

    def alloc(self, req: RequestState, capacity: Optional[int] = None) -> None:
        # Rebuilding a fresh *batched* tensor every run() call is inherent
        # to this runner and deliberately not optimized away (that copy is
        # the point of the C1-vs-C2 contrast). But the *persisted*
        # per-request store doesn't need to pay for that twice: honoring
        # `capacity` here (whatever the scheduler decided, same as
        # FlatRunner) means RequestKV.extend() writes into a preallocated
        # buffer at a fixed offset -- O(1) per iteration -- instead of
        # `torch.cat`-growing it -- O(current length) per iteration, i.e.
        # O(L^2) total over a generation for no reason. That growth cost
        # was never part of what this runner exists to demonstrate; it was
        # just an avoidable bug from reasoning "we rebuild anyway" without
        # separating "rebuild the batch view" from "grow the request's own
        # history."
        self.kv[req.rid] = RequestKV(
            self.lm.n_layers, self.lm.n_kv_heads, self.lm.head_dim,
            self.lm.device, self.lm.dtype, capacity=capacity)

    @torch.inference_mode()
    def run(self, batch: list[RequestState]) -> torch.Tensor:
        from transformers import DynamicCache   # local import: keep torch-only at module load

        lm = self.lm
        B = len(batch)
        new_toks = [r.tokens_this_iter() for r in batch]
        new_lens = [len(t) for t in new_toks]
        past_lens = [self.kv[r.rid].length for r in batch]
        L_new = max(new_lens)
        L_past = max(past_lens)
        pad_id = lm.tokenizer.pad_token_id
        device, dtype = lm.device, lm.dtype

        ids = torch.full((B, L_new), pad_id, dtype=torch.long, device=device)
        for row, toks in enumerate(new_toks):
            ids[row, :len(toks)] = torch.tensor(toks, device=device)

        position_ids = torch.tensor(
            [[past_lens[row] + j for j in range(L_new)] for row in range(B)],
            device=device)

        past_cache = None
        if L_past > 0:
            past_tuples = []
            for li in range(lm.n_layers):
                k_layer = torch.zeros((B, lm.n_kv_heads, L_past, lm.head_dim),
                                      dtype=dtype, device=device)
                v_layer = torch.zeros_like(k_layer)
                for row, req in enumerate(batch):
                    p = past_lens[row]
                    if p == 0:
                        continue
                    entry = self.kv[req.rid]
                    # entry.k[li] is sized to full `capacity` under
                    # reservation, not `entry.length` -- slice explicitly
                    # rather than assuming the stored tensor's own shape
                    # matches how much of it is actually committed.
                    k_layer[row, :, L_past - p:] = entry.k[li][0, :, :p]
                    v_layer[row, :, L_past - p:] = entry.v[li][0, :, :p]
                past_tuples.append((k_layer, v_layer))
            past_cache = DynamicCache(ddp_cache_data=tuple(past_tuples))

        neg = torch.finfo(dtype).min
        mask4d = torch.full((B, 1, L_new, L_past + L_new), neg, dtype=dtype, device=device)
        for row in range(B):
            p, n = past_lens[row], new_lens[row]
            if p > 0:
                mask4d[row, 0, :, L_past - p:L_past] = 0.0
            for j in range(L_new):
                upto = min(j + 1, n)
                if upto > 0:
                    mask4d[row, 0, j, L_past:L_past + upto] = 0.0

        out = lm.model(input_ids=ids, attention_mask=mask4d, position_ids=position_ids,
                       past_key_values=past_cache, use_cache=True)

        last_idx = torch.tensor([n - 1 for n in new_lens], device=device)
        logits = out.logits[torch.arange(B, device=device), last_idx, :]

        new_cache = out.past_key_values
        for li in range(lm.n_layers):
            layer = new_cache.layers[li]
            for row, req in enumerate(batch):
                n = new_lens[row]
                k_new = layer.keys[row:row + 1, :, L_past:L_past + n]
                v_new = layer.values[row:row + 1, :, L_past:L_past + n]
                self.kv[req.rid].extend(li, k_new, v_new)
        for row, req in enumerate(batch):
            self.kv[req.rid].commit(new_lens[row])

        return logits

    def free(self, req: RequestState) -> None:
        self.kv.pop(req.rid).free()
