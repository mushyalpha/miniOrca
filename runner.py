from __future__ import annotations

from typing import Optional, Protocol

import torch

from kv_manager import RequestKV
from model_loader import LoadedModel
from request import RequestState
from selective_batching import selective_batching_forward


class Runner(Protocol):
    def alloc(self, req: RequestState, capacity: Optional[int]) -> None: ...
    def run(self, batch: list[RequestState]) -> torch.Tensor: ...   
    def free(self, req: RequestState) -> None: ...


class SingleRunner:

    def __init__(self, lm: LoadedModel):
        self.lm = lm
        self.cache: dict[int, object] = {}

    def alloc(self, req: RequestState, capacity: Optional[int] = None) -> None:
        self.cache[req.rid] = None   
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

    def __init__(self, lm: LoadedModel):
        self.lm = lm
        self.kv: dict[int, RequestKV] = {}

    def alloc(self, req: RequestState, capacity: Optional[int] = None) -> None:

        self.kv[req.rid] = RequestKV(
            self.lm.n_layers, self.lm.n_kv_heads, self.lm.head_dim,
            self.lm.device, self.lm.dtype, capacity=capacity)

    @torch.inference_mode()
    def run(self, batch: list[RequestState]) -> torch.Tensor:
        from transformers import DynamicCache   

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
