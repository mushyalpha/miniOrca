"""Selective batching (§3 S2, Figure 5).

One flat [sum(L), H] tensor through every parameterised op; Split before
Attention, Merge after. "selective batching ... splits the batch and
processes each request individually for the Attention operation while
applying token-wise (instead of request-wise) batching to other operations."

We rewire dataflow around the loaded weights -- q_proj, mlp, layernorms are
used exactly as HF built them. No attention math is reimplemented.
"""
from __future__ import annotations

import importlib
from functools import lru_cache

import torch
import torch.nn.functional as F

from model_loader import LoadedModel
from request import Phase, RequestState


@lru_cache(maxsize=8)
def _ops(module_name: str):
    """Borrow the model family's own rope/gqa helpers so we can't drift from
    its convention (Llama/Qwen use 'half' rotation; other families don't)."""
    mod = importlib.import_module(module_name)
    return mod.apply_rotary_pos_emb, mod.repeat_kv


def _cos_sin(m, h, position_ids):
    if hasattr(m, "rotary_emb"):                      # transformers >= ~4.43
        return m.rotary_emb(h, position_ids)
    return m.layers[0].self_attn.rotary_emb(h, position_ids)   # older layout


def build_boundaries(batch: list[RequestState]):
    """cu_seqlens bookkeeping: which flat rows belong to which request.
    Built once, reused for every split and merge."""
    flat_ids, positions, bounds, off = [], [], [], 0
    for r in batch:
        toks = r.tokens_this_iter()
        assert toks, f"request {r.rid} has nothing to run"
        if r.phase is Phase.INITIATION:
            assert r.kv_len == 0, "initiation with a non-empty cache"
        flat_ids.extend(toks)
        positions.extend(range(r.position_offset, r.position_offset + len(toks)))
        bounds.append((r, off, off + len(toks)))
        off += len(toks)
    return flat_ids, positions, bounds


@torch.inference_mode()
def selective_batching_forward(lm: LoadedModel, batch: list[RequestState],
                               kv: dict) -> torch.Tensor:
    """Run one iteration. Returns [len(batch), vocab] -- the last-token
    logits, one row per request, in the order given."""
    m = lm.model.model
    apply_rope, repeat_kv = _ops(type(m).__module__)
    nh, nkv, hd, n_rep = lm.n_heads, lm.n_kv_heads, lm.head_dim, lm.n_rep

    flat_ids, positions, bounds = build_boundaries(batch)
    T = len(flat_ids)
    ids = torch.tensor([flat_ids], dtype=torch.long, device=lm.device)
    position_ids = torch.tensor([positions], dtype=torch.long, device=lm.device)

    h = m.embed_tokens(ids)                              # [1, T, H]
    cos, sin = _cos_sin(m, h, position_ids)

    for li, layer in enumerate(m.layers):
        sa = layer.self_attn

        # ---- batched over tokens, no notion of requests -----------------
        x = layer.input_layernorm(h)
        q = sa.q_proj(x).view(1, T, nh, hd)
        k = sa.k_proj(x).view(1, T, nkv, hd)
        v = sa.v_proj(x).view(1, T, nkv, hd)
        if hasattr(sa, "q_norm"):                        # Qwen3-style QK-norm
            q, k = sa.q_norm(q), sa.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # RoPE is per-token, not per-request: one flat call with per-token
        # position_ids. This is why the Split can start at Attention.
        q, k = apply_rope(q, k, cos, sin)

        # ---- SPLIT: the only op that needs a notion of requests ---------
        outs = []
        for r, s, e in bounds:
            K, V = kv[r.rid].extend(li, k[:, :, s:e], v[:, :, s:e])
            # is_causal is only correct when q_len == kv_len (fresh prefill).
            # Decode has q_len == 1 and needs no mask. Anything else (chunked
            # prefill) requires an explicit mask -- see assert in build_boundaries.
            outs.append(F.scaled_dot_product_attention(
                q[:, :, s:e], repeat_kv(K, n_rep), repeat_kv(V, n_rep),
                is_causal=(e - s > 1)))
        attn = torch.cat(outs, dim=2)                    # ---- MERGE ------

        # ---- batched over tokens again ----------------------------------
        h = h + sa.o_proj(attn.transpose(1, 2).reshape(1, T, nh * hd))
        h = h + layer.mlp(layer.post_attention_layernorm(h))

    h = m.norm(h)
    # Only the last row of each request produces a token; running lm_head on
    # a 512-token prompt's other 511 rows is pure waste.
    last = torch.tensor([e - 1 for _, _, e in bounds], device=lm.device)
    logits = lm.model.lm_head(h[:, last, :])[0]           # [B, vocab]

    for r, s, e in bounds:
        kv[r.rid].commit(e - s)
    return logits


# --------------------------------------------------------------- self-test
def validate(model_id: str = "Qwen/Qwen2.5-0.5B") -> None:
    """Run this before engine_orca. Two flat requests of different lengths
    must produce the same logits as each run alone through HF's own forward."""
    from kv_manager import RequestKV
    from model_loader import load_model_and_tokenizer

    lm = load_model_and_tokenizer(model_id, force_fp32=True, device="cpu")
    prompts = [[10, 20, 30, 40, 50], list(range(100, 117))]

    ref = []
    for p in prompts:
        out = lm.model(input_ids=torch.tensor([p]))
        ref.append(out.logits[0, -1])

    reqs = [RequestState(rid=i, prompt_ids=p, max_gen_tokens=4, arrival_time=0.0)
            for i, p in enumerate(prompts)]
    kv = {r.rid: RequestKV(lm.n_layers, lm.n_kv_heads, lm.head_dim,
                           lm.device, lm.dtype, capacity=r.max_tokens)
          for r in reqs}
    got = selective_batching_forward(lm, reqs, kv)

    for i in range(len(prompts)):
        d = (got[i] - ref[i]).abs().max().item()
        print(f"  req {i}: max |Δlogit| = {d:.2e}  "
              f"argmax {'OK' if got[i].argmax() == ref[i].argmax() else 'MISMATCH'}")
        assert d < 1e-3, "flat forward diverges from HF"

    # And now a mixed batch: one prefill + one decode, the case that is
    # impossible to batch without selective batching (§3 C2, case 3).
    for r in reqs:
        r.append_token(int(got[reqs.index(r)].argmax()), 0.0)
    reqs.append(RequestState(rid=99, prompt_ids=[7, 8, 9], max_gen_tokens=4,
                             arrival_time=0.0))
    kv[99] = RequestKV(lm.n_layers, lm.n_kv_heads, lm.head_dim, lm.device,
                       lm.dtype, capacity=reqs[-1].max_tokens)
    out = selective_batching_forward(lm, reqs, kv)
    assert out.shape == (3, lm.vocab_size)
    print("  mixed initiation+increment batch OK", out.shape)


if __name__ == "__main__":
    validate()
