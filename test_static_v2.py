"""
Validates the left-padded StaticBatchingEngine against your own checklist:
  1. Correctness: batch of different-length prompts must match the
     no_batch oracle exactly, greedy, fp32 CPU, tiny random Qwen2 (same
     modeling_qwen2 code path as Qwen2.5-0.5B, no network needed).
  2. Waste shape: with realistic length ranges, pad_tokens and
     finished_tokens should both be substantial, not near zero.
Absolute-magnitude checks (hold_after_completion in seconds, not ms) need
the real model's real per-iteration latency and are checked separately in
test_static_real.py against actual Qwen2.5-0.5B, since a tiny random model
runs iterations in microseconds regardless of scheduling correctness.
"""
from __future__ import annotations

import random

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import engines  # noqa: F401 -- import side effects populate driver.ENGINES
from driver import ENGINES, VirtualClock, run
from metrics import report
from model_loader import LoadedModel
from trace import Trace, TraceRequest, materialize_prompts

torch.manual_seed(0)
VOCAB = 2000


def make_tiny_lm() -> LoadedModel:
    cfg = Qwen2Config(
        vocab_size=VOCAB, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
        max_position_embeddings=1024, rms_norm_eps=1e-6,
        pad_token_id=0, eos_token_id=1,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()

    class FakeTok:
        pad_token_id = 0
        vocab_size = VOCAB
        all_special_ids = [0, 1]

    return LoadedModel(
        model=model, tokenizer=FakeTok(), device=torch.device("cpu"), dtype=torch.float32,
        eos_id=1, n_layers=cfg.num_hidden_layers, n_heads=cfg.num_attention_heads,
        n_kv_heads=cfg.num_key_value_heads, head_dim=cfg.hidden_size // cfg.num_attention_heads,
        hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size,
    )


# ------------------------------------------------------------- check 1
def check_correctness(lm: LoadedModel) -> None:
    print("=== check 1: left-padding correctness vs no_batch oracle ===")
    lens = [5, 12, 40]
    gens = [6, 6, 6]
    reqs = [
        TraceRequest(rid=i, n_input_tokens=L, max_gen_tokens=G, arrival_time=0.0)
        for i, (L, G) in enumerate(zip(lens, gens))
    ]
    trace = Trace(requests=reqs, arrival_rate=1.0, seed=0)
    materialize_prompts(trace, lm.tokenizer, seed=0)

    oracle = ENGINES["no_batch"](lm)
    mc_oracle = run(oracle, trace, clock=VirtualClock(), check_fcfs=True)
    ids_oracle = {r.rid: r.generated_ids for r in mc_oracle.returned}

    static = ENGINES["static"](lm, max_bs=3)
    mc_static = run(static, trace, clock=VirtualClock(), check_fcfs=True)
    ids_static = {r.rid: r.generated_ids for r in mc_static.returned}

    all_match = True
    for rid in sorted(ids_oracle):
        match = ids_oracle[rid] == ids_static[rid]
        all_match &= match
        print(f"  req {rid} (len={lens[rid]:3d}): {'MATCH' if match else 'MISMATCH'}  "
              f"oracle={ids_oracle[rid]}  static={ids_static[rid]}")
    if not all_match:
        print("  MISMATCH -- dumping position_ids diagnostic for the shortest row")
        raise AssertionError("static batching diverged from the no_batch oracle")
    print("  PASS: all rows match under left-padding + explicit position_ids.\n")


# ------------------------------------------------------------- check 2
def check_waste_shape(lm: LoadedModel) -> None:
    print("=== check 2: waste shape at realistic length ranges ===")
    rng = random.Random(0)
    n = 12
    reqs = []
    for i in range(n):
        n_in = rng.randint(32, 512)
        n_out = rng.randint(1, 128)
        reqs.append(TraceRequest(rid=i, n_input_tokens=n_in, max_gen_tokens=n_out, arrival_time=0.0))
    trace = Trace(requests=reqs, arrival_rate=1.0, seed=0)
    materialize_prompts(trace, lm.tokenizer, seed=0)

    lens = [r.n_input_tokens for r in reqs]
    gens = [r.max_gen_tokens for r in reqs]
    print(f"  input lens : {lens}  (mean {sum(lens)/n:.0f}, max {max(lens)})")
    print(f"  output lens: {gens}  (mean {sum(gens)/n:.1f}, max {max(gens)})")

    static = ENGINES["static"](lm, max_bs=n)
    mc = run(static, trace, clock=VirtualClock(), check_fcfs=True)
    rep = report(mc, label=f"static max_bs={n}, realistic lengths")

    prefill_pad_pct = 100 * rep["pad_token_slots"] / (rep["useful_token_slots"] + rep["pad_token_slots"])
    print(f"\n  prefill pad %          : {prefill_pad_pct:.1f}% "
          f"(expect roughly (max_len - mean_len)/max_len ~ "
          f"{100*(max(lens)-sum(lens)/n)/max(lens):.1f}%)")
    decode_iters = rep["n_iterations"] - 1   # minus the one prefill iteration
    expected_waste_pct = 100 * (1 - (sum(gens) / n) / max(gens)) if max(gens) else 0.0
    print(f"  finished-row waste     : {rep['finished_token_slots']} slot-iters "
          f"(expect roughly (1 - mean_gen/max_gen) ~ {expected_waste_pct:.1f}% of decode slot-iters)")

    assert rep["pad_token_slots"] > 0, "expected nonzero prefill padding -- did the batch shrink?"
    assert rep["finished_token_slots"] > 0, "expected nonzero finished-row waste -- did the batch shrink?"
    print("  PASS: both waste mechanisms are nonzero, as expected for heterogeneous lengths.\n")


if __name__ == "__main__":
    lm = make_tiny_lm()
    check_correctness(lm)
    check_waste_shape(lm)
