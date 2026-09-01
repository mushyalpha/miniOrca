"""
End-to-end validation before trusting anything: run the SAME trace through
no_batch (the correctness oracle) and static (the padded, position_ids-
corrected baseline) via the shared driver, and assert identical
generated_ids. Uses a tiny randomly-initialized Qwen2 (no network,
same modeling_qwen2 code path Qwen2.5-0.5B uses) so this runs in
milliseconds. VirtualClock keeps it fast and deterministic.

Trace is deliberately adversarial for both claims under test:
  - req0/1/2 arrive simultaneously with DIFFERENT input lengths
    (6/11/4 tokens) -- exercises the position_ids fix for mixed-length
    prefill in one static batch.
  - req0/1/2 have DIFFERENT max_gen_tokens (5/2/8) -- forces req1 to
    finish early and sit as Figure 3 waste (finished_tokens) while req0
    and req2 keep running, and forces the batch's return to be held
    until req2 (the slowest) finishes -- C1's hold_after_completion.
  - req3 arrives much later, alone -- a clean late-joiner case for
    no_batch's queueing behavior.
"""
from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import engines  # noqa: F401  -- populates driver.ENGINES
from driver import ENGINES, VirtualClock, run
from metrics import check_iteration_fcfs, report
from model_loader import LoadedModel
from trace import Trace, TraceRequest, materialize_prompts

torch.manual_seed(0)
# materialize_prompts() samples ids from [1000, vocab_size) -- needs a vocab
# comfortably above 1000 for that range to be non-empty. Real tokenizers
# (~150k vocab) never hit this; noting it as a fragility in the hardcoded
# constant for small/toy vocabs, not fixing it since it's their file.
VOCAB = 2000


def make_tiny_lm() -> LoadedModel:
    cfg = Qwen2Config(
        vocab_size=VOCAB, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
        max_position_embeddings=256, rms_norm_eps=1e-6,
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


def make_trace(lm: LoadedModel) -> Trace:
    reqs = [
        TraceRequest(rid=0, n_input_tokens=6,  max_gen_tokens=5, arrival_time=0.0),
        TraceRequest(rid=1, n_input_tokens=11, max_gen_tokens=2, arrival_time=0.0),
        TraceRequest(rid=2, n_input_tokens=4,  max_gen_tokens=8, arrival_time=0.0),
        TraceRequest(rid=3, n_input_tokens=9,  max_gen_tokens=3, arrival_time=100.0),
    ]
    trace = Trace(requests=reqs, arrival_rate=1.0, seed=0)
    materialize_prompts(trace, lm.tokenizer, seed=0)
    return trace


def main() -> None:
    lm = make_tiny_lm()
    trace = make_trace(lm)

    no_batch = ENGINES["no_batch"](lm)
    mc_nb = run(no_batch, trace, clock=VirtualClock(), check_fcfs=True)
    rep_nb = report(mc_nb, label="no_batch")

    static = ENGINES["static"](lm, max_bs=4)
    mc_static = run(static, trace, clock=VirtualClock(), check_fcfs=True)
    rep_static = report(mc_static, label="static (max_bs=4)")

    # --- the actual correctness oracle check ---
    ids_nb = {r.rid: r.generated_ids for r in mc_nb.returned}
    ids_static = {r.rid: r.generated_ids for r in mc_static.returned}

    print("\n--- correctness check: no_batch vs static generated_ids ---")
    all_match = True
    for rid in sorted(ids_nb):
        match = ids_nb[rid] == ids_static[rid]
        all_match &= match
        status = "MATCH" if match else "MISMATCH"
        print(f"  req {rid}: {status}  nb={ids_nb[rid]}  static={ids_static[rid]}")
    assert all_match, "static batching diverged from the no-batch oracle"
    print("All requests match -- padding + explicit position_ids are correct.")

    # --- Figure 3's story, made concrete ---
    print("\n--- C1, made concrete (static engine) ---")
    for r in sorted(mc_static.returned, key=lambda r: r.rid):
        print(f"  req {r.rid}: completed at t={r.completion_time:.6f}, "
              f"returned at t={r.return_time:.6f}, "
              f"held {r.hold_after_completion:.6f}s after finishing")

    print(f"\nstatic useful_fraction = {rep_static['useful_fraction']*100:.1f}% "
          f"(pad={rep_static['pad_token_slots']}, finished-waste={rep_static['finished_token_slots']})")
    print(f"no_batch useful_fraction = {rep_nb['useful_fraction']*100:.1f}% "
          f"(single-flight, no batching waste possible by construction)")


if __name__ == "__main__":
    main()
