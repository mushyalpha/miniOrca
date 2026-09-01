"""§3 C2 case 3: "each request is in the different phase: initiation or
increment." A batch containing both an INITIATION and an INCREMENT request
must run in ONE forward, not two -- that's the case static batching cannot
represent at all, and the case nano-vllm-style "schedule prefills XOR
decodes" schedulers deliberately avoid. It's a core claim of the paper, and
`select()` mixing them is more faithful than schedulers that don't.

This is a standing guard, not a one-off check: `selective_batching.validate()`
and `test_selective_tiny.py` already assert this for a hand-built batch, but
that's only a guarantee about the *runner* accepting a mixed batch if handed
one. This test guarantees the *scheduler* actually produces one during a
real run -- so if a future change (e.g. teaching a runner to reject mixed
batches, or "simplifying" select() to schedule phases separately) quietly
reintroduces the split, this fails immediately instead of only showing up
as a subtle throughput regression three files away.
"""
from __future__ import annotations

import engines  # noqa: F401
from driver import ENGINES, VirtualClock, run
from test_selective_tiny import make_tiny_lm
from trace import Trace, TraceRequest, materialize_prompts


def make_trace(lm):
    reqs = [
        TraceRequest(rid=0, n_input_tokens=8, max_gen_tokens=20, arrival_time=0.0),
        TraceRequest(rid=1, n_input_tokens=15, max_gen_tokens=20, arrival_time=0.0),
        # arrives while req0/req1 are mid-decode -- its INITIATION iteration
        # must land in the same forward as their INCREMENT iterations.
        TraceRequest(rid=2, n_input_tokens=6, max_gen_tokens=5, arrival_time=1e-4),
    ]
    trace = Trace(requests=reqs, arrival_rate=1.0, seed=0)
    materialize_prompts(trace, lm.tokenizer, seed=0)
    return trace


def test_mixed_phase_batch(engine_name: str) -> None:
    lm = make_tiny_lm()
    engine = ENGINES[engine_name](lm, max_bs=8)
    mc = run(engine, make_trace(lm), clock=VirtualClock())

    mixed_iters = [it.index for it in mc.iterations
                   if it.n_initiation > 0 and it.n_increment > 0]
    assert mixed_iters, (
        f"{engine_name}: no iteration ever mixed an INITIATION request with "
        f"an INCREMENT request. Either the trace stopped exercising case 3, "
        f"or the scheduler regressed into scheduling prefills and decodes "
        f"separately (the exact nano-vllm-style split this guards against).")
    print(f"  {engine_name}: PASS -- mixed-phase batch at iteration(s) {mixed_iters}")


if __name__ == "__main__":
    print("=== §3 C2 case 3: mixed INITIATION+INCREMENT batch, standing guard ===")
    for name in ("orca", "iter_naive", "iter_padded"):
        test_mixed_phase_batch(name)
    print("\nPASS: none of the iteration-level engines split prefill from decode.")
