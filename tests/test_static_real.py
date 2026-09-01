from __future__ import annotations

import time

import engines  # noqa: F401
from driver import ENGINES, WallClock, run
from metrics import report
from model_loader import load_model_and_tokenizer
from trace import Trace, TraceRequest, materialize_prompts

MODEL = "Qwen/Qwen2.5-0.5B"


def main() -> None:
    print(f"loading {MODEL} (fp32, CPU/MPS/CUDA whichever is available)...")
    t0 = time.time()
    lm = load_model_and_tokenizer(MODEL, force_fp32=True)
    print(f"  loaded on {lm.device} in {time.time()-t0:.1f}s")

    reqs = [
        TraceRequest(rid=0, n_input_tokens=48, max_gen_tokens=1, arrival_time=0.0),
        TraceRequest(rid=1, n_input_tokens=48, max_gen_tokens=48, arrival_time=0.0),
        TraceRequest(rid=2, n_input_tokens=48, max_gen_tokens=64, arrival_time=0.0),
        TraceRequest(rid=3, n_input_tokens=48, max_gen_tokens=32, arrival_time=0.0),
    ]
    trace = Trace(requests=reqs, arrival_rate=1.0, seed=0)
    materialize_prompts(trace, lm.tokenizer, seed=0)

    static = ENGINES["static"](lm, max_bs=4)
    t0 = time.time()
    mc = run(static, trace, clock=WallClock(), check_fcfs=True)
    wall = time.time() - t0
    rep = report(mc, label=f"static max_bs=4 on real {MODEL}, WallClock")

    r0 = next(r for r in mc.returned if r.rid == 0)
    print(f"\n  total wall time for the whole run     : {wall:.2f}s")
    print(f"  req 0 (max_gen=1) completion_time      : {r0.completion_time:.3f}s")
    print(f"  req 0 (max_gen=1) return_time           : {r0.return_time:.3f}s")
    print(f"  req 0 (max_gen=1) hold_after_completion : {r0.hold_after_completion:.3f}s")
    print(f"  batch's longest request needed {max(r.max_gen_tokens for r in reqs)} "
          f"decode iters -> req 0 held for ~{max(r.max_gen_tokens for r in reqs)-1} of them")

    assert r0.hold_after_completion > 0.5, (
        f"expected req 0's hold time to be clearly in the seconds range, "
        f"got {r0.hold_after_completion:.3f}s -- static batching may have "
        f"silently early-returned it (bug), or the model is running "
        f"implausibly fast for real Qwen2.5-0.5B")
    print("\n  PASS: hold_after_completion is O(seconds), matching C1's claim "
          "that static holds finished requests for the whole batch duration.")


if __name__ == "__main__":
    main()
