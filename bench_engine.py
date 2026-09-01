"""§6.1-style microbenchmark: isolates a Runner's per-iteration cost from
scheduling entirely. No driver, no admission, no arrivals -- fix a
homogeneous, all-decode-phase batch of N requests up front (one INITIATION
call to get every request into steady state), then time `runner.run(batch)`
alone, repeated, across batch sizes.

This deliberately measures decode-phase steady state only. Prefill cost at
these sequence lengths would dominate and drown out the per-iteration
signal this exists to isolate; end-to-end prefill+decode cost is what
run_experiments.py's full traces are for.

What this can tell you that a full run can't: PaddedRunner does two things
FlatRunner doesn't -- pads every row out to the batch's longest new-token
count (irrelevant here, all rows contribute exactly 1 token) and reassembles
per-request K/V into a dense tensor every call (a real O(B x L) copy, and
the whole reason this benchmark exists). SingleRunner does neither, but
also shares no compute across requests at all. Comparing all three across
batch size should show FlatRunner's ms/iteration growing sub-linearly with
B (shared kernel launch), SingleRunner's growing ~linearly (no sharing),
and PaddedRunner's growing faster than FlatRunner's as L grows (the copy
cost) even though both share the batched-GEMM benefit at a fixed L.

Usage:
    python bench_engine.py --tiny                     # offline, structural
    python bench_engine.py --model Qwen/Qwen2.5-0.5B   # real weights
"""
from __future__ import annotations

import argparse
import time

import torch

from model_loader import LoadedModel, load_model_and_tokenizer
from request import RequestState
from runner import FlatRunner, PaddedRunner, Runner, SingleRunner

RUNNERS = {"single": SingleRunner, "flat": FlatRunner, "padded": PaddedRunner}


def make_requests(lm: LoadedModel, batch_size: int, prompt_len: int,
                  n_gen: int, seed: int) -> list[RequestState]:
    g = torch.Generator().manual_seed(seed)
    lo, hi = 100, min(lm.vocab_size, 2000)
    reqs = []
    for i in range(batch_size):
        ids = torch.randint(lo, hi, (prompt_len,), generator=g).tolist()
        reqs.append(RequestState(rid=i, prompt_ids=ids, max_gen_tokens=n_gen,
                                 arrival_time=0.0))
    return reqs


def _step(runner: Runner, reqs: list[RequestState]) -> None:
    logits = runner.run(reqs)
    tok = logits.argmax(dim=-1)
    for row, r in enumerate(reqs):
        r.append_token(int(tok[row]), now=0.0)


def bench_one(runner_name: str, lm: LoadedModel, batch_size: int, prompt_len: int,
             n_iters: int, warmup: int, seed: int) -> float:
    """Returns ms/iteration, steady-state decode only."""
    runner = RUNNERS[runner_name](lm)
    reqs = make_requests(lm, batch_size, prompt_len, n_gen=warmup + n_iters + 1, seed=seed)

    for r in reqs:
        runner.alloc(r, capacity=r.max_tokens)
    _step(runner, reqs)                     # INITIATION -- excluded from timing
    for _ in range(warmup):
        _step(runner, reqs)                 # let any lazy init / cache warm-up settle

    if lm.device.type == "mps":
        torch.mps.synchronize()
    elif lm.device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _step(runner, reqs)
    if lm.device.type == "mps":
        torch.mps.synchronize()
    elif lm.device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    for r in reqs:
        runner.free(r)
    return 1000.0 * dt / n_iters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tiny", action="store_true",
                   help="offline random Qwen2, structural comparison only")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--n-iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runners", nargs="+", default=["single", "flat", "padded"],
                   choices=list(RUNNERS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    if a.tiny:
        from test_selective_tiny import make_tiny_lm
        lm = make_tiny_lm()
        label = "tiny-random-Qwen2 (structural, not representative of real ms/iter)"
    else:
        lm = load_model_and_tokenizer(a.model, force_fp32=a.fp32)
        label = f"{a.model} ({lm.device.type}, {lm.dtype})"

    print(f"=== bench_engine: {label} ===")
    print(f"{'batch':>6} | " + " | ".join(f"{r:>10}" for r in a.runners))

    rows = []
    for bs in a.batch_sizes:
        if "single" in a.runners and bs > 8 and a.tiny is False:
            pass  # single scales ~linearly; still fine to run, just slow -- caller's choice
        times = {}
        for name in a.runners:
            ms = bench_one(name, lm, bs, a.prompt_len, a.n_iters, a.warmup, a.seed)
            times[name] = ms
        rows.append({"batch_size": bs, **{f"{k}_ms_per_iter": v for k, v in times.items()}})
        print(f"{bs:>6} | " + " | ".join(f"{times[r]:>8.2f}ms" for r in a.runners))

    if a.out:
        import json
        with open(a.out, "w") as f:
            json.dump({"label": label, "prompt_len": a.prompt_len, "rows": rows}, f, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
