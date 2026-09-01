"""One command, one model load, every engine, one clean comparison table.

Built for *reporting* results (writeup/portfolio/social), not for iterating
on the implementation -- see driver.py for running a single engine, and
bench_engine.py for isolating a runner's per-iteration cost. This script
just loads the model once (saves ~5 redundant weight loads) and runs the
same trace through every requested engine back to back, so the numbers are
guaranteed to be dtype-matched and trace-matched -- the two things that
caused confusion earlier when engines were run one at a time across
separate commands.

Usage:
    python run_all.py --trace t.json --fp32 --out results.json
    python run_all.py --trace t.json --fp32 --engines orca static --max-bs 8
    python run_all.py --tiny   # offline smoke test, no network/weights needed
"""
from __future__ import annotations

import argparse
import time

import engines  # noqa: F401  -- populates driver.ENGINES
from config import Config
from driver import ENGINES, VirtualClock, WallClock, run
from metrics import dump, report
from model_loader import LoadedModel, load_model_and_tokenizer
from trace import Trace

DEFAULT_ENGINES = ["no_batch", "static", "orca", "iter_naive", "iter_padded"]

# (report key, column header, format spec)
SUMMARY_COLS = [
    ("throughput_req_s", "thpt req/s", "{:.3f}"),
    ("latency_p50", "lat p50 s", "{:.2f}"),
    ("latency_p99", "lat p99 s", "{:.2f}"),
    ("queue_delay_p50", "queue p50 s", "{:.2f}"),
    ("hold_after_completion_p50", "held s", "{:.3f}"),
    ("useful_fraction", "useful", "{:.1%}"),
    ("mean_iter_ms", "ms/iter", "{:.1f}"),
]


def make_tiny_trace(lm: LoadedModel, n: int = 60, rate: float = 4.0) -> Trace:
    from trace import generate_poisson_trace, materialize_prompts
    t = generate_poisson_trace(n, rate, seed=0, input_len_range=(8, 24),
                               output_len_range=(2, 12))
    materialize_prompts(t, lm.tokenizer, seed=0)
    return t


def print_summary(reports: list[dict]) -> None:
    header = f"{'engine':<16}" + "".join(f"{h:>13}" for _, h, _ in SUMMARY_COLS)
    rule = "-" * len(header)
    print(f"\n{'='*len(header)}\n{'SUMMARY':^{len(header)}}\n{'='*len(header)}")
    print(header)
    print(rule)
    for rep in reports:
        row = f"{rep['engine']:<16}"
        for key, _, fmt in SUMMARY_COLS:
            row += f"{fmt.format(rep[key]):>13}"
        print(row)
    print(rule)
    print("(all engines: same model, same dtype, same trace, same max_bs -- "
          "the only variable is the engine)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", default=None, help="required unless --tiny")
    p.add_argument("--tiny", action="store_true",
                   help="offline random-weight Qwen2 + synthetic trace, no network needed")
    p.add_argument("--model", default=Config.model)
    p.add_argument("--engines", nargs="+", default=DEFAULT_ENGINES, choices=sorted(ENGINES),
                   help=f"default: {DEFAULT_ENGINES}")
    p.add_argument("--max-bs", type=int, default=Config.max_bs)
    p.add_argument("--max-batched-tokens", type=int, default=Config.max_batched_tokens)
    p.add_argument("--n-slots", type=int, default=0, help="0 = trace.total_slots")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--clock", choices=["wall", "virtual"], default="wall")
    p.add_argument("--check-fcfs", action="store_true")
    p.add_argument("--progress-every", type=int, default=1000)
    p.add_argument("--out", default="results.json")
    a = p.parse_args()

    if not a.tiny and not a.trace:
        p.error("--trace is required unless --tiny is set")

    if a.tiny:
        from test_selective_tiny import make_tiny_lm
        lm = make_tiny_lm()
        trace = make_tiny_trace(lm)
        model_label = "tiny-random-Qwen2"
    else:
        lm = load_model_and_tokenizer(a.model, force_fp32=a.fp32)
        trace = Trace.load(a.trace)
        model_label = a.model

    n_slots = a.n_slots or trace.total_slots
    if a.tiny:
        # static's default slot_policy="max_seq_len" charges a fixed 2048
        # slots/request regardless of actual length (§6.1) -- a real trace's
        # total_slots clears that easily, but the tiny smoke trace's short
        # requests don't. Not a bug in the engine; just not what this
        # smoke test is trying to exercise.
        n_slots = max(n_slots, a.max_bs * 2048)
    print(f"trace: {len(trace.requests)} requests, arrival_rate={trace.arrival_rate}, "
          f"total_slots={trace.total_slots}, n_slots(used)={n_slots}")
    print(f"model: {model_label}  device={lm.device}  dtype={lm.dtype}")
    print(f"engines: {a.engines}\n")

    reports = []
    for name in a.engines:
        print(f"\n{'#'*70}\n# {name}\n{'#'*70}")
        engine = ENGINES[name](lm, max_bs=a.max_bs, n_slots=n_slots,
                               max_batched_tokens=a.max_batched_tokens)
        clock = WallClock() if a.clock == "wall" else VirtualClock()

        t0 = time.perf_counter()
        mc = run(engine, trace, clock, check_fcfs=a.check_fcfs,
                progress_every=a.progress_every)
        wall_s = time.perf_counter() - t0

        rep = report(mc, label=f"{name} max_bs={a.max_bs}", verbose=True)
        rep["engine"] = name
        rep["wall_clock_run_s"] = wall_s
        rep["config"] = Config(
            model=model_label, dtype="fp32" if a.fp32 else "auto", engine=name,
            max_bs=a.max_bs, max_batched_tokens=a.max_batched_tokens, n_slots=n_slots,
            trace_path=a.trace or "(--tiny)", clock=a.clock, check_fcfs=a.check_fcfs,
        ).to_dict()
        reports.append(rep)

    print_summary(reports)
    dump(reports, a.out)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
