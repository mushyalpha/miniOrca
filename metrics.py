from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Optional

from request import Phase, RequestState


@dataclass
class IterationRecord:
    index: int
    start: float
    end: float
    batch_size: int                 
    useful_tokens: int = 0          
    pad_tokens: int = 0            
    finished_tokens: int = 0        
    n_initiation: int = 0
    n_increment: int = 0
    kv_slots_used: int = 0          

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def total_tokens(self) -> int:
        return self.useful_tokens + self.pad_tokens + self.finished_tokens


@dataclass
class MetricsCollector:
    iterations: list[IterationRecord] = field(default_factory=list)
    returned: list[RequestState] = field(default_factory=list)
    t_start: Optional[float] = None
    t_end: Optional[float] = None

    def record_iteration(self, rec: IterationRecord) -> None:
        self.iterations.append(rec)

    def record_return(self, req: RequestState) -> None:
        self.returned.append(req)


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def report(mc: MetricsCollector, label: str = "", verbose: bool = True) -> dict:
    reqs = [r for r in mc.returned if r.return_time is not None]
    assert reqs, "no returned requests"

    lat = [r.latency for r in reqs]
    nlat = [r.normalized_latency for r in reqs]       
    queue = [r.queue_delay for r in reqs]
    hold = [r.hold_after_completion for r in reqs]

    first_arrival = min(r.arrival_time for r in reqs)
    last_return = max(r.return_time for r in reqs)
    window = max(1e-9, last_return - first_arrival)

    useful = sum(i.useful_tokens for i in mc.iterations)
    pad = sum(i.pad_tokens for i in mc.iterations)
    fin = sum(i.finished_tokens for i in mc.iterations)
    total = useful + pad + fin

    gen = sum(r.n_generated for r in reqs)
    offered = len(reqs) / max(1e-9, (max(r.arrival_time for r in reqs) - first_arrival))

    out = {
        "label": label,
        "n_requests": len(reqs),
        "window_s": window,
        "throughput_req_s": len(reqs) / window,
        "offered_rate_req_s": offered,
        "gen_token_throughput": gen / window,
        # Figure 10 reports *median normalized* latency; keep both.
        "latency_p50": median(lat), "latency_p90": _pct(lat, .9), "latency_p99": _pct(lat, .99),
        "norm_latency_p50": median(nlat), "norm_latency_p90": _pct(nlat, .9),
        "queue_delay_p50": median(queue), "queue_delay_p99": _pct(queue, .99),
        "hold_after_completion_p50": median(hold),   # ~0 iff early return works
        "n_iterations": len(mc.iterations),
        "mean_batch_size": sum(i.batch_size for i in mc.iterations) / len(mc.iterations),
        "mean_tokens_per_iter": total / len(mc.iterations),
        "mean_iter_ms": 1e3 * sum(i.duration for i in mc.iterations) / len(mc.iterations),
        "p99_iter_ms": 1e3 * _pct([i.duration for i in mc.iterations], .99),
        # the waste story
        "useful_token_slots": useful,
        "pad_token_slots": pad,
        "finished_token_slots": fin,
        "useful_fraction": useful / max(1, total),
        "peak_kv_slots": max((i.kv_slots_used for i in mc.iterations), default=0),
    }
    if verbose:
        _print(out)
    return out


def _print(d: dict) -> None:
    print(f"\n=== {d['label']} ===")
    print(f"  throughput      {d['throughput_req_s']:.3f} req/s "
          f"(offered {d['offered_rate_req_s']:.3f})")
    print(f"  latency p50/p90/p99   {d['latency_p50']:.2f} / "
          f"{d['latency_p90']:.2f} / {d['latency_p99']:.2f} s")
    print(f"  norm latency p50      {1e3*d['norm_latency_p50']:.1f} ms/token")
    print(f"  queue delay p50/p99   {d['queue_delay_p50']:.2f} / {d['queue_delay_p99']:.2f} s")
    print(f"  held after finish p50 {d['hold_after_completion_p50']:.2f} s")
    print(f"  iters {d['n_iterations']}  mean bs {d['mean_batch_size']:.2f}  "
          f"mean {d['mean_tokens_per_iter']:.1f} tok/iter  "
          f"{d['mean_iter_ms']:.1f} ms/iter")
    print(f"  token slots  useful {d['useful_token_slots']} | "
          f"pad {d['pad_token_slots']} | finished {d['finished_token_slots']} "
          f"=> {100*d['useful_fraction']:.1f}% useful")


def check_iteration_fcfs(reqs: list[RequestState]) -> None:
    active = [r for r in reqs if not r.is_finished]
    s = sorted(active, key=lambda r: r.arrival_time)
    for a, b in zip(s, s[1:]):
        assert a.n_iterations >= b.n_iterations, (
            f"FCFS violated: r{a.rid} (t={a.arrival_time:.2f}, "
            f"{a.n_iterations} iters) < r{b.rid} (t={b.arrival_time:.2f}, "
            f"{b.n_iterations} iters)")


def dump(reports: list[dict], path: str) -> None:
    with open(path, "w") as f:
        json.dump(reports, f, indent=2)
