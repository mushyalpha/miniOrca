"""Shared outer loop: arrivals, clock, metrics. Engines differ only in
step(). Without this, every engine reimplements 'has this arrived yet'
and you get the same drift you avoided in the model forward."""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from metrics import IterationRecord, MetricsCollector, check_iteration_fcfs, report
from request import RequestState, State
from trace import Trace

# Every engine module does `from driver import ENGINES`. If this file is run
# directly (`python3 driver.py ...`), it executes as `__main__`, which is a
# *different* module object than "driver" -- so those imports would trigger
# a second, fresh execution of this whole file, with its own separate
# ENGINES dict (and its own separate IterationRecord/RequestState-adjacent
# classes) that main()'s own globals never see populated. Registering this
# module under its real name up front makes "import driver" and running
# this file directly resolve to the *same* module object either way.
sys.modules.setdefault("driver", sys.modules[__name__])


# ---------------------------------------------------------------- clocks
class Clock(Protocol):
    def now(self) -> float: ...
    def advance_to(self, t: float) -> None: ...
    def observe_iteration(self, wall_seconds: float) -> None: ...


@dataclass
class WallClock:
    """Honest: real elapsed time. `speedup > 1` compresses arrivals,
    i.e. raises offered load without regenerating the trace."""
    speedup: float = 1.0
    _t0: float = field(default_factory=time.perf_counter)

    def now(self) -> float:
        return (time.perf_counter() - self._t0) * self.speedup

    def advance_to(self, t: float) -> None:
        gap = (t - self.now()) / self.speedup
        if gap > 0:
            time.sleep(gap)

    def observe_iteration(self, wall_seconds: float) -> None:
        pass


@dataclass
class VirtualClock:
    """Reproducible: trace time advances only by measured model time, so
    idle gaps cost nothing and results don't depend on how long you sat
    in Python. Throughput is still real (it's real GPU seconds), but the
    system never idles, so treat it as a saturated-load measurement."""
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance_to(self, t: float) -> None:
        self.t = max(self.t, t)

    def observe_iteration(self, wall_seconds: float) -> None:
        self.t += wall_seconds


# --------------------------------------------------------------- engines
@dataclass
class IterationResult:
    record: IterationRecord
    returned: list[RequestState] = field(default_factory=list)


@runtime_checkable
class Engine(Protocol):
    """Every engine owns `pool` (Figure 4's request pool). The driver only
    inserts arrivals — Algorithm 1's 'concurrent threads inserting newly
    arrived requests'. The engine removes finished requests and reports
    which ones it is returning to the client *now*."""
    pool: list[RequestState]

    def has_work(self) -> bool: ...
    def step(self, now: float, iter_index: int) -> IterationResult: ...


# ---------------------------------------------------------------- driver
def run(engine: Engine, trace: Trace, clock: Optional[Clock] = None,
        max_iterations: int = 1_000_000, check_fcfs: bool = False,
        progress_every: int = 0) -> MetricsCollector:
    clock = clock or WallClock()
    pending = sorted(trace.to_request_states(), key=lambda r: r.arrival_time)
    mc = MetricsCollector(t_start=clock.now())
    i = 0

    while pending or engine.has_work():
        now = clock.now()

        # 1. admit arrivals (§3 S1: a new request gets its chance after the
        #    currently scheduled iteration, not after the current batch)
        while pending and pending[0].arrival_time <= now:
            engine.pool.append(pending.pop(0))

        # 2. nothing runnable -> skip to the next arrival
        if not engine.has_work():
            if not pending:
                break
            clock.advance_to(pending[0].arrival_time)
            continue

        # 3. one iteration
        t0 = time.perf_counter()
        result = engine.step(now, i)
        wall = time.perf_counter() - t0
        clock.observe_iteration(wall)
        result.record.start, result.record.end = now, clock.now()
        mc.record_iteration(result.record)

        # 4. returns. The engine decides *when* a finished request is
        #    released; that gap is C1's latency penalty for static.
        for req in result.returned:
            req.return_time = clock.now()
            req.state = State.DONE
            mc.record_return(req)

        if check_fcfs:
            check_iteration_fcfs([r for r in engine.pool if r.n_iterations > 0])

        i += 1
        if progress_every and i % progress_every == 0:
            print(f"  iter {i:6d}  pool={len(engine.pool):3d}  "
                  f"pending={len(pending):3d}  done={len(mc.returned):4d}")
        if i >= max_iterations:
            raise RuntimeError(f"hit max_iterations={max_iterations}; "
                               f"{len(engine.pool)} stuck in pool "
                               f"(deadlock? check n_slots)")

    mc.t_end = clock.now()
    assert not engine.pool, f"{len(engine.pool)} requests never returned"
    return mc


# ------------------------------------------------------------------ CLI
ENGINES = {}   # name -> factory(loaded_model, **kwargs); registered by each engine module


def main() -> None:
    import engines  # noqa: F401  -- populates ENGINES; must happen before
                     # the parser is built, or --engine choices is always None.
    from config import Config

    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True)
    p.add_argument("--engine", required=True, choices=sorted(ENGINES) or None)
    p.add_argument("--model", default=Config.model)
    p.add_argument("--max-bs", type=int, default=Config.max_bs)
    p.add_argument("--max-batched-tokens", type=int, default=Config.max_batched_tokens)
    p.add_argument("--n-slots", type=int, default=0,
                   help="0 = use trace.total_slots (never blocks on reservation)")
    p.add_argument("--slot-policy", choices=["max_seq_len", "max_tokens"],
                   default=Config.slot_policy, help="static engine only")
    p.add_argument("--reserve", dest="reserve", action="store_true")
    p.add_argument("--no-reserve", dest="reserve", action="store_false")
    p.set_defaults(reserve=Config.reserve)
    p.add_argument("--fcfs-break", dest="fcfs_break", action="store_true")
    p.add_argument("--no-fcfs-break", dest="fcfs_break", action="store_false")
    p.set_defaults(fcfs_break=Config.fcfs_break)
    p.add_argument("--clock", choices=["wall", "virtual"], default=Config.clock)
    p.add_argument("--speedup", type=float, default=Config.speedup)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--check-fcfs", action="store_true", default=Config.check_fcfs)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    cfg = Config(model=a.model, dtype="fp32" if a.fp32 else "auto", engine=a.engine,
                max_bs=a.max_bs, max_batched_tokens=a.max_batched_tokens,
                n_slots=a.n_slots, reserve=a.reserve, fcfs_break=a.fcfs_break,
                slot_policy=a.slot_policy, trace_path=a.trace, clock=a.clock,
                speedup=a.speedup, check_fcfs=a.check_fcfs, seed=a.seed)

    from model_loader import load_model_and_tokenizer
    lm = load_model_and_tokenizer(cfg.model, force_fp32=(cfg.dtype == "fp32"))
    trace = Trace.load(cfg.trace_path)
    n_slots = cfg.n_slots or trace.total_slots
    engine = ENGINES[cfg.engine](lm, max_bs=cfg.max_bs, n_slots=n_slots,
                                 max_batched_tokens=cfg.max_batched_tokens,
                                 reserve=cfg.reserve, fcfs_break=cfg.fcfs_break,
                                 slot_policy=cfg.slot_policy)

    clock = WallClock(cfg.speedup) if cfg.clock == "wall" else VirtualClock()
    mc = run(engine, trace, clock, check_fcfs=cfg.check_fcfs, progress_every=200)
    rep = report(mc, label=f"{cfg.engine} max_bs={cfg.max_bs} rate={trace.arrival_rate}")
    rep["config"] = cfg.to_dict()
    if a.out:
        from metrics import dump
        dump([rep], a.out)


if __name__ == "__main__":
    main()
