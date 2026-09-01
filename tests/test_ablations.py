from __future__ import annotations

import _path   # noqa: F401 -- sys.path bootstrap for direct python3 invocation
import engines  # noqa: F401
from driver import ENGINES, VirtualClock, run
from metrics import report
from scheduler import KVDeadlock
from test_selective_tiny import make_tiny_lm
from trace import Trace, TraceRequest, materialize_prompts


def make_hitchhiker_trace(lm):
    reqs = [
        TraceRequest(rid=0, n_input_tokens=8, max_gen_tokens=20, arrival_time=0.0),
        TraceRequest(rid=1, n_input_tokens=15, max_gen_tokens=20, arrival_time=0.0),
        TraceRequest(rid=2, n_input_tokens=6, max_gen_tokens=5, arrival_time=1e-4),
    ]
    trace = Trace(requests=reqs, arrival_rate=1.0, seed=0)
    materialize_prompts(trace, lm.tokenizer, seed=0)
    return trace


def check_engine(lm, engine_name: str, ids_oracle: dict) -> None:
    print(f"=== {engine_name} vs no_batch oracle (mixed-phase hitchhiker trace) ===")
    engine = ENGINES[engine_name](lm, max_bs=8)
    mc = run(engine, make_hitchhiker_trace(lm), clock=VirtualClock(), check_fcfs=True)
    ids = {r.rid: r.generated_ids for r in mc.returned}

    all_match = True
    for rid in sorted(ids_oracle):
        match = ids_oracle[rid] == ids[rid]
        all_match &= match
        print(f"  req {rid}: {'MATCH' if match else 'MISMATCH'}")
    assert all_match, f"{engine_name} diverged from the no_batch oracle"

    rep = report(mc, label=engine_name, verbose=False)
    assert rep["pad_token_slots"] == 0, f"{engine_name}: pad_token_slots should be 0"
    assert rep["finished_token_slots"] == 0, f"{engine_name}: finished_token_slots should be 0"


    mixed = any(it.n_initiation > 0 and it.n_increment > 0 for it in mc.iterations)
    print(f"  zero pad/finished waste: OK.  mixed-phase iteration occurred: {mixed}")
    assert mixed, f"{engine_name}: trace never actually produced a mixed-phase batch"
    print(f"  PASS: {engine_name} matches oracle with zero waste.\n")


def check_deadlock_generalizes(lm, engine_name: str) -> None:
    print(f"=== {engine_name}: §4.2 deadlock generalizes from orca for free ===")

    def make_trace():
        reqs = [TraceRequest(rid=i, n_input_tokens=10, max_gen_tokens=15, arrival_time=0.0)
                for i in range(4)]
        t = Trace(requests=reqs, arrival_rate=1.0, seed=0)
        materialize_prompts(t, lm.tokenizer, seed=0)
        return t

    n_slots = 35
    naive = ENGINES[engine_name](lm, max_bs=4, n_slots=n_slots, reserve=False)
    try:
        run(naive, make_trace(), clock=VirtualClock(), check_fcfs=True)
        raise AssertionError(f"expected KVDeadlock for {engine_name} reserve=False")
    except KVDeadlock as e:
        print(f"  reserve=False: KVDeadlock as expected -> {e}")
    print(f"  PASS: {engine_name} deadlocks under the same conditions as orca, "
          f"with zero engine-specific deadlock-handling code.\n")


if __name__ == "__main__":
    lm = make_tiny_lm()

    oracle = ENGINES["no_batch"](lm)
    mc_oracle = run(oracle, make_hitchhiker_trace(lm), clock=VirtualClock(), check_fcfs=True)
    ids_oracle = {r.rid: r.generated_ids for r in mc_oracle.returned}

    check_engine(lm, "iter_naive", ids_oracle)
    check_engine(lm, "iter_padded", ids_oracle)
    check_deadlock_generalizes(lm, "iter_naive")
    check_deadlock_generalizes(lm, "iter_padded")
