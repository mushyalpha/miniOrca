from __future__ import annotations

import _path   
import engines
from driver import ENGINES, VirtualClock, run
from engines.engine_orca import KVDeadlock
from metrics import report
from test_selective_tiny import make_tiny_lm
from trace import Trace, TraceRequest, materialize_prompts


def check_correctness_and_waste(lm) -> None:
    print("=== check 1+2: orca vs no_batch oracle, mid-batch join, zero waste ===")
    reqs = [
        TraceRequest(rid=0, n_input_tokens=8, max_gen_tokens=20, arrival_time=0.0),
        TraceRequest(rid=1, n_input_tokens=15, max_gen_tokens=20, arrival_time=0.0),

        TraceRequest(rid=2, n_input_tokens=6, max_gen_tokens=5, arrival_time=1e-4),
    ]
    trace = Trace(requests=reqs, arrival_rate=1.0, seed=0)
    materialize_prompts(trace, lm.tokenizer, seed=0)

    oracle = ENGINES["no_batch"](lm)
    mc_oracle = run(oracle, trace, clock=VirtualClock(), check_fcfs=True)
    ids_oracle = {r.rid: r.generated_ids for r in mc_oracle.returned}

    orca = ENGINES["orca"](lm, max_bs=8)
    mc_orca = run(orca, trace, clock=VirtualClock(), check_fcfs=True)
    ids_orca = {r.rid: r.generated_ids for r in mc_orca.returned}

    all_match = True
    for rid in sorted(ids_oracle):
        match = ids_oracle[rid] == ids_orca[rid]
        all_match &= match
        print(f"  req {rid}: {'MATCH' if match else 'MISMATCH'}  "
              f"oracle={ids_oracle[rid]}  orca={ids_orca[rid]}")
    assert all_match, "orca diverged from the no_batch oracle"

    rep = report(mc_orca, label="orca (tiny model)", verbose=False)
    print(f"\n  pad_token_slots      = {rep['pad_token_slots']} (must be 0)")
    print(f"  finished_token_slots = {rep['finished_token_slots']} (must be 0)")
    print(f"  useful_fraction      = {100*rep['useful_fraction']:.1f}% (should be 100%)")
    assert rep["pad_token_slots"] == 0
    assert rep["finished_token_slots"] == 0
    assert abs(rep["useful_fraction"] - 1.0) < 1e-9
    print("  PASS: exact match vs oracle, zero pad/finished waste by construction.\n")

def check_deadlock(lm) -> None:
    print("=== check 3: §4.2 deadlock -- reserve=True throttles, reserve=False deadlocks ===")

    def make_trace():
        reqs = [TraceRequest(rid=i, n_input_tokens=10, max_gen_tokens=15, arrival_time=0.0)
                for i in range(4)]
        t = Trace(requests=reqs, arrival_rate=1.0, seed=0)
        materialize_prompts(t, lm.tokenizer, seed=0)
        return t

    n_slots = 35

    reserved = ENGINES["orca"](lm, max_bs=4, n_slots=n_slots, reserve=True)
    mc = run(reserved, make_trace(), clock=VirtualClock(), check_fcfs=True)
    rep = report(mc, label=f"orca reserve=True, n_slots={n_slots}", verbose=False)
    print(f"  reserve=True : completed cleanly, {rep['n_iterations']} iterations, "
          f"peak_kv_slots={rep['peak_kv_slots']} (<= {n_slots})")
    assert rep["peak_kv_slots"] <= n_slots

    naive = ENGINES["orca_no_reserve"](lm, max_bs=4, n_slots=n_slots)
    try:
        run(naive, make_trace(), clock=VirtualClock(), check_fcfs=True)
        raise AssertionError("expected KVDeadlock with reserve=False at this n_slots, "
                             "but the run completed -- band is wrong, tighten n_slots")
    except KVDeadlock as e:
        print(f"  reserve=False: KVDeadlock as expected -> {e}")

    print("\n  PASS: identical scheduler, identical n_slots -- reservation is the only "
          "difference between 'throttles' and 'deadlocks'.\n")


if __name__ == "__main__":
    lm = make_tiny_lm()
    check_correctness_and_waste(lm)
    check_deadlock(lm)
