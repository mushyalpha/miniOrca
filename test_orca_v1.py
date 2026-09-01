"""
Runs driver.py's three suggested checks against the tiny random Qwen2 model
(the real-download path is still network-blocked -- see test_selective_tiny.py
for why substituting a tiny model here is still a faithful test of the
scheduling/dataflow logic, independent of actual weight values):

  1. orca matches the no_batch oracle exactly, including a request that
     hitches a ride mid-batch (§6.2) -- this is the scenario static batching
     structurally cannot represent.
  2. pad_tokens == 0 and finished_tokens == 0 for every iteration, by
     construction -- the headline structural difference vs static.
  3. The §4.2 deadlock: same constrained n_slots, reserve=True throttles
     admission (slow but always completes); reserve=False deadlocks.
"""
from __future__ import annotations

import engines  # noqa: F401
from driver import ENGINES, VirtualClock, run
from engine_orca import KVDeadlock
from metrics import report
from test_selective_tiny import make_tiny_lm
from trace import Trace, TraceRequest, materialize_prompts


# ------------------------------------------------------------- check 1 & 2
def check_correctness_and_waste(lm) -> None:
    print("=== check 1+2: orca vs no_batch oracle, mid-batch join, zero waste ===")
    reqs = [
        TraceRequest(rid=0, n_input_tokens=8, max_gen_tokens=20, arrival_time=0.0),
        TraceRequest(rid=1, n_input_tokens=15, max_gen_tokens=20, arrival_time=0.0),
        # arrives after the first iteration or two -- must hitch a ride into
        # the *already-running* batch. Static batching cannot do this at all.
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


# ------------------------------------------------------------------ check 3
def check_deadlock(lm) -> None:
    print("=== check 3: §4.2 deadlock -- reserve=True throttles, reserve=False deadlocks ===")
    # Same construction each time: 4 requests, max_tokens = 10+15 = 25 each,
    # n_slots = 35 -- fits ~1.4 full reservations, but nearly 4x a single
    # increment's worth, exactly the "admits more than one, exhausts under
    # naive growth" band the paper describes.
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
