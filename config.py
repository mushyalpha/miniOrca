"""One frozen config, serialized into every results JSON.

Every knob that used to be scattered across CLI flags and engine-constructor
kwargs lives here instead. Three weeks from now, "did run #47 have
reservation on" should be answerable by grepping a JSON file, not by
remembering which flags you passed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    model: str = "Qwen/Qwen2.5-0.5B"
    dtype: str = "auto"          # "auto" | "fp32"
    engine: str = "orca"

    # Scheduler knobs (static, orca, iter_naive, iter_padded -- unused ones
    # are silently ignored via each engine's **_ignored kwarg)
    max_bs: int = 8
    max_batched_tokens: int = 0        # 0 = unlimited (orca-family only)
    n_slots: int = 1 << 20             # 0 = "use trace.total_slots" (CLI only)
    reserve: bool = True                # orca-family: §4.2 reservation vs naive
    fcfs_break: bool = True             # orca-family: Alg 1 line 25's break vs continue
    slot_policy: str = "max_seq_len"    # static only: FasterTransformer-style fixed prealloc

    # Trace / run
    trace_path: str = ""
    clock: str = "wall"                 # "wall" | "virtual"
    speedup: float = 1.0
    check_fcfs: bool = False
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
