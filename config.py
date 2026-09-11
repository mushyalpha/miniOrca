from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    model: str = "Qwen/Qwen2.5-0.5B"
    dtype: str = "auto"         
    engine: str = "orca"


    max_bs: int = 8
    max_batched_tokens: int = 0        
    n_slots: int = 1 << 20           
    reserve: bool = True                
    fcfs_break: bool = True            
    slot_policy: str = "max_seq_len"    

    trace_path: str = ""
    clock: str = "wall"                
    speedup: float = 1.0
    check_fcfs: bool = False
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
