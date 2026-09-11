from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Phase(Enum):
    INITIATION = "initiation"
    INCREMENT = "increment"


class State(Enum):
    WAITING = "waiting"      
    RUNNING = "running"     
    DONE = "done"


@dataclass
class RequestState:
    rid: int
    prompt_ids: list[int]
    max_gen_tokens: int         
    arrival_time: float

    phase: Phase = Phase.INITIATION
    state: State = State.WAITING
    generated_ids: list[int] = field(default_factory=list)

    kv_len: int = 0
    kv_cache: object = None     

    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None
    return_time: Optional[float] = None

    n_iterations: int = 0       

    @property
    def n_input_tokens(self) -> int:
        return len(self.prompt_ids)

    @property
    def max_tokens(self) -> int:
        return self.n_input_tokens + self.max_gen_tokens

    @property
    def n_generated(self) -> int:
        return len(self.generated_ids)

    @property
    def is_finished(self) -> bool:
        return self.n_generated >= self.max_gen_tokens

    @property
    def is_selectable(self) -> bool:
        return self.state != State.RUNNING and not self.is_finished

    def tokens_this_iter(self) -> list[int]:
        if self.phase is Phase.INITIATION:
            return list(self.prompt_ids)
        return [self.generated_ids[-1]]

    @property
    def current_len(self) -> int:
        return len(self.tokens_this_iter())

    @property
    def position_offset(self) -> int:

        return self.kv_len

    def append_token(self, token_id: int, now: float) -> None:

        if self.first_token_time is None:
            self.first_token_time = now
        self.generated_ids.append(token_id)
        self.kv_len += self.current_len if self.phase is Phase.INITIATION else 1
        self.phase = Phase.INCREMENT
        self.n_iterations += 1
        if self.is_finished:
            self.completion_time = now
            self.state = State.DONE

    @property
    def latency(self) -> float:
        assert self.return_time is not None
        return self.return_time - self.arrival_time

    @property
    def normalized_latency(self) -> float:
        return self.latency / max(1, self.n_generated)

    @property
    def queue_delay(self) -> float:
        assert self.first_token_time is not None
        return self.first_token_time - self.arrival_time

    @property
    def hold_after_completion(self) -> float:
        return self.return_time - self.completion_time
