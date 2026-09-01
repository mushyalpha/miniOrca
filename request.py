"""Request state — §2 (phases), §4.2 / Algorithm 1 (state, max_tokens)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Phase(Enum):
    """§2. INITIATION = one-shot forward over all prompt tokens.
    INCREMENT = one token per iteration, using cached K/V."""
    INITIATION = "initiation"
    INCREMENT = "increment"


class State(Enum):
    """Algorithm 1's req.state. The paper overloads one field with
    {INITIATION, INCREMENT, RUNNING}; we split phase/state and treat
    `state != RUNNING` (Alg 1 line 19) as `is_selectable`."""
    WAITING = "waiting"      # in pool, not currently in flight
    RUNNING = "running"      # iteration in flight (matters once pipelined)
    DONE = "done"


@dataclass
class RequestState:
    rid: int
    prompt_ids: list[int]
    max_gen_tokens: int          # §6: forced; the model never emits EOS
    arrival_time: float

    phase: Phase = Phase.INITIATION
    state: State = State.WAITING
    generated_ids: list[int] = field(default_factory=list)

    # Attention K/V manager bookkeeping (§3, §4.2). kv_len is how many
    # tokens are actually materialized; max_tokens is what we *reserved*.
    kv_len: int = 0
    kv_cache: object = None      # engine-defined; per-request, never global

    # Timestamps. completion_time != return_time is the whole of C1:
    # static batching generates the last token, then holds the response
    # until every row in the batch drains.
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None
    return_time: Optional[float] = None

    n_iterations: int = 0        # for the iteration-level FCFS check (§4.2)

    @property
    def n_input_tokens(self) -> int:
        return len(self.prompt_ids)

    @property
    def max_tokens(self) -> int:
        """§4.2 fn.6: n_input_tokens + max_gen_tokens. The number of K/V
        slots reserved at admission, so growth can never fail mid-flight."""
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
        """What the engine must feed this request in the next iteration.
        Prompt in bulk for INITIATION, last token only for INCREMENT."""
        if self.phase is Phase.INITIATION:
            return list(self.prompt_ids)
        return [self.generated_ids[-1]]

    @property
    def current_len(self) -> int:
        return len(self.tokens_this_iter())

    @property
    def position_offset(self) -> int:
        """RoPE start position for this iteration's tokens."""
        return self.kv_len

    def append_token(self, token_id: int, now: float) -> None:
        """EOS is deliberately ignored (§6): forcing max_gen_tokens keeps
        token counts deterministic across engines and traces replayable."""
        if self.first_token_time is None:
            self.first_token_time = now
        self.generated_ids.append(token_id)
        self.kv_len += self.current_len if self.phase is Phase.INITIATION else 1
        self.phase = Phase.INCREMENT
        self.n_iterations += 1
        if self.is_finished:
            self.completion_time = now
            self.state = State.DONE

    # --- convenience -------------------------------------------------
    @property
    def latency(self) -> float:
        assert self.return_time is not None
        return self.return_time - self.arrival_time

    @property
    def normalized_latency(self) -> float:
        """Figure 10's y-axis: latency / number of generated tokens."""
        return self.latency / max(1, self.n_generated)

    @property
    def queue_delay(self) -> float:
        assert self.first_token_time is not None
        return self.first_token_time - self.arrival_time

    @property
    def hold_after_completion(self) -> float:
        """Time spent finished-but-not-returned. ~0 for Orca, large for static."""
        return self.return_time - self.completion_time
