"""Algorithm 1 (§4.2): re-decides batch composition from scratch every
iteration. Lives here exactly once -- orca, iter_naive, and iter_padded
differ *only* in which Runner they hand the chosen batch to, never in how
the batch is chosen. Owns admission policy and the K/V slot budget;
storage is entirely the Runner's problem (`runner.alloc`/`runner.free`).
"""
from __future__ import annotations

from driver import IterationResult
from kv_manager import KVOutOfMemory, SlotAllocator
from metrics import IterationRecord
from request import Phase, RequestState, State
from runner import Runner


class KVDeadlock(RuntimeError):
    pass


class IterationLevelScheduler:
    def __init__(self, runner: Runner, max_bs: int = 8, n_slots: int = 1 << 20,
                 max_batched_tokens: int = 0, reserve: bool = True,
                 fcfs_break: bool = True, **_ignored):
        self.runner = runner
        self.max_bs = max_bs
        self.alloc = SlotAllocator(n_slots)
        # 0 = unlimited. NOT in the paper; Orca bounds a batch by max_bs only.
        # But an 8-request batch is 8 flat tokens when decoding and ~4000 when
        # eight prompts prefill together, a 500x swing in iteration time that
        # wrecks p99. Leave at 0 for a faithful run; set it to study the knob.
        self.max_batched_tokens = max_batched_tokens
        # reserve=False is the "naive implementation" of §4.2 that deadlocks.
        self.reserve = reserve
        # Algorithm 1 uses `break` (line 25), preserving iteration-level FCFS.
        # `continue` would raise utilisation and allow starvation.
        self.fcfs_break = fcfs_break

        self.pool: list[RequestState] = []
        self._reserved: dict[int, int] = {}   # rid -> slots charged, scheduler's own bookkeeping

    # ------------------------------------------------------------ Engine API
    def has_work(self) -> bool:
        return bool(self.pool)

    def step(self, now: float, iter_index: int) -> IterationResult:
        batch = self.select()

        if not batch:
            self._diagnose_empty_batch()

        for r in batch:
            r.state = State.RUNNING                       # Alg 1 lines 6-7

        logits = self.runner.run(batch)

        n_init = sum(1 for r in batch if r.phase is Phase.INITIATION)
        n_tokens = sum(r.current_len for r in batch)
        greedy = logits.argmax(dim=-1)

        returned: list[RequestState] = []
        for row, r in enumerate(batch):
            r.append_token(int(greedy[row]), now)         # flips phase to INCREMENT
            r.state = State.DONE if r.is_finished else State.WAITING
            if r.is_finished:
                # §3 S1: "it can detect the completion of a request and
                # immediately return its generated tokens to the client."
                self._release(r)
                returned.append(r)

        rec = IterationRecord(
            index=iter_index, start=now, end=now, batch_size=len(batch),
            useful_tokens=n_tokens,      # all of them; nothing padded or dead
            pad_tokens=0, finished_tokens=0,
            n_initiation=n_init, n_increment=len(batch) - n_init,
            kv_slots_used=self.alloc.used,
        )
        return IterationResult(record=rec, returned=returned)

    # --------------------------------------------------- Algorithm 1: Select
    def select(self) -> list[RequestState]:
        """Selects at most max_bs requests by arrival time, reserving K/V
        space (via the Runner) for any request scheduled for the first
        time."""
        batch: list[RequestState] = []
        n_tokens = 0
        candidates = sorted((r for r in self.pool if r.is_selectable),
                            key=lambda r: (r.arrival_time, r.rid))

        for req in candidates:
            if len(batch) == self.max_bs:
                break

            if (self.max_batched_tokens and batch
                    and n_tokens + req.current_len > self.max_batched_tokens):
                break

            if req.phase is Phase.INITIATION:
                need = req.max_tokens if self.reserve else req.current_len
                if not self._charge(req, need):
                    if self.fcfs_break:
                        break
                    continue
                self.runner.alloc(req, capacity=req.max_tokens if self.reserve else None)
                self._reserved[req.rid] = need
            elif not self.reserve:
                # The naive path: an in-flight request must find room for the
                # next token's K/V *now*. This is what §4.2 warns about.
                if not self._charge(req, 1):
                    if self.fcfs_break:
                        break
                    continue
                self._reserved[req.rid] += 1

            batch.append(req)
            n_tokens += req.current_len
        return batch

    def _charge(self, req: RequestState, n: int) -> bool:
        try:
            self.alloc.alloc(n)
            return True
        except KVOutOfMemory:
            return False

    def _release(self, req: RequestState) -> None:
        """Alg 1 lines 14-15: n_rsrv -= req.max_tokens on finish."""
        self.alloc.release(self._reserved.pop(req.rid))
        self.runner.free(req)
        self.pool.remove(req)

    def _diagnose_empty_batch(self) -> None:
        in_flight = [r for r in self.pool if r.phase is Phase.INCREMENT]
        if in_flight:
            raise KVDeadlock(
                f"no request can be scheduled: {len(in_flight)} requests are "
                f"mid-generation but all {self.alloc.n_slots} K/V slots are "
                f"held ({self.alloc.used} used). This is §4.2's deadlock; it "
                f"cannot happen with reserve=True.")
        head = min(self.pool, key=lambda r: r.arrival_time)
        raise RuntimeError(
            f"n_slots={self.alloc.n_slots} cannot fit even the head-of-line "
            f"request (needs {head.max_tokens}). Raise n_slots.")
