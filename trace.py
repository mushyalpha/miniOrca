"""Workload synthesis, §6: 'we synthesize a trace of client requests
because there is no publicly-available request trace'.

  n_input_tokens  ~ U(32, 512)      (§6.2)
  max_gen_tokens  ~ U(1, 128)       (§6.2)
  arrival times   ~ Poisson process (§6)
  EOS is never emitted; every request runs exactly max_gen_tokens iters.

Generate once, save to JSON, replay into every engine.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from request import RequestState


@dataclass
class TraceRequest:
    rid: int
    n_input_tokens: int
    max_gen_tokens: int
    arrival_time: float
    prompt_ids: Optional[list[int]] = None

    @property
    def max_tokens(self) -> int:
        return self.n_input_tokens + self.max_gen_tokens


@dataclass
class Trace:
    requests: list[TraceRequest]
    arrival_rate: float
    seed: int
    homogeneous: bool = False

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"arrival_rate": self.arrival_rate, "seed": self.seed,
                       "homogeneous": self.homogeneous,
                       "requests": [asdict(r) for r in self.requests]}, f)

    @staticmethod
    def load(path: str) -> "Trace":
        with open(path) as f:
            d = json.load(f)
        return Trace(requests=[TraceRequest(**r) for r in d["requests"]],
                     arrival_rate=d["arrival_rate"], seed=d["seed"],
                     homogeneous=d.get("homogeneous", False))

    def to_request_states(self) -> list[RequestState]:
        assert all(r.prompt_ids is not None for r in self.requests), \
            "call materialize_prompts() first"
        return [RequestState(rid=r.rid, prompt_ids=r.prompt_ids,
                             max_gen_tokens=r.max_gen_tokens,
                             arrival_time=r.arrival_time)
                for r in self.requests]

    @property
    def total_slots(self) -> int:
        """Sum of max_tokens — the n_slots value at which reservation
        never blocks. Set n_slots below this to exercise Alg 1 line 25."""
        return sum(r.max_tokens for r in self.requests)


def generate_poisson_trace(num_requests: int, arrival_rate: float, seed: int = 0,
                           input_len_range: tuple[int, int] = (32, 512),
                           output_len_range: tuple[int, int] = (1, 128),
                           homogeneous: Optional[tuple[int, int]] = None) -> Trace:
    """homogeneous=(in_len, gen_len) reproduces Figure 11's control trace,
    where no request finishes early and static batching stops looking bad."""
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0 / arrival_rate, size=num_requests)
    arrivals = np.cumsum(gaps) - gaps[0]          # first request at t=0
    reqs = []
    for i in range(num_requests):
        if homogeneous is not None:
            n_in, n_out = homogeneous
        else:
            n_in = int(rng.integers(input_len_range[0], input_len_range[1] + 1))
            n_out = int(rng.integers(output_len_range[0], output_len_range[1] + 1))
        reqs.append(TraceRequest(rid=i, n_input_tokens=n_in,
                                 max_gen_tokens=n_out,
                                 arrival_time=float(arrivals[i])))
    return Trace(reqs, arrival_rate, seed, homogeneous is not None)


def materialize_prompts(trace: Trace, tokenizer, mode: str = "random",
                        seed: int = 0, texts: Optional[list[str]] = None) -> Trace:
    """§6 again: 'we have neither the actual model checkpoint nor the actual
    input text'. Random ids hit exact target lengths, which is what makes
    lengths comparable across engines. Use mode='text' for readable output."""
    rng = np.random.default_rng(seed)
    if mode == "random":
        lo, hi = 1000, min(tokenizer.vocab_size, 100_000)
        special = set(tokenizer.all_special_ids)
        for r in trace.requests:
            ids = []
            while len(ids) < r.n_input_tokens:
                cand = int(rng.integers(lo, hi))
                if cand not in special:
                    ids.append(cand)
            r.prompt_ids = ids
    elif mode == "text":
        assert texts, "mode='text' needs a corpus"
        for i, r in enumerate(trace.requests):
            ids = tokenizer(texts[i % len(texts)], add_special_tokens=False)["input_ids"]
            while len(ids) < r.n_input_tokens:
                ids = ids + ids
            r.prompt_ids = ids[:r.n_input_tokens]
    else:
        raise ValueError(mode)
    return trace


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("-n", "--num-requests", type=int, default=200)
    p.add_argument("-r", "--arrival-rate", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--homogeneous", nargs=2, type=int, default=None,
                   metavar=("IN_LEN", "GEN_LEN"))
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    a = p.parse_args()

    from transformers import AutoTokenizer
    t = generate_poisson_trace(a.num_requests, a.arrival_rate, a.seed,
                               homogeneous=tuple(a.homogeneous) if a.homogeneous else None)
    materialize_prompts(t, AutoTokenizer.from_pretrained(a.model), seed=a.seed)
    t.save(a.out)
    print(f"{a.num_requests} reqs, span {t.requests[-1].arrival_time:.1f}s, "
          f"total_slots={t.total_slots}")
