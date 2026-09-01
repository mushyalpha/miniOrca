# Mini-Orca

A from-scratch, faithful-comparison implementation of LLM inference
scheduling strategies, built to understand the core claims of
**[ORCA: A Distributed Serving System for Transformer-Based Generative
Models](https://www.usenix.org/conference/osdi22/presentation/yu)** (OSDI '22).

Every engine is a composition of two independent pieces:

- a **scheduler**, which decides *which requests run this iteration* and
  owns the K/V memory budget (`scheduler.py`'s `IterationLevelScheduler` is
  Algorithm 1's `Select()`, implemented exactly once);
- a **runner**, which decides *how a chosen batch turns into logits*
  (`runner.py`: `SingleRunner`, `FlatRunner`, `PaddedRunner`).

Six engines fall out of that split, registered in `driver.ENGINES`:

| Engine | Scheduler | Runner | What it demonstrates |
|---|---|---|---|
| `no_batch` | request-level, bs=1 | single request | Correctness oracle every other engine is checked against. |
| `static` | request-level, closed batch | dense `[B,L,H]` | FasterTransformer baseline (§3, Fig. 3): batch closed at admission, late arrivals wait for the whole batch, finished rows keep computing until the slowest row drains. |
| `orca` | **iteration-level** (Alg. 1) | **selective batching** (§3 S2) | Orca's actual contribution. Finished requests return immediately; new requests join an in-flight batch mid-iteration; nothing is ever padded or recomputed after finishing. |
| `iter_naive` | iteration-level (Alg. 1) | single request, looped | C1 without C2: same admission/scheduling wins as `orca`, but no shared compute — isolates what selective batching's *batching* actually buys on top of iteration-level scheduling alone. |
| `iter_padded` | iteration-level (Alg. 1) | dense `[B,L,H]`, K/V reassembled every iteration | C1 with *naive* batching: the "reassemble into a padded tensor" pattern that predates selective batching. Gets iteration-level scheduling's wins, pays a real per-iteration gather/scatter cost and mixed-phase padding cost that `orca` doesn't. |
| `orca_no_reserve` / `orca_greedy_select` | iteration-level, ablated | selective batching | Pre-wired ablations: naive (non-reserving) K/V accounting (§4.2's deadlock), and `continue`-instead-of-`break` in `Select()` (breaks iteration-level FCFS). |

Because `orca`, `iter_naive`, and `iter_padded` share the *identical*
`IterationLevelScheduler` instance type, any behavioral difference between
them (including the §4.2 deadlock, see below) is attributable entirely to
the runner — there is no scheduling-policy drift to control for.

The point of this repo isn't a fast inference engine — it's an apples-to-
apples testbed where every engine shares the same model, the same tokenizer
config, the same request trace, and the same greedy decoding, so that
differences in throughput/latency/waste are attributable to the
**scheduler and runner choice**, not to incidental implementation drift.

## Take Notice

Every non-trivial piece of logic here is checked against a **correctness
oracle**, not just benchmarked:

- `engine_static`'s padded, position-corrected forward pass is checked
  token-for-token against `engine_no_batch` (`test_pipeline.py`,
  `test_static_v2.py`).
- `selective_batching.py`'s flatten → per-token linear ops → split →
  per-request attention → merge dataflow is checked against HF's own dense
  `model(...)` forward, logit-for-logit (`test_selective_tiny.py`,
  `selective_batching.validate()`).
- `IterationLevelScheduler`'s `Select()` (used by `orca`) is checked
  token-for-token against the oracle on a trace where a request arrives
  mid-batch and must "hitch a ride" — the exact scenario static batching
  cannot represent (`test_orca_v1.py`).
- `iter_naive` and `iter_padded` are checked the *same* way, on the *same*
  mixed-phase trace, against the *same* oracle (`test_ablations.py`) —
  confirming that swapping the runner underneath an identical scheduler
  changes nothing about what gets generated.
- `PaddedRunner`'s gather-into-padded-batch / scatter-back-out mechanism was
  validated standalone before being wired in at all: an explicit 4D causal
  mask, checked logit-for-logit against each request run alone through HF's
  own forward, specifically with *different* past-lengths and *different*
  new-token-counts in the same batch (the case a scalar "one past length for
  the whole batch" assumption — which HF's automatic 2D→4D mask expansion
  makes — would get wrong silently, not loudly).
- The §4.2 K/V reservation claim is checked as a *correctness* property, not
  a benchmark: with identical `n_slots`, `reserve=True` always completes
  (just throttles admission) and `reserve=False` provably deadlocks
  (`test_orca_v1.py`, `test_ablations.py`, raises `scheduler.KVDeadlock`).
  It reproduces identically for `orca`, `iter_naive`, and `iter_padded` with
  zero engine-specific deadlock-handling code, because all three share one
  scheduler.
- A standing regression guard (`test_mixed_phase.py`) asserts that every
  iteration-level engine actually produces at least one batch mixing an
  INITIATION request with INCREMENT requests (§3 C2 case 3) during a normal
  run — the case a closed static batch can never represent, and the case a
  nano-vLLM-style "schedule prefills XOR decodes" scheduler would silently
  give up.

**Known gap:** several of these tests currently run against a tiny,
randomly-initialized Qwen2 model (same `modeling_qwen2` code path as
Qwen2.5-0.5B, just tiny dimensions) instead of real Qwen2.5-0.5B weights,
because the weight download was network-blocked in the environment this was
built in (`huggingface_hub`'s `hf_xet` transfer client hung indefinitely on
the actual `.safetensors` blob; plain `curl` to the same redirect URL
worked fine). This substitution is valid for everything **except** absolute
latency magnitudes: the tensor-plumbing and scheduling correctness checks
above are properties of shapes/positions/masks/admission logic, not of
weight values, so a tiny model exercises the identical code path. It is
*not* valid for claims like "held-after-completion is O(seconds)" — that
needs real per-iteration latency. `test_static_real.py` is written for
exactly that check; run it once you have working network access to
`huggingface.co`.

## Files

### Shared infrastructure (every engine depends on these; keeping them
identical across engines is what makes the comparison fair)

| File | What it is |
|---|---|
| `request.py` | `RequestState`: the lifecycle object for one request (prompt, generated tokens, KV bookkeeping, phase, timestamps). Defines `Phase` (INITIATION/INCREMENT) and `State` (WAITING/RUNNING/DONE). |
| `config.py` | `Config`: one frozen dataclass holding every knob (model, engine, `max_bs`, `n_slots`, `reserve`, `fcfs_break`, `slot_policy`, trace path, clock, seed...). `driver.py`'s CLI builds one from argv and embeds it (`rep["config"]`) in every `--out` report, so "what were run #47's settings" is answerable by reading the JSON, not by reconstructing the command line. |
| `model_loader.py` | One loader (`load_model_and_tokenizer`) used by every engine, so dtype/device/attention-impl never silently drifts between engines. Also caches model config (n_heads, head_dim, GQA repeat factor, etc.) needed by `selective_batching.py` and `runner.py`. |
| `trace.py` | Synthetic workload generation (§6: Poisson arrivals, `n_input_tokens ~ U(32,512)`, `max_gen_tokens ~ U(1,128)`), plus `materialize_prompts()` to turn lengths into concrete random token ids. Traces save/load as JSON for exact replay across engines. |
| `metrics.py` | `IterationRecord` (one forward pass, whatever the engine) and `MetricsCollector` → `report()`: throughput, latency percentiles, queue delay, **hold-after-completion** (the C1 headline metric), and the waste breakdown (`useful` / `pad` / `finished` token-slots). Also `check_iteration_fcfs()`, a scheduler invariant check. |
| `kv_manager.py` | `SlotAllocator` (a budget of K/V "slots" that can refuse allocations — this is what makes the §4.2 deadlock demo *real*, not simulated) and `RequestKV` (per-request K/V buffers; either preallocated to `max_tokens` (reservation) or grown by `torch.cat` per token (naive, and slower/deadlock-prone)). Used directly by `FlatRunner` and `PaddedRunner`. |
| `driver.py` | The shared outer loop: admits arrivals, drives `engine.step()`, records metrics, releases finished requests. Defines the `Engine` protocol every engine implements, `WallClock` (real time) and `VirtualClock` (reproducible, advances only by measured model time), and the CLI (`python driver.py --trace ... --engine ...`). |
| `engines.py` | Import-side-effects registry: importing this populates `driver.ENGINES` with every engine below. |

### Scheduler / runner split, and the engines built from it

| File | What it is |
|---|---|
| `scheduler.py` | `IterationLevelScheduler` — Algorithm 1's `Select()`, implemented exactly once. Owns admission policy (FCFS by arrival time, `max_bs`, optional `max_batched_tokens`) and the K/V slot budget (`break` vs `continue` on a failed charge, reservation vs naive accounting). Delegates all actual storage/compute to whichever `Runner` it's constructed with via `runner.alloc()` / `runner.run()` / `runner.free()`. Also defines `KVDeadlock`. |
| `runner.py` | The `Runner` protocol plus three implementations, each request-scoped (no batch-scoped state anywhere, which is why none of them need a `begin_batch`/`end_batch` pair): `SingleRunner` (HF's own incremental cache, one request per forward, looped for larger batches — no cross-request compute sharing at all); `FlatRunner` (selective batching, wraps `selective_batching_forward` + `RequestKV`); `PaddedRunner` (dense `[B,L,H]` forward — gathers each request's persisted K/V into a fresh padded tensor every call via an explicit 4D causal mask, runs one step, scatters the result back out — the "naive continuous batching" pattern that predates selective batching). |
| `engine_orca.py` | `OrcaEngine(lm, **kw) = IterationLevelScheduler(FlatRunner(lm), **kw)`. Also registers `orca_no_reserve` (naive K/V accounting — §4.2's deadlock) and `orca_greedy_select` (`continue` instead of `break` in `Select()` — breaks iteration-level FCFS). |
| `engine_ablations.py` | `iter_naive = IterationLevelScheduler(SingleRunner(lm), **kw)` and `iter_padded = IterationLevelScheduler(PaddedRunner(lm), **kw)` — the *same* scheduler as `orca`, different runner. Isolates C1 (iteration-level scheduling: no head-of-line blocking, immediate return, mid-batch joins) from C2 (selective batching's shared/flat compute). |
| `engine_static.py` | `StaticBatchingEngine`. Left-padded prompt batches (so `logits[:, -1, :]` is always the true last token of every row, and explicit `position_ids` so HF doesn't silently shift a padded row's RoPE positions). Dead rows get `attention_mask=0` on new columns (keeps the KV cache uncorrupted) but the forward pass still runs on them — that's the point. Supports a `slot_policy="max_seq_len"` option that reproduces FasterTransformer's actual fixed-preallocation OOM behavior (§6.1) as an optional ablation. Kept as its own request-level-scheduling engine rather than retrofitted onto `scheduler.py`, since request-level admission ("close the batch, drain it, only then admit more") is a genuinely different policy from Algorithm 1, not a parameterization of it. |
| `engine_no_batch.py` | The correctness oracle. Batch size always 1, manual greedy-decoding loop, HF's own incremental cache. Every other engine's `generated_ids` must match this exactly under the same trace. |
| `selective_batching.py` | The mechanism behind `FlatRunner`: one flat `[sum(L), H]` tensor through every parameterised op (embedding, q/k/v projections, MLP, norms — all reused directly from the loaded HF model, no reimplemented math), **Split** immediately before attention (each request attends only to its own K/V), **Merge** immediately after. RoPE and GQA (`repeat_kv`) helpers are borrowed live from the loaded model's own `modeling_*` module so this can't silently drift from the model family's convention. |

### Validation scripts (run these before trusting any benchmark numbers)

| File | What it checks | Needs real weights? |
|---|---|---|
| `test_pipeline.py` | `no_batch` vs `static` on an adversarial hand-built trace (different lengths, different gen lengths, a late-arriving lone request); prints the C1 story (`hold_after_completion` per request) concretely. | No (tiny model) |
| `test_static_v2.py` | Left-padding correctness vs oracle at lengths 5/12/40; waste-shape sanity check (`pad_tokens`, `finished_tokens` both nonzero and right order of magnitude) at realistic `(32,512)×(1,128)` length ranges. | No (tiny model) |
| `test_selective_tiny.py` | `selective_batching_forward` vs HF's dense `model(...)` forward: two different-length prefills, a mixed prefill+decode batch, a three-way pure-decode batch. Max logit delta should be float32 noise (~1e-7). | No (tiny model) |
| `test_orca_v1.py` | `orca` vs oracle including a mid-batch "hitchhiker" arrival; confirms `pad_tokens == 0` and `finished_tokens == 0` by construction; the §4.2 deadlock demo (`reserve=True` throttles, `reserve=False` raises `KVDeadlock`, identical `n_slots`). | No (tiny model) |
| `test_ablations.py` | `iter_naive` and `iter_padded` vs the *same* oracle, on the *same* mixed-phase trace as `test_orca_v1.py`; confirms both share `orca`'s zero-waste property and both reproduce the §4.2 deadlock with zero engine-specific code. | No (tiny model) |
| `test_mixed_phase.py` | Standing regression guard: asserts every iteration-level engine (`orca`, `iter_naive`, `iter_padded`) actually produces a batch mixing an INITIATION request with INCREMENT requests during a real run (§3 C2 case 3) — catches a future regression toward scheduling prefills and decodes separately. | No (tiny model) |
| `test_static_real.py` | The one absolute-magnitude check that needs real timing: `hold_after_completion_p50` should be **seconds**, not milliseconds, on real Qwen2.5-0.5B. | **Yes** |

### Microbenchmark

| File | What it is |
|---|---|
| `bench_engine.py` | §6.1-style isolation of a **runner's** per-iteration cost from scheduling entirely. Fixes a homogeneous, steady-state-decode batch (no admission, no arrivals, no mixed phases) and times `runner.run(batch)` directly across batch sizes, for `single`/`flat`/`padded`. Run with `--tiny` for an offline structural comparison, or `--model Qwen/Qwen2.5-0.5B` for real numbers. This is deliberately *not* end-to-end throughput — see "What's not built yet". |

## Setup

```bash
pip install -r requirements.txt
```

## Running the validation suite

```bash
python test_pipeline.py
python test_static_v2.py
python test_selective_tiny.py
python test_orca_v1.py
python test_ablations.py
python test_mixed_phase.py
```

All six should print `PASS` and exit 0, using a tiny randomly-initialized
Qwen2 model — no network access or model download required.

Once you have a working connection to `huggingface.co` (this needs the
actual `.safetensors` weight file, ~1GB, not just tokenizer/config):

```bash
python test_static_real.py
```

## Running an actual comparison

```bash
python trace.py --out t.json -n 200 -r 4.0
python driver.py --trace t.json --engine no_batch --fp32
python driver.py --trace t.json --engine static --max-bs 8 --fp32
python driver.py --trace t.json --engine orca --max-bs 8 --fp32 --check-fcfs
python driver.py --trace t.json --engine iter_naive --max-bs 8 --fp32 --check-fcfs
python driver.py --trace t.json --engine iter_padded --max-bs 8 --fp32 --check-fcfs
```

`--check-fcfs` turns on the iteration-level FCFS invariant check (§4.2) at
every step — cheap, and it catches a `continue`-instead-of-`break` scheduler
bug that a throughput number alone would hide. `iter_naive` vs `orca` vs
`iter_padded` at the same `max_bs` is C1-vs-C2, isolated: all three admit
and schedule identically (literally the same scheduler class), so any
throughput/latency gap between them is attributable entirely to the runner.

Every run's exact settings are embedded in `--out`'s JSON under `"config"`,
so a report file is self-describing without needing the command line that
produced it:

```bash
python driver.py --trace t.json --engine orca --max-bs 8 --fp32 --out orca.json
```

To see the §4.2 deadlock on demand (works identically for `orca`,
`iter_naive`, and `iter_padded` — same scheduler, same deadlock, zero
engine-specific handling):

```bash
python driver.py --trace t.json --engine orca_no_reserve --n-slots 4000
python driver.py --trace t.json --engine iter_naive --no-reserve --n-slots 4000
```

`n_slots` needs to be low enough that a few requests exhaust it but high
enough to admit more than one — with `input_len_range=(32,512)` and
`max_gen=(1,128)`, `max_tokens` averages ~336, so `n_slots ≈ 3–5×` that is
the interesting band. The `reserve=True` engines at the *same* `n_slots`
just throttle admission and complete; only `reserve=False` deadlocks. That
side-by-side is the more interesting result — it's a correctness
difference, not a performance one.

## Microbenchmark: isolating a runner's per-iteration cost

```bash
python bench_engine.py --tiny --batch-sizes 1 2 4 8 16 32
python bench_engine.py --model Qwen/Qwen2.5-0.5B --batch-sizes 1 2 4 8 16 32
```

Fixes a homogeneous, steady-state-decode batch (no scheduler, no admission,
no mixed phases — see "What's not built yet" for why this is a deliberately
narrower question than end-to-end throughput) and times `SingleRunner`,
`FlatRunner`, and `PaddedRunner` directly. Expect `single` to scale roughly
linearly with batch size (no cross-request compute sharing at all), `flat`
sub-linearly (shared batched GEMMs, no reassembly cost), and `padded` to sit
between them, growing faster than `flat` as prompt length grows (the O(B×L)
gather/scatter cost is real and separate from the GEMM sharing both get).

## What's not built yet

- `test_correctness.py` — formalize the ad hoc oracle checks above into one
  script that runs every registered engine over the same trace and asserts
  agreement in one place, instead of duplicated per-engine test files.
- `run_experiments.py` — the actual sweep (`arrival_rate` × engine ×
  `max_bs`) and a Figure-10-style plot, using the full driver + scheduler
  (prefill included, mixed workloads, real arrival dynamics) rather than
  `bench_engine.py`'s fixed decode-only batch.
- **Contiguous K/V pool + fragmentation study** — deliberately deferred, not
  forgotten. `kv_manager.py`'s `SlotAllocator` tracks a slot *count*, not
  slot *placement*; it faithfully reproduces the §4.2 admission/deadlock
  behavior the paper describes, but says nothing about fragmentation from
  variable-length, out-of-order allocate/free (the problem PagedAttention
  solves). That's a distinct, interesting follow-on, not a correction to
  what's here.
- **Varlen/FlashAttention-style fused attention** — deliberately not
  attempted on this hardware (no CUDA, only CPU/MPS available); `flash_attn`
  is CUDA-only. `bench_engine.py` covers the same underlying question
  (kernel-launch/reassembly overhead vs GEMM sharing) with what's actually
  runnable here.

One expectation to set for the `run_experiments.py` sweep once it exists:
on a 0.5B model on a single CPU/GPU, Orca's throughput win over static will
be real but nowhere near the paper's reported 36.9×. That number comes from
a 175B model where parameter reads dominate and inter-layer pipelining is
in play. The win here comes almost entirely from eliminating padding and
dead-row compute, so `useful_fraction` (≈1.0 vs ≈0.4–0.6) and
`hold_after_completion_p50` are the numbers that actually carry the
comparison at this scale.
