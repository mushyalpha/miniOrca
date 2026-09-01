<p align="center">
  <img src="assets/miniOrca_logo.png" alt="miniOrca logo" width="300">
</p>

# miniOrca

implementation from scrath of the scheduling architecture behind the paper **[Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)** (OSDI '22) that introduced **iteration-level scheduling** (continuous batching) and **selective batching** to LLM inference.

## Why I Built This

I built this because I wanted to understand Orca's scheduling ideas by building them myself and measuring the difference. miniOrca implements engines in which the only variable is the scheduling and batching strategy.

## Results

Benchmarked on Qwen2.5-0.5B (fp32, max batch size 8, 200 requests at 4.0 req/s):

| Engine | Throughput (req/s) | Latency p50 (s) | Latency p99 (s) | Token Slot Utilization |
|---|---|---|---|---|
| `no_batch` | 0.89 | 91.6 | 167.7 | 100% |
| `static` | 3.40 | 3.3 | 6.0 | 59.9% |
| **`orca`** | **3.40** | **2.7** | **6.2** | **100%** |
| `iter_naive` | 0.90 | 92.7 | 167.8 | 100% |
| `iter_padded` | 3.21 | 5.2 | 9.1 | 100% |

**Results:** Orca matches static batching's throughput while cutting median latency by 20% and eliminating all padding waste (100% vs 60% useful token slots). Comparing `orca` vs `iter_naive` isolates the impact of selective batching on the same scheduler, ~3.8× throughput difference. The `iter_padded` engine shows that iteration-level scheduling alone recovers most of the throughput, but selective batching still wins on latency by avoiding per-iteration K/V reassembly.

## Quick Start

```bash
pip install -r requirements.txt
python trace.py --out t.json -n 200 -r 4.0
python run_all.py --trace t.json --fp32 --max-bs 8 --out results.json
```
