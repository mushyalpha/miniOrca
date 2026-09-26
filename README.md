# Mini-Orca

A from-scratch PyTorch implementation of Orca (OSDI '22) iteration-level scheduling and selective batching.

### Key Features
* **Zero Waste** - Selective batching eliminates padding and dead-rows (100% useful compute).
* **Readable Codebase** - Clean implementation of an LLM control plane in ~1,600 lines of Python.
* **Optimization Suite** - 6-engine ablation framework, explicit 4D causal masking, and deadlock-free K/V slot reservation.

### Installation
```bash
git clone https://github.com/mushyalpha/miniOrca.git
cd miniOrca
pip install -r requirements.txt
```

### Model Download
To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen2.5-0.5B \
  --local-dir ~/huggingface/Qwen2.5-0.5B/ \
  --local-dir-use-symlinks False
```

### Quick Start
Mini-Orca provides a CLI to generate Poisson workloads and run ablation comparisons across engines:

```bash
# Generate a Poisson arrival trace (200 requests, 4 req/s)
python trace.py --out traces/workload.json -n 200 -r 4.0 --model ~/huggingface/Qwen2.5-0.5B/

# Run the end-to-end comparison (Static Batching vs. Orca)
python run_all.py \
  --trace traces/workload.json \
  --model ~/huggingface/Qwen2.5-0.5B/ \
  --max-bs 8 \
  --clock wall
```

### Benchmark
See `run_all.py` and `bench_engine.py` for benchmark suites.

**Test Configuration:**
* **Hardware:** NVIDIA Cloud GPU (BF16)
* **Model:** Qwen2.5-0.5B
* **Workload:** 200 Poisson requests (λ = 4.0 req/s)
* **Input/Output Lengths:** Heterogeneous random sampling 

**Performance Results:**

| Inference Engine | Throughput (req/s) | Median Latency (s) | Useful Compute % |
| :--- | :--- | :--- | :--- |
| FasterTransformer (Static) | 3.47 | 1.20 | 69.8% |
| **Mini-Orca (Continuous)** | **3.51** | **0.65** | **100.0%** |

*(Mini-Orca cuts median latency by 45% and eliminates 30% of compute waste caused by padding and dead-rows).*
