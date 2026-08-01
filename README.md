# CAKE-Serve: Layer-Adaptive KV Cache Compression for vLLM

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Base: Tangram](https://img.shields.io/badge/base-Tangram%20vLLM-orange.svg)](https://github.com/aiha-lab/tangram)

Integrating the **ICLR 2025 CAKE KV Cache eviction algorithm** into Tangram/vLLM,
making layer-adaptive budget allocation and temporal-aware token scoring work
on a real online serving engine with physical KV page reclamation.

---

## What Problem Does This Solve?

Long-context LLM inference is bottlenecked by KV cache memory. Existing
approaches (SnapKV, StreamingLLM) compress at a **uniform ratio across all
layers** — but not all layers need the same KV budget.

CAKE (ICLR 2025) showed that **different layers have different attention entropy
and temporal variance** — and allocating more budget to "volatile" layers
improves quality at the same total memory.

**But CAKE's official implementation runs on HuggingFace Transformers** with
monkey-patched attention — it cannot be used for online vLLM serving.

**Tangram** already supports non-uniform KV compression with physical page
reclamation — but lacks CAKE's **layer preference model** and **temporal
variance scoring**.

CAKE-Serve bridges this gap.

---

## Quick Start

```bash
# Clone this fork (not official vLLM)
git clone https://github.com/XinzeLi-isuc/tangram.git cake-serve
cd cake-serve
pip install -e .

python -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='meta-llama/Llama-3.1-8B-Instruct',
    compression_ratio=0.25,
    compression_scorer='cake',
    compression_level='cake_layer',
    page_group_size=4,
    compression_cake_window_size=32,
    compression_cake_gamma=1.0,
    compression_cake_tau1=1.0,
    compression_cake_tau2=1.0,
)

outputs = llm.generate(['Long context prompt...'], SamplingParams(temperature=0))
print(outputs[0].outputs[0].text)
"
```

---

## Key Results (A6000 48GB, Llama-3.1-8B-Instruct BF16)

> All numbers below are measured on this fork with the unified config
> (window=32, sink=4, floor=0, chunk=2048, page_group=4). Full logs in
> `results/raw/`.

### 1. KV Cache Retention ✅

End-to-end physical ratio = `kept_token_cells / logical_token_cells`.

| Config | 8K | 16K | 32K |
|--------|:--:|:---:|:---:|
| FullKV | 1.0000 | 1.0000 | 1.0000 |
| **CAKE 25%** | **0.2541** | **0.2537** | **0.2536** |
| CAKE 50% | 0.5037 | 0.5035 | 0.5038 |

- **~74.6% estimated KV capacity reduction** at 25% requested ratio
- window=32, sink=4, floor=0, chunk=2048, page_group=4
- `final_step_shrink_ratio` = kept / resident_before_final ≈ 0.85

### 2. 32K Offline Batch Performance ✅

TokensPrompt, unified config, 128 output tokens per request.

| Config | b=1 | b=4 | b=8 | b=10 | vs FullKV@b=10 |
|--------|----:|----:|----:|-----:|:--------------:|
| FullKV | 11.9s | 38.0s | 76.1s | 93.7s | — |
| **CAKE 25%** | **9.9s** | **31.0s** | **59.2s** | **73.3s** | **1.28×** |
| CAKE 50% | 11.0s | 36.0s | 73.6s | 95.5s | 0.98× |

- 8K: no speedup (KV pool too small); 16K: 1.07×
- Full results: `results/raw/day14_perf/perf_results_*.json`

### 3. E2E Correctness ✅

| Check | Result |
|-------|--------|
| cake_layer non-uniform | PASS (24/32 layers) |
| Budget correctness (chunk 2048) | phys=0.2541 |
| Budget correctness (chunk 8192) | phys=0.2515 |
| Chunk-size stability | MODERATE (Spearman=0.52) |

### 4. Quality: RULER 5-way Ablation ✅ (4K / 8K / 16K)

13 tasks × 50 samples/task, exact-answer matching, macro average accuracy.

| Config | 4K | 8K | 16K |
|--------|:--:|:--:|:--:|
| FullKV (upper bound) | 0.9508 | 0.9385 | 0.9262 |
| SnapKV + uniform | 0.6615 | 0.5323 | 0.4846 |
| SnapKV + crosslayer_cluster | 0.6585 | 0.5431 | 0.4877 |
| CAKE + uniform | 0.6323 | 0.5200 | 0.4908 |
| **CAKE + cake_layer** | **0.6615** | **0.5446** | **0.5277** |

- **CAKE + cake_layer ≥ SnapKV baselines at every length**, with the gap
  widening as context grows: 16K leads SnapKV_cluster by **+4.0 pp**
  (0.5277 vs 0.4877)
- CAKE layer budget (cake_layer vs uniform) is worth **+1.2–3.7 pp**
- Full results: `results/raw/day20_ruler_4096|8192|16384/`

### 5. CAKE Scorer Hyperparameter Sensitivity (γ ablation) ✅

CAKE + cake_layer, tau1=1.6, tau2=0.4 fixed.

| gamma | 4K | 8K |
|-------|:--:|:--:|
| 1 | 0.6662 | 0.5585 |
| 50 | 0.6677 | 0.5554 |
| 200 | 0.6615 | 0.5477 |

- Quality is robust to γ within ±0.01; default (γ=1) is a safe choice
- Full results: `results/raw/day21_paper_params/`

### 6. Online Serving ✅ (16K, 3 configs × 3 QPS)

OpenAI-compatible server, 200 requests, 16K input / 128 output.

| Config | QPS | TTFT p50 | TTFT p99 | tok/s | req/s |
|--------|:--:|:--:|:--:|:--:|:--:|
| FullKV | 2.0 | 141.0s | 283.8s | 4254 | 0.510 |
| **CAKE 25%** | **2.0** | **116.6s** | **229.2s** | **4987** | **0.598** |
| CAKE 50% | 2.0 | 134.7s | 273.6s | 4379 | 0.526 |

- **CAKE 25%: +17.2% token throughput, −17.3% TTFT p50 vs FullKV** at QPS=2.0
- CAKE 50%: +2.9% token throughput
- Full results: `results/raw/day17_serving/`

### 7. Unit Tests ✅

- pytest (CPU): **28/28** passed
- retention parser: **7/7** passed
- py_compile: scripts clean

Full results: `results/raw/day10_memory/`, `results/raw/day14_perf/`, `results/raw/day16_e2e/`
---

## Architecture

```
CakeScorer (token scores + layer preference)
    ↓
KVCompressor (preference aggregation via token-weighted mean)
    ↓
CakeLayerLevel (budget: floor → cap → redistribute → page-align)
    ↓
CompressionExecutor (write keep decision → free physical KV pages)
```

Full architecture: [docs/architecture.md](docs/architecture.md)

---

## Experiment Quick Reference

| Experiment | Script | Dataset |
|-----------|--------|---------|
| Quality (5-way ablation) | `benchmarks/tangram/benchmark_ruler.sh` | RULER |
| Quality (multi-turn) | `SCORER=cake bash benchmarks/tangram/benchmark_scbench.sh` | SCBench |
| Offline batch | `scripts/bench_offline_batch.py` | Synthetic |
| 32K Performance | `scripts/bench_performance.py` | SCBench (real text) |
| Retention / Memory | `scripts/test_memory.py` | SCBench (real text) |
| Smoke test | `scripts/smoke_test.py` | SCBench (real text) |

See [docs/benchmark_protocol.md](docs/benchmark_protocol.md) for full methodology.

---

## CAKE Hyperparameters

| Parameter | CLI Flag | Default |
|-----------|---------|:-------:|
| gamma | `--compression-cake-gamma` | 1.0 |
| tau1 (entropy) | `--compression-cake-tau1` | 1.0 |
| tau2 (variance) | `--compression-cake-tau2` | 1.0 |
| window_size | `--compression-cake-window-size` | 32 |
| kernel_size | `--compression-cake-kernel-size` | 5 |

---

## Limitations

- **TP=1 only**: CakeLayerLevel raises `NotImplementedError` for TP > 1
- **Chunked prefill approximation**: Token-weighted mean preference aggregation
- **Scorer overhead not fully isolated** from end-to-end latency
- **Offline perf**: single repeated prompt per batch; no multi-sample diversity
- **Online serving**: 3 fixed QPS levels, no saturation curve; preemption not
  separately counted in the serving benchmark
- **SCBench multi-turn**: pilot only; 16K truncation of 380K-char contexts and
  exact-string matching limit its usefulness (see `results/raw/day18_scbench_pilot/`)

Full list: [docs/limitations.md](docs/limitations.md)

---

## Tests

```bash
# Install pytest
pip install pytest

# Run all tests
python -m pytest tests/cake_serve -v

# Individual suites
python -m pytest tests/cake_serve/test_p0_budget_invariants.py -v
python -m pytest tests/cake_serve/test_cake_layer_level.py -v
python -m pytest tests/cake_serve/test_preference_chain.py -v
```

---

## Project Status

- ✅ Algorithm port (CakeScorer + CakeLayerLevel)
- ✅ Physical KV page reclamation
- ✅ Chunked prefill adaptation (CAKE-Chunk)
- ✅ Preference lifecycle fix (cake_layer ≠ uniform)
- ✅ Smoke test pipeline (3/3 PASS)
- ✅ Retention verification (8K/16K/32K, corrected metric)
- ✅ 32K performance benchmark (unified config)
- ✅ Chunk-size sensitivity experiment
- ✅ RULER 5-way ablation (4K/8K/16K, 13 tasks)
- ✅ γ hyperparameter ablation (4K/8K)
- ✅ Online serving benchmark (16K, 3 configs × 3 QPS)
- ⬜ SCBench multi-turn evaluation (pilot done; full eval pending)

---

## Attribution

- **CAKE**: ICLR 2025, [antgroup/cakekv](https://github.com/antgroup/cakekv) (Apache 2.0)
- **Tangram/vLLM**: [aiha-lab/tangram](https://github.com/aiha-lab/tangram) (Apache 2.0)
- Full attribution: [ATTRIBUTION.md](ATTRIBUTION.md)
- Upstream README preserved at [README.upstream.md](README.upstream.md)