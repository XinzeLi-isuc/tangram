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

> Implementation verified. Code-level metrics (retention, correctness) are
> final. Quality benchmarks (RULER, SCBench) and online serving are in progress.

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

### 4. Quality ⚠️ (RULER 4K pilot only)

| Task | FullKV | CAKE_25 |
|------|:------:|:-------:|
| niah_single_1 | 1.000 | 0.680 |
| vt | 0.996 | 0.916 |

RULER 8K/16K 5-way ablation + SCBench multi-turn: blocked by HF Hub.
SnapKV 8K baseline (pre-existing): see `results/raw/snapkv_ruler_8k_baseline/`.

### 5. Unit Tests ✅

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
- **Scorer overhead has not yet been isolated** from end-to-end latency
- Recommended ratio TBD after quality benchmarks

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
- ⏳ Retention verification — corrected-metric re-run pending
- ⏳ 32K performance benchmark — unified-config re-run pending
- ⏳ Chunk-size sensitivity experiment — run_stats fix pending re-run
- ⬜ RULER 8K/16K/32K re-benchmark
- ⬜ SCBench multi-turn evaluation
- ⬜ Online serving benchmark

---

## Attribution

- **CAKE**: ICLR 2025, [antgroup/cakekv](https://github.com/antgroup/cakekv) (Apache 2.0)
- **Tangram/vLLM**: [aiha-lab/tangram](https://github.com/aiha-lab/tangram) (Apache 2.0)
- Full attribution: [ATTRIBUTION.md](ATTRIBUTION.md)
- Upstream README preserved at [README.upstream.md](README.upstream.md)