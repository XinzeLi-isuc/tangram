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

## Key Results (A6000 48GB, Llama-3.1-8B-Instruct BF16, SCBench real text)

> ⚠️ 32K performance data was collected before the unified config fix (8e2d0ca).
> Re-run pending. Readme stale until re-run completes.

### 1. Smoke Test — 3/3 PASS

| Config | Output | Time | Status |
|--------|:------:|:----:|:------:|
| FullKV | 64 tok | 15.8s | OK |
| CAKE+uniform | 64 tok | 15.1s | OK |
| CAKE+cake_layer | 64 tok | 14.5s | OK |

### 2. Retention Verification (end-to-end physical ratio, logical capacity denominator)

*Extracted with `_retention_utils.summarize_retention()` using `logical_tokens_by_req`.*

| Config | 8K | 16K | 32K | Method |
|--------|:--:|:---:|:---:|--------|
| FullKV | 1.000 | 1.000 | 1.000 | No compression |
| CAKE 25% | ~0.25 | ~0.25 | ~0.25 | cake_layer |
| CAKE 50% | ~0.50 | ~0.50 | ~0.50 | cake_layer |

- window=32, sink=4, floor=0, chunk=2048, page_group=4
- `effective_evictable_ratio` = (kept − sink − win) / (logical − sink − win)
- `final_step_shrink_ratio` = kept / resident_before_final (previous metric, ~0.85)
- Results: `results/raw/day10_memory/` (pending re-run with fixed metrics)

### 3. 32K Performance (32768 tokens, 128 output, batch 1–10)

*Collected pre-config-unification. Real SCBench text. Re-run pending.*

| Config | b=1 | b=2 | b=4 | b=6 | b=8 | b=10 |
|--------|----:|----:|----:|----:|----:|-----:|
| Native-vLLM | 11.7s | 20.5s | 37.7s | — | — | — |
| Tangram FullKV | 11.9s | 20.6s | 37.8s | 55.1s | 75.7s | 93.2s |
| CAKE 50% | 10.9s | 19.6s | 35.7s | 52.3s | 72.8s | 94.1s |
| CAKE 25% | **9.8s** | **17.0s** | **30.8s** | **44.7s** | **58.8s** | **74.6s** |

Speedup vs Tangram FullKV @ batch=10: CAKE_25 = 1.25×, CAKE_50 = 0.99×

### 4. Estimated KV-cache Capacity Reduction

| Config | 8K | 16K | 32K |
|--------|---:|----:|----:|
| CAKE 25% | ~0.75× | ~0.75× | ~0.75× |
| CAKE 50% | ~0.50× | ~0.50× | ~0.50× |

Theoretical KV capacity reduction = 1 − effective_physical_ratio.
NOT measured via nvidia-smi (vLLM pre-allocates GPU pool).

### 5. E2E Verification

| Check | Result |
|-------|--------|
| cake_layer non-uniform | PASS (24/32 layers deviate) |
| Budget correctness (chunk 2048) | PASS (~25% physical ratio) |
| Budget correctness (chunk 8192) | PASS (~25% physical ratio) |
| Chunk-size sensitivity | MODERATE/UNSTABLE (pending re-run) |

Results: `results/raw/day16_e2e/` (pending re-run with fixed metrics)

### 6. Verification

- pytest: **28/28** passed (CPU tests)
- retention parser unit tests: **7/7** passed
- py_compile: **5/5** OK

Full results: `results/raw/smoke/`, `results/raw/day10_memory/`, `results/raw/day14_perf/`
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
| Quality (6-way ablation) | `benchmarks/tangram/benchmark_ruler.sh` | RULER |
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
- **Scorer overhead**: scales with sequence length, acceptable at 32K+
- **CAKE 25% recommended** for best speed/quality trade-off

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
- ✅ Retention verification (effective ratio measured)
- ✅ 32K performance benchmark (real SCBench data)
- ⬜ RULER 8K/16K/32K re-benchmark
- ⬜ SCBench multi-turn evaluation
- ⬜ Online serving benchmark

---

## Attribution

- **CAKE**: ICLR 2025, [antgroup/cakekv](https://github.com/antgroup/cakekv) (Apache 2.0)
- **Tangram/vLLM**: [aiha-lab/tangram](https://github.com/aiha-lab/tangram) (Apache 2.0)
- Full attribution: [ATTRIBUTION.md](ATTRIBUTION.md)
- Upstream README preserved at [README.upstream.md](README.upstream.md)