# Attribution

## CAKE Algorithm

CAKE (ICLR 2025) — Layer-adaptive KV cache eviction with temporal-aware
token scoring and per-layer budget allocation.

- Paper: "CAKE: Context-Aware KV Cache Eviction for Efficient Long-Context LLM Inference"
- Repository: https://github.com/antgroup/cakekv
- License: Apache 2.0

**CAKE algorithm ported to Tangram/vLLM**:
- `vllm/v1/attention/compression/cake.py` — CakeScorer (token scores + layer preference)
- `vllm/v1/attention/compression/selection_level.py` — CakeLayerLevel (budget allocation)

## Tangram / vLLM

Tangram is a vLLM fork with non-uniform KV cache compression support,
including ragged paging, continuous batching, chunked prefill, and
physical page reclamation.

- Repository: https://github.com/aiha-lab/tangram
- Base: vLLM (https://github.com/vllm-project/vllm)
- License: Apache 2.0

**CAKE-Serve modifications to Tangram**:
- `vllm/v1/attention/compression/scorer.py` — CakeScorer registration
- `vllm/v1/attention/compression/compressor.py` — Preference passing + CAKE-Chunk aggregation
- `vllm/config/cache.py` — CAKE hyperparameters (tau1/tau2/gamma/window/kernel)
- `vllm/engine/arg_utils.py` — CLI args for all CAKE parameters
- `vllm/v1/worker/compression_model_runner_mixin.py` — Config → compressor wiring

## New Files (Original Work)

### Algorithm & Integration
- `vllm/v1/attention/compression/cake.py` — CakeScorer (ported from CAKE reference)
- `tests/cake_serve/` — Unit/integration tests

### Benchmarks & Scripts
- `scripts/_cake_constants.py` — Shared config
- `scripts/bench_offline_batch.py` — Offline batch throughput benchmark
- `scripts/bench_performance.py` — Latency/throughput at 32K
- `scripts/bench_scbench.py` — SCBench quality evaluation
- `scripts/ablation_day15.py` — Parameter ablation experiments
- `scripts/export_cake_reference.py` — CAKE reference data export
- `scripts/phase1_paper_equivalence.py` — Paper-equivalence verification
- `scripts/phase2_chunked_prefill.py` — Chunked prefill simulation

### Documentation
- `docs/architecture.md` — Project architecture
- `docs/limitations.md` — Known limitations
- `docs/benchmark_protocol.md` — Benchmark methodology
- `results/processed/day*.md` — Daily progress reports

## Datasets

- **RULER**: Long-context evaluation benchmark (https://github.com/hsiehjackson/RULER)
- **SCBench**: Multi-turn long-context benchmark (https://modelscope.cn/datasets/microsoft/SCBench)