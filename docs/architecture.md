# CAKE-Serve Architecture

## Overview

CAKE-Serve integrates the ICLR 2025 CAKE KV Cache eviction algorithm into
Tangram (a vLLM fork with non-uniform KV cache compression).

CAKE adds two capabilities missing from Tangram's existing scorers:
1. **Temporal-aware token scoring**: `S = Mean_q(A) + γ × Var_q(A)`
2. **Layer-adaptive budget allocation**: `B_l ∝ P_l = Entropy(A)^(1/τ1) × Var(A)^(1/τ2)`

## Architecture

```
┌─────────────────────────────────────────────────────┐
│ vLLM Engine (llm_engine.py)                         │
│  └─ EngineArgs → CacheConfig (CAKE hyperparameters)│
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│ CompressionModelRunnerMixin                         │
│  └─ set_qk_scorers(cake_window, cake_gamma, ...)   │
│  └─ _run_compression_layer_loop()                  │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│ KVCompressor (compressor.py)                        │
│  ┌─ set_qk_scorers() → build_qk_scorer("cake")    │
│  ├─ receive_score() → pending_preference           │
│  ├─ prepare_keep_decision() → _collect_preferences │
│  └─ compute_kept_lengths_per_rank()               │
└──────┬─────────────────────┬───────────────────────┘
       │                     │
┌──────▼──────┐     ┌───────▼──────────────┐
│ CakeScorer  │     │ CakeLayerLevel       │
│ (cake.py)   │     │ (selection_level.py) │
│             │     │                      │
│ token_      │     │ compute_counts()     │
│ scores +    │     │   ↓                  │
│ layer_      │     │ floor→cap→          │
│ preference  │     │ redistribute→round   │
└─────────────┘     └──────────────────────┘
       │                     │
       └──────────┬──────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│ CompressionExecutor (executor.py)                   │
│  └─ Write keep decision → free physical KV pages   │
└─────────────────────────────────────────────────────┘
```

## Key Design Decisions

1. **Two-axis separation**: Scorer (axis 2: token importance) and Level (axis 1: budget
   allocation) are orthogonal. This enables component-level ablation:
   `scorer={snapkv|cake}` × `level={uniform|cake_layer}`.

2. **Per-request state isolation**: Layer preferences live in
   `req_state[req_id].pending_preference`, not in a global scorer object. This
   avoids cross-request state pollution under continuous batching.

3. **Chunked prefill adaptation (CAKE-Chunk)**: Under chunked prefill, each chunk's
   CakeScorer produces a local preference estimate. These are aggregated via
   token-weighted averaging. This is a serving approximation of the original CAKE
   one-shot algorithm.

4. **Page-group budget**: All page groups within a layer share the same
   CAKE-allocated budget (preference is per-layer, not per-group).

## File Map

| File | Role |
|------|------|
| `vllm/v1/attention/compression/cake.py` | CakeScorer (token scores + layer preference) |
| `vllm/v1/attention/compression/scorer.py` | Registry: adds CakeScorer to _QK_SCORERS |
| `vllm/v1/attention/compression/selection_level.py` | CakeLayerLevel (budget allocation) |
| `vllm/v1/attention/compression/compressor.py` | KVCompressor: preference passing + chunk aggregation |
| `vllm/config/cache.py` | CacheConfig: all CAKE hyperparameters |
| `vllm/engine/arg_utils.py` | CLI args: --compression-cake-* |
| `vllm/v1/worker/compression_model_runner_mixin.py` | Wires config → compressor |