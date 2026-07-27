# CAKE-Serve Limitations

## Chunked Prefill Approximation (CAKE-Chunk)

The original CAKE algorithm computes layer preference from one full-context
observation window. Under chunked prefill, each chunk sees only its local Q/K
slice. We aggregate local estimates via token-weighted averaging:

    P̂_l = (Σ_t n_t × P_l^(t)) / Σ_t n_t

Spearman ρ ≈ 0.6 at 128-token chunks, approaching 1.0 as chunk → full prompt.
This is a serving adaptation at the cost of strict equivalence.

## Scorer Overhead

CAKE scorer overhead: ~0.58ms/layer vs SnapKV's ~0.11ms/layer (5.1x).
For a 32-layer model, this adds ~18ms per prefill step. This overhead is:
- Acceptable for long contexts (32K tokens: ~80% of time is compute, not scoring)
- Not ideal for short prompts (<2K tokens: scoring becomes 10%+ of latency)

## TP=1 Only

CAKE layer preference is computed from each TP rank's local KV heads. Without
cross-rank aggregation, different ranks may derive different preferences and
inconsistent block decisions. `tensor_parallel_size > 1` with
`scorer=cake + level=cake_layer` raises `NotImplementedError`.

## Retention Ratio Range

- **25%–50%**: Recommended operating range. Quality stable, memory savings significant.
- **10%**: Quality degradation observed. Preference ordering becomes dominated by
  quantization noise at very low retention ratios.

## Short Prompt Benefit

Compression benefit is proportional to KV cache size. For prompts <2K tokens:
- KV cache is small (<256 MB for Llama-3.1-8B)
- Scorer overhead may negate compression savings
- Best ROI at 16K+ contexts

## Not Implemented (First Version)

- Tensor Parallel (TP > 1)
- Multi-GPU / data parallel
- Speculative decoding
- KV Cache CPU/NVMe offload
- Custom sparse attention kernels
- Dynamic per-request compression ratio