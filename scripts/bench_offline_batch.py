"""
Offline Batch Benchmark (formerly "Continuous Batching")
========================================================
Static batch workload: a fixed set of prompts submitted via llm.generate()
in a single call. This is NOT true online continuous batching (which would
use vllm serve + async arrival of requests).

Purpose: measure throughput scaling as batch size grows, isolating the
KV compression benefit from prefix caching and online scheduling effects.

Limitations acknowledged:
  - Requests share a common corpus (prefix overlap possible)
  - No async arrival; all prompts submitted at once
  - enable_prefix_caching is disabled for fair comparison

For true continuous batching benchmarks, use:
    vllm serve ...  +  vllm bench serve ...

Usage:
    conda activate cake-serve
    cd ~/cake-serve
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_offline_batch.py 2>&1
"""
import json
import os
import time
import numpy as np

from _cake_constants import MODEL_PATH as MODEL
OUTPUT_DIR = "results/raw/day12_batching"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# A single long text for prompt generation
TEXT = """KV cache compression is a critical technique for efficient large language model inference. 
The key-value (KV) cache stores the intermediate key and value tensors from the attention mechanism 
across different layers, allowing the model to avoid recomputing them for each new token. However, 
as the context length grows, the KV cache can become extremely large, consuming gigabytes of GPU memory.

There are several approaches to KV cache compression. The first is token eviction, where less important 
tokens are removed from the cache based on attention scores. Methods like SnapKV, H2O, and StreamingLLM 
fall into this category. The second approach is quantization, which reduces the precision of stored 
values from FP16 to INT8 or INT4. The third is architectural modification, such as multi-query attention 
or grouped-query attention.

The goal of KV cache compression is to reduce memory usage while maintaining model quality. 
This is especially important for long-context applications like document summarization, 
multi-turn dialogue, and code generation where the context can be tens of thousands of tokens long.

CAKE (Cascading and Adaptive KV Cache Eviction) is a recent ICLR 2025 method that approaches 
this problem by analyzing layer-specific attention patterns. It observes that different layers 
have different attention behaviors - some layers focus on a small set of tokens while others 
distribute attention more broadly. CAKE allocates more cache budget to layers with more diverse 
or unstable attention patterns, and uses a temporal-aware scoring mechanism that considers both 
the mean and variance of attention scores over time.

This approach is particularly innovative because it treats KV cache allocation as a global optimization 
problem across layers, rather than applying the same compression ratio to every layer."""

# Repeat to make long corpus
CORPUS = (TEXT + "\n\n") * 20


def make_prompts(tokenizer, batch_size, length_dist, rng_seed=42):
    """Generate a batch of prompts with mixed lengths."""
    from vllm import SamplingParams
    rng = np.random.RandomState(rng_seed)
    prompts = []
    lengths = []
    for _ in range(batch_size):
        target_len = rng.choice(length_dist[0], p=length_dist[1])
        # Tokenize and truncate
        encoded = tokenizer.encode(CORPUS)
        # Repeat if needed
        while len(encoded) < target_len:
            encoded = encoded + encoded
        truncated = encoded[:target_len]
        text = tokenizer.decode(truncated)
        prompts.append(text)
        lengths.append(target_len)
    return prompts, lengths


def run_config(name, scorer, level, ratio, batch_sizes, lengths, max_tokens=128):
    """Run one config across all batch sizes with a single model load."""
    print(f"\n{'='*70}")
    print(f"  CONFIG: {name}")
    print(f"  scorer={scorer}, level={level}, ratio={ratio}")
    print(f"  batch_sizes={batch_sizes}")
    print(f"{'='*70}")
    
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    # Load model once with max possible batch size
    max_bs = max(batch_sizes)
    max_len = max(lengths[0]) + max_tokens + 256
    
    print(f"  Loading model (max_num_seqs={max_bs}, max_model_len={max_len})...")
    llm = LLM(
        model=MODEL,
        compression_ratio=ratio,
        compression_scorer=scorer,
        compression_level=level,
        max_model_len=max_len,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=1,
        max_num_seqs=max_bs,
    )
    
    sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
    
    results = {}
    for bs in batch_sizes:
        # Generate prompts for this batch size
        prompts, prompt_lengths = make_prompts(
            tokenizer, bs, lengths, rng_seed=42 + bs)
        
        print(f"  Batch size {bs}: {len(prompts)} prompts, "
              f"avg_len={np.mean(prompt_lengths):.0f}, "
              f"max_len={max(prompt_lengths)}")
        
        try:
            start = time.time()
            outputs = llm.generate(prompts, sampling_params)
            elapsed = time.time() - start
            
            n_completed = len(outputs)
            total_output = sum(len(o.outputs[0].token_ids) for o in outputs if o.outputs)
            total_input = sum(len(o.prompt_token_ids) for o in outputs)
            
            throughput_req = n_completed / elapsed
            throughput_tok = total_output / elapsed
            
            result = {
                "batch_size": bs,
                "n_completed": n_completed,
                "elapsed_s": round(elapsed, 2),
                "throughput_req_per_s": round(throughput_req, 2),
                "throughput_tok_per_s": round(throughput_tok, 2),
                "avg_prompt_len": int(np.mean(prompt_lengths)),
                "max_prompt_len": max(prompt_lengths),
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
            }
            results[bs] = result
            print(f"    ✅ {elapsed:.1f}s, {throughput_req:.1f} req/s, {throughput_tok:.1f} tok/s")
            
        except Exception as e:
            error_msg = str(e)[:300]
            print(f"    ❌ FAIL: {error_msg}")
            results[bs] = {
                "batch_size": bs,
                "n_completed": 0,
                "error": error_msg,
                "throughput_req_per_s": 0,
                "throughput_tok_per_s": 0,
            }
            # On OOM, stop higher batch sizes
            if "OOM" in error_msg or "out of memory" in error_msg.lower():
                print("    → OOM, stopping higher batch sizes")
                break
    
    del llm
    return results


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def main():
    print("=" * 70)
    print("Day 12: Continuous Batching Benchmark (v2)")
    print("=" * 70)
    
    # Mixed length distribution: 25% 4K, 35% 8K, 25% 16K, 15% 32K
    length_dist = ([4096, 8192, 16384, 32768], [0.25, 0.35, 0.25, 0.15])
    
    # Test fewer batch sizes to save time
    batch_sizes = [1, 2, 4, 8, 16, 24]
    
    all_results = {}
    
    # Test 1: FullKV
    r1 = run_config("FullKV", "snapkv", "uniform", 1.0, batch_sizes, length_dist)
    all_results["fullkv"] = r1
    
    # Test 2: CAKE-Serve 50%
    r2 = run_config("CAKE_50", "cake", "cake_layer", 0.5, batch_sizes, length_dist)
    all_results["cake_50"] = r2
    
    # Test 3: CAKE-Serve 25%
    r3 = run_config("CAKE_25", "cake", "cake_layer", 0.25, batch_sizes, length_dist)
    all_results["cake_25"] = r3
    
    # === Summary ===
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    header = f"{'Config':<15} {'Batch':>6} {'Status':>10} {'Time(s)':>8} {'req/s':>8} {'tok/s':>8}"
    print(f"\n{header}")
    print("-" * 60)
    for config_name in ["fullkv", "cake_50", "cake_25"]:
        config_results = all_results.get(config_name, {})
        for bs in sorted(config_results.keys()):
            r = config_results[bs]
            status = "✅" if r["n_completed"] > 0 else "❌"
            t = r.get("elapsed_s", "N/A")
            t_str = f"{t:.1f}" if t else "OOM"
            print(f"{config_name:<15} {bs:>6} {status:>10} {t_str:>8} "
                  f"{r.get('throughput_req_per_s', 0):>8.1f} "
                  f"{r.get('throughput_tok_per_s', 0):>8.1f}")
    
    # Save
    output = {
        "config": {"model": MODEL, "max_tokens": 128, "length_dist": length_dist},
        "results": {k: {str(bs): v for bs, v in r.items()} for k, r in all_results.items()},
    }
    with open(os.path.join(OUTPUT_DIR, "bench_results.json"), "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n  Results: {OUTPUT_DIR}/bench_results.json")
    print("\n[DONE]")


if __name__ == "__main__":
    main()