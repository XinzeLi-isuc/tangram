"""
Test: CAKE-Serve with chunked prefill enabled via vLLM API.
Verifies that the full integration path works with compression_chunk_size < prompt length.
"""
import sys
import os
import json
import time

from _cake_constants import MODEL_PATH as MODEL

# Short prompt that fits in one chunk
SHORT_PROMPT = "What is KV cache compression? Explain briefly."

# Long prompt that forces chunked prefill (> 2048 tokens)
LONG_TEXT = """KV cache compression is a critical technique for efficient large language model inference. 
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

# Repeat to make it long
LONG_PROMPT = LONG_TEXT + "\n\n" + LONG_TEXT + "\n\n" + LONG_TEXT + "\n\n" + LONG_TEXT

OUTPUT_DIR = "results/raw/day11_chunked_integration"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from vllm import LLM, SamplingParams

def test_config(scorer, level, ratio, chunk_size, prompt, name, max_tokens=64):
    print(f"\n{'='*60}")
    print(f"  [{name}] scorer={scorer}, level={level}, ratio={ratio}, chunk_size={chunk_size}")
    print(f"  Prompt length: ~{len(prompt)} chars")
    print(f"{'='*60}")
    
    try:
        llm = LLM(
            model=MODEL,
            compression_ratio=ratio,
            compression_scorer=scorer,
            compression_level=level,
            compression_chunk_size=chunk_size,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=1,
        )
        
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_tokens,
        )
        
        start = time.time()
        outputs = llm.generate([prompt], sampling_params)
        elapsed = time.time() - start
        
        result = {
            "name": name,
            "scorer": scorer,
            "level": level,
            "ratio": ratio,
            "chunk_size": chunk_size,
            "success": True,
            "elapsed_s": round(elapsed, 2),
            "output_len": len(outputs[0].outputs[0].token_ids) if outputs[0].outputs else 0,
            "output_text": outputs[0].outputs[0].text[:200] if outputs[0].outputs else "",
        }
        
        print(f"  ✅ Success: {result['output_len']} tokens, {elapsed:.1f}s")
        print(f"  Output: {result['output_text'][:100]}")
        
        del llm
        return result
        
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return {
            "name": name,
            "scorer": scorer,
            "level": level,
            "ratio": ratio,
            "chunk_size": chunk_size,
            "success": False,
            "error": str(e),
        }

def main():
    results = []
    
    # Test 1: One-shot (default chunk_size=2048, short prompt < 2048)
    r = test_config(
        scorer="cake", level="cake_layer", ratio=0.5,
        chunk_size=2048, prompt=SHORT_PROMPT,
        name="one-shot_short"
    )
    results.append(r)
    
    # Test 2: One-shot (large chunk_size, short prompt)
    # This tests the non-chunked path
    r = test_config(
        scorer="cake", level="cake_layer", ratio=0.5,
        chunk_size=4096, prompt=SHORT_PROMPT,
        name="one-shot_large_chunk"
    )
    results.append(r)
    
    # Test 3: Chunked prefill (small chunk_size, long prompt)
    # Long prompt > 2048, chunk_size=1024 → forces 2+ chunks
    r = test_config(
        scorer="cake", level="cake_layer", ratio=0.5,
        chunk_size=1024, prompt=LONG_PROMPT,
        name="chunked_1024"
    )
    results.append(r)
    
    # Test 4: Chunked prefill (even smaller chunk)
    if r["success"]:
        r = test_config(
            scorer="cake", level="cake_layer", ratio=0.5,
            chunk_size=512, prompt=LONG_PROMPT,
            name="chunked_512"
        )
        results.append(r)
    
    # Save results
    with open(os.path.join(OUTPUT_DIR, "integration_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"  {status} {r['name']}: chunk_size={r.get('chunk_size')}, "
              f"output_len={r.get('output_len', 'N/A')}, "
              f"time={r.get('elapsed_s', 'N/A')}s")
    
    n_pass = sum(1 for r in results if r["success"])
    print(f"\n  Passed: {n_pass}/{len(results)}")
    print(f"\n[DONE]")

if __name__ == "__main__":
    main()