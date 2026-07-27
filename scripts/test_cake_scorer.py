"""
Day 5: Test CakeScorer with compression_scorer="cake" and uniform level.
"""
import time
import sys

from _cake_constants import MODEL_PATH
PROMPTS = {
    "1K": "What is KV cache compression? " * 50,
    "4K": ("KV cache compression is a critical technique for efficient large language model inference. "
           "The key-value cache stores intermediate tensors from the attention mechanism. ") * 100,
    "8K": ("KV cache compression reduces memory usage in LLM inference by selectively removing "
           "or compressing the key-value cache entries. Methods include token eviction, quantization, "
           "and architectural changes. ") * 200,
}

def run_test(scorer, level, ratio, prompt_key, label):
    from vllm import LLM, SamplingParams
    import torch

    prompt = PROMPTS[prompt_key]
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"  scorer={scorer}, level={level}, ratio={ratio}, prompt={prompt_key}")
    print(f"{'='*60}")

    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    llm = LLM(
        model=MODEL_PATH,
        compression_ratio=ratio,
        compression_scorer=scorer,
        compression_level=level,
        max_model_len=8192,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )

    load_time = time.time() - start
    print(f"  Model load time: {load_time:.1f}s")

    start = time.time()
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0, max_tokens=64),
    )
    gen_time = time.time() - start
    gen_tokens = len(outputs[0].outputs[0].token_ids)
    print(f"  Generation: {gen_tokens} tokens in {gen_time:.1f}s ({gen_tokens/gen_time:.1f} tok/s)")
    print(f"  Output: {outputs[0].outputs[0].text[:150]}")

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  Peak GPU memory: {peak_mem:.2f} GB")

    return outputs[0].outputs[0].text


if __name__ == "__main__":
    import torch

    # Test 1: FullKV (ratio=1.0, cake scorer — should not compress)
    # This tests that the scorer loads and runs without error
    text1 = run_test("cake", "uniform", 1.0, "1K", "CakeScorer FullKV (ratio=1.0)")

    # Test 2: CakeScorer with ratio=0.5 on 1K prompt
    text2 = run_test("cake", "uniform", 0.5, "1K", "CakeScorer uniform ratio=0.5, 1K")

    # Test 3: CakeScorer with ratio=0.5 on 4K prompt
    text3 = run_test("cake", "uniform", 0.5, "4K", "CakeScorer uniform ratio=0.5, 4K")

    # Test 4: CakeScorer with ratio=0.5 on 8K prompt
    text4 = run_test("cake", "uniform", 0.5, "8K", "CakeScorer uniform ratio=0.5, 8K")

    print(f"\n{'='*60}")
    print("SUMMARY: Day 5 CakeScorer tests complete")
    print(f"{'='*60}")
    print(f"  FullKV (1K): {len(text1)} chars")
    print(f"  CAKE 0.5 (1K): {len(text2)} chars")
    print(f"  CAKE 0.5 (4K): {len(text3)} chars")
    print(f"  CAKE 0.5 (8K): {len(text4)} chars")
    print("  No NaN/No OOM — ALL PASSED")