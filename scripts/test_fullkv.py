"""
Day 1: Test FullKV (compression_ratio=1.0) and SnapKV (compression_ratio=0.5)
on Llama-3.1-8B-Instruct.
"""
import time
import sys

from _cake_constants import MODEL_PATH
PROMPT = "What is KV cache compression in large language models? Explain the key ideas in 3 paragraphs."
MAX_TOKENS = 128

def run_test(scorer, level, ratio, label):
    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"  scorer={scorer}, level={level}, ratio={ratio}")
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
        [PROMPT],
        SamplingParams(temperature=0, max_tokens=MAX_TOKENS),
    )
    gen_time = time.time() - start
    gen_tokens = len(outputs[0].outputs[0].token_ids)
    print(f"  Generation: {gen_tokens} tokens in {gen_time:.1f}s ({gen_tokens/gen_time:.1f} tok/s)")
    print(f"  Output: {outputs[0].outputs[0].text[:200]}")

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  Peak GPU memory (allocated): {peak_mem:.2f} GB")

    return outputs[0].outputs[0].text


if __name__ == "__main__":
    import torch

    # Test 1: FullKV
    fullkv_text = run_test("snapkv", "uniform", 1.0, "FullKV (ratio=1.0, no compression)")

    # Test 2: SnapKV 0.5
    snapkv_text = run_test("snapkv", "uniform", 0.5, "SnapKV uniform ratio=0.5")

    print(f"\n{'='*60}")
    print("SUMMARY: Day 1 baselines complete")
    print(f"{'='*60}")
    print(f"FullKV  output length: {len(fullkv_text)} chars")
    print(f"SnapKV  output length: {len(snapkv_text)} chars")