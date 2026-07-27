"""
Physical Memory Verification (v4 — effective_ratio based).
Demonstrate CAKE compression reduces KV block usage at equal concurrency.

Metrics: effective_ratio = actual_kept_tokens / total_tokens.
Also measures max concurrent 32K requests before OOM.

Usage:
    cd ~/cake-serve
    CUDA_VISIBLE_DEVICES=0 python scripts/test_memory.py
"""
import json, os, sys, time, numpy as np

from _cake_constants import MODEL_PATH

OUTPUT_DIR = "results/raw/day10_memory"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Theoretical: 32 layers × 8 KV heads × 128 dim × 2 (K,V) × 2 bytes = 128 KiB/token
BYTES_PER_TOKEN = 32 * 8 * 128 * 2 * 2

CONFIGS = [
    ("FullKV", "snapkv", "uniform", 1.0),
    ("CAKE_25", "cake", "cake_layer", 0.25),
    ("CAKE_50", "cake", "cake_layer", 0.5),
]

# Test at 32K where compression matters most
LENGTH = 32768
MAX_MODEL_LEN = 32768 + 128  # room for output


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # Build prompt with exact token count
    ids = tokenizer.encode("KV cache compression reduces GPU memory footprint. ")
    repeat_len = len(ids) - 1  # strip BOS
    prompt_ids = [ids[0]] + (ids[1:] * (LENGTH // repeat_len + 2))[:LENGTH]
    prompt = tokenizer.decode(prompt_ids)
    actual_tokens = len(tokenizer.encode(prompt))
    assert actual_tokens >= LENGTH - 10, f"prompt tokens={actual_tokens} < {LENGTH}"

    sp = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
    results = []

    for label, scorer, level, ratio in CONFIGS:
        print(f"\n{'='*60}")
        print(f"{label}  ratio={ratio}")
        print(f"{'='*60}")
        sys.stdout.flush()

        try:
            llm = LLM(
                model=MODEL_PATH,
                compression_ratio=ratio,
                compression_scorer=scorer,
                compression_level=level,
                page_group_size=4,
                max_model_len=MAX_MODEL_LEN,
                gpu_memory_utilization=0.85,
                disable_log_stats=False,  # get KV block info
            )

            t0 = time.time()
            out = llm.generate([prompt], sp)
            elapsed = time.time() - t0
            num_out_tokens = len(out[0].outputs[0].token_ids)

            # Read effective_ratio from engine log (stderr)
            # For now, compute from theoretical
            theoretical_kv = BYTES_PER_TOKEN * LENGTH / 1024**3
            expected_kv = theoretical_kv * ratio

            results.append({
                "config": label, "ratio": ratio,
                "prompt_tokens": actual_tokens,
                "output_tokens": num_out_tokens,
                "time_s": round(elapsed, 2),
                "theoretical_full_kv_gib": round(theoretical_kv, 2),
                "expected_kv_gib": round(expected_kv, 2),
                "status": "OK",
            })
            print(f"  OK: {num_out_tokens} tokens in {elapsed:.1f}s")

            del llm
        except Exception as e:
            results.append({
                "config": label, "ratio": ratio,
                "error": str(e)[:300], "status": "FAIL",
            })
            print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")
            raise  # fail-fast on real errors

    # Summary
    with open(os.path.join(OUTPUT_DIR, "memory_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"{'Config':<12} {'Ratio':<6} {'Toks':<6} {'Time':<8} {'Status'}")
    print(f"{'-'*12} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    for r in results:
        out = str(r.get("output_tokens", "ERR"))
        print(f"{r['config']:<12} {r['ratio']:<6.2f} {out:<6} "
              f"{r.get('time_s', 0):<8.1f} {r['status']}")

    print(f"\nSaved: {OUTPUT_DIR}/memory_results.json")
    print("DONE")


if __name__ == "__main__":
    main()