"""
Day 10: Physical Memory Verification (v3)
Minimal imports at top level to avoid multiprocessing spawn issues.
"""
import json
import os
import subprocess
import sys
import time

OUTPUT_DIR = "results/raw/day10_memory"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIGS = [
    ("FullKV", "snapkv", "uniform", 1.0),
    ("Uniform_50", "snapkv", "uniform", 0.5),
    ("Uniform_25", "snapkv", "uniform", 0.25),
    ("CAKE_50", "cake", "cake_layer", 0.5),
    ("CAKE_25", "cake", "cake_layer", 0.25),
]
LENGTHS = [8192, 16384, 32768]


def get_nvidia_smi():
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=5)
    for line in r.stdout.strip().split("\n"):
        parts = line.split(", ")
        if parts[0] == "0":
            return int(parts[1]), int(parts[2])
    return 0, 0


def main():
    # Import torch and vllm inside main to avoid spawn issues
    import torch
    from vllm import LLM, SamplingParams

    BYTES_PER_TOKEN = 32 * 8 * 128 * 2 * 2
    results = []

    for length in LENGTHS:
        prompt = "KV cache compression reduces memory. " * (length // 50)
        theoretical_gib = BYTES_PER_TOKEN * length / 1024**3

        for label, scorer, level, ratio in CONFIGS:
            print(f"\n{'='*60}")
            print(f"{label}  len={length}  ratio={ratio}")
            print(f"{'='*60}")
            sys.stdout.flush()

            torch.cuda.reset_peak_memory_stats()
            mem_before, _ = get_nvidia_smi()

            try:
                start = time.time()
                llm = LLM(
                    model="/home/lixinze/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
                    compression_ratio=ratio,
                    compression_scorer=scorer,
                    compression_level=level,
                    max_model_len=length + 256,
                    gpu_memory_utilization=0.90,
                )
                load_time = time.time() - start

                start = time.time()
                out = llm.generate([prompt[:length]],
                                   SamplingParams(temperature=0, max_tokens=32))
                gen_time = time.time() - start
                out_text = out[0].outputs[0].text[:60]
                out_len = len(out[0].outputs[0].token_ids)

                del llm
                torch.cuda.empty_cache()
                time.sleep(1)

            except Exception as e:
                print(f"  ERROR: {e}")
                out_text = str(e)[:60]
                out_len = 0
                load_time = 0
                gen_time = 0

            mem_after, mem_free = get_nvidia_smi()
            peak_alloc = torch.cuda.max_memory_allocated() / 1024**3

            results.append({
                "config": label, "length": length, "ratio": ratio,
                "theoretical_full_kv_gib": round(theoretical_gib, 2),
                "expected_kv_at_ratio_gib": round(theoretical_gib * ratio, 2),
                "load_time_s": round(load_time, 1),
                "gen_time_s": round(gen_time, 2),
                "output_len": out_len,
                "output_preview": out_text,
                "peak_allocated_gib": round(peak_alloc, 2),
                "nvidia_smi_used_mib": mem_after,
                "nvidia_smi_free_mib": mem_free,
            })

            print(f"  Load: {load_time:.1f}s  Gen: {gen_time:.2f}s  Peak: {peak_alloc:.2f}GiB  nvidia: {mem_after}MiB")
            sys.stdout.flush()

    with open(f"{OUTPUT_DIR}/memory_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*90}")
    print(f"{'Config':<12} {'Len':<6} {'Ratio':<6} {'Expected':<10} {'PeakAlloc':<10} {'nvidia-smi':<10} {'Status':<8}")
    print(f"{'-'*12} {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for r in results:
        status = 'OK' if r['output_len'] > 0 else 'FAIL'
        print(f"{r['config']:<12} {r['length']:<6} {r['ratio']:<6.2f} "
              f"{r['expected_kv_at_ratio_gib']:<8.2f}GiB "
              f"{r['peak_allocated_gib']:<8.2f}GiB "
              f"{r['nvidia_smi_used_mib']:<8}MiB {status:<8}")
    print(f"\nSaved to {OUTPUT_DIR}/memory_results.json")


if __name__ == "__main__":
    main()