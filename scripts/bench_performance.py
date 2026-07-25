"""
Day 14: Performance Benchmark (32K context, 128 output tokens)
FullKV vs CAKE 25% vs CAKE 50%
Each config: warmup 2x, measurement 5x
"""
import json, os, time, numpy as np

MODEL = "/home/lixinze/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "results/raw/day14_perf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEXT = ("KV cache compression is a critical technique for efficient LLM inference. " * 5000)

def make_32k_prompt(tokenizer):
    encoded = tokenizer.encode(TEXT)
    return tokenizer.decode(encoded[:32768])

def run_config(name, scorer, level, ratio, batch_sizes, warmup=2, trials=5):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_32k = make_32k_prompt(tokenizer)
    sp = SamplingParams(temperature=0, max_tokens=128)
    results = {}
    
    for bs in batch_sizes:
        print(f"\n  [{name}] batch={bs}", flush=True)
        prompts = [prompt_32k] * bs
        try:
            llm = LLM(model=MODEL, compression_ratio=ratio,
                      compression_scorer=scorer, compression_level=level,
                      max_model_len=33792, gpu_memory_utilization=0.90,
                      tensor_parallel_size=1, max_num_seqs=bs)
            for w in range(warmup):
                _ = llm.generate(prompts, sp)
                print(f"    warmup {w+1}/{warmup} done", flush=True)
            times = []
            for t in range(trials):
                t0 = time.time()
                out = llm.generate(prompts, sp)
                elapsed = time.time() - t0
                times.append(elapsed)
                print(f"    trial {t+1}/{trials}: {elapsed:.1f}s", flush=True)
            t_arr = np.array(times)
            results[bs] = {
                "batch_size": bs, "config": name,
                "median_s": float(np.median(t_arr)),
                "p50_s": float(np.percentile(t_arr, 50)),
                "p95_s": float(np.percentile(t_arr, 95)),
                "mean_s": float(np.mean(t_arr)),
                "std_s": float(np.std(t_arr)),
                "times_s": t_arr.tolist(),
                "throughput_req_s": float(bs / np.median(t_arr)),
            }
            del llm
        except Exception as e:
            print(f"    FAIL: {str(e)[:200]}", flush=True)
            results[bs] = {"batch_size": bs, "error": str(e)[:300]}
            if "OOM" in str(e).upper():
                break
        with open(os.path.join(OUTPUT_DIR, f"{name}_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
    return results

print("Day 14: Performance Benchmark (32K)", flush=True)
print("="*60, flush=True)

batch_sizes = [1, 2, 4, 6, 8, 10]

r1 = run_config("FullKV", "snapkv", "uniform", 1.0, batch_sizes)
r2 = run_config("CAKE_25", "cake", "cake_layer", 0.25, batch_sizes)
r3 = run_config("CAKE_50", "cake", "cake_layer", 0.5, batch_sizes)

print("\n\n" + "="*60, flush=True)
print("SUMMARY", flush=True)
print("="*60, flush=True)
print(f"{'Config':<12} {'Batch':>5} {'Med(s)':>8} {'P95(s)':>8} {'req/s':>8}", flush=True)
print("-"*45, flush=True)
for name, r in [("FullKV", r1), ("CAKE_25", r2), ("CAKE_50", r3)]:
    for bs in sorted(r.keys()):
        d = r[bs]
        if "median_s" in d:
            print(f"{name:<12} {bs:>5} {d['median_s']:>8.1f} {d['p95_s']:>8.1f} {d['throughput_req_s']:>8.1f}", flush=True)
        else:
            print(f"{name:<12} {bs:>5} {'OOM':>8}", flush=True)

all_results = {"FullKV": r1, "CAKE_25": r2, "CAKE_50": r3}
with open(os.path.join(OUTPUT_DIR, "perf_results.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nResults saved", flush=True)
print("DONE", flush=True)