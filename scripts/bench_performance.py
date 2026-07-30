"""Offline batch performance benchmark with real SCBench data.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/bench_performance.py [8192] [16384]
"""
import json, os, sys, time
import numpy as np

from _cake_constants import MODEL_PATH as MODEL
from _real_data import build_real_prompt_ids
from vllm.inputs import TokensPrompt
from _experiment_config import (
    CAKE_WINDOW_SIZE, CAKE_N_SINK_TOKENS, CAKE_FLOOR_MIN,
    CAKE_CHUNK_SIZE, CAKE_PAGE_GROUP_SIZE,
    MAX_OUTPUT_TOKENS, GPU_MEMORY_UTILIZATION,
)

OUT_DIR = "results/raw/day14_perf"
os.makedirs(OUT_DIR, exist_ok=True)

CONFIGS = [
    ("FullKV", "snapkv", "uniform", 1.0),
    ("CAKE_25", "cake", "cake_layer", 0.25),
    ("CAKE_50", "cake", "cake_layer", 0.5),
]
BATCHES = [1, 4, 8, 10]
WARMUP, TRIALS = 2, 5


def run_config(name, scorer, level, ratio, prompt_ids, bs_list):
    from vllm import LLM, SamplingParams

    inp_len = len(prompt_ids)
    model_len = inp_len + MAX_OUTPUT_TOKENS
    sp = SamplingParams(temperature=0, max_tokens=MAX_OUTPUT_TOKENS,
                        min_tokens=MAX_OUTPUT_TOKENS, ignore_eos=True)
    results = {}

    for bs in bs_list:
        print(f"\n  [{name}] batch={bs}", flush=True)
        try:
            llm = LLM(
                model=MODEL, compression_ratio=ratio,
                compression_scorer=scorer, compression_level=level,
                page_group_size=CAKE_PAGE_GROUP_SIZE,
                compression_window_size=CAKE_WINDOW_SIZE,
                compression_n_sink_tokens=CAKE_N_SINK_TOKENS,
                compression_floor_min=CAKE_FLOOR_MIN,
                compression_chunk_size=CAKE_CHUNK_SIZE,
                max_model_len=model_len,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                max_num_seqs=bs + 2,
            )

            prompt = TokensPrompt(prompt_token_ids=prompt_ids)
            prompt_list = [prompt] * bs

            for _ in range(WARMUP):
                _ = llm.generate(prompt_list, sp)

            times = []
            for trial in range(TRIALS):
                t0 = time.time()
                out = llm.generate(prompt_list, sp)
                elapsed = time.time() - t0
                times.append(elapsed)
                out_lens = [len(o.outputs[0].token_ids) for o in out]
                if not all(l == MAX_OUTPUT_TOKENS for l in out_lens):
                    raise RuntimeError(
                        f"Short output: {out_lens}, expected {MAX_OUTPUT_TOKENS}")
                print(f"    trial {trial+1}/{TRIALS}: {elapsed:.1f}s "
                      f"(output={out_lens[:3]}...)", flush=True)

            del llm
            med = round(float(np.median(times)), 2)
            p95 = round(float(np.percentile(times, 95)), 2)
            thr = round(bs / med, 4)

            results[str(bs)] = {
                "batch_size": bs, "config": name, "input_tokens": inp_len,
                "median_s": med, "p95_s": p95, "throughput_req_s": thr,
                "trials": times,
            }
            print(f"    median={med}s p95={p95}s thr={thr} req/s", flush=True)

        except Exception as e:
            print(f"    FAIL: {e}", flush=True)
            results[str(bs)] = {"batch_size": bs, "config": name, "error": str(e)}

    out_path = os.path.join(OUT_DIR, f"{name}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    lengths = [int(l) for l in sys.argv[1:]] if len(sys.argv) > 1 else [32768]

    for length in lengths:
        prompt_ids = build_real_prompt_ids(tokenizer, length)
        print(f"\n{'#'*60}")
        print(f"# PERFORMANCE: {length} tokens")
        print(f"{'#'*60}")

        all_results = {}
        for label, scorer, level, ratio in CONFIGS:
            print(f"\n{'='*50}")
            print(f"  {label}")
            print(f"{'='*50}")
            r = run_config(label, scorer, level, ratio, prompt_ids, BATCHES)
            all_results[label] = {"length": length, "results": r}

        # Summary
        print(f"\n{'='*60}\nSUMMARY {length}\n{'='*60}")
        hdr = f"  {'Config':<12} {'Batch':>5} {'Med(s)':>8} {'P95(s)':>8} {'req/s':>8}"
        print(hdr)
        print(f"  {'-'*45}")
        for label in [c[0] for c in CONFIGS]:
            for bs in BATCHES:
                k = str(bs)
                if k in all_results[label]["results"]:
                    d = all_results[label]["results"][k]
                    if "median_s" in d:
                        print(f"  {label:<12} {bs:>5} {d['median_s']:>8.1f} "
                              f"{d['p95_s']:>8.1f} {d['throughput_req_s']:>8.3f}")
        print()

        # Aggregate
        agg_path = os.path.join(OUT_DIR, f"perf_results_{length}.json")
        with open(agg_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Saved: {agg_path}", flush=True)


if __name__ == "__main__":
    main()
