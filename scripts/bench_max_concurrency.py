"""Max stable concurrency benchmark.

Tests maximum concurrent requests without OOM or invalid outputs.
Each concurrency level runs in fresh engine to avoid state leakage.

Usage: CUDA_VISIBLE_DEVICES=2 python scripts/bench_max_concurrency.py --length 32768
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _cake_constants import MODEL_PATH as MODEL
from _experiment_config import (
    CAKE_WINDOW_SIZE, CAKE_N_SINK_TOKENS, CAKE_FLOOR_MIN,
    CAKE_CHUNK_SIZE, CAKE_PAGE_GROUP_SIZE, GPU_MEMORY_UTILIZATION,
)
from _real_data import build_real_prompt_ids

CONFIGS = [
    ("FullKV",  "snapkv", "uniform",    1.0),
    ("CAKE_25", "cake",   "cake_layer", 0.25),
    ("CAKE_50", "cake",   "cake_layer", 0.50),
]

CONCURRENCY_LEVELS = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48]


def test_one(config_name, scorer, level, ratio, seq_len, concurrency):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kwargs = dict(
        model=MODEL,
        compression_ratio=ratio,
        compression_scorer=scorer,
        compression_level=level,
        page_group_size=CAKE_PAGE_GROUP_SIZE,
        compression_window_size=CAKE_WINDOW_SIZE,
        compression_n_sink_tokens=CAKE_N_SINK_TOKENS,
        compression_floor_min=CAKE_FLOOR_MIN,
        compression_chunk_size=CAKE_CHUNK_SIZE,
        max_model_len=seq_len + 256,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        disable_log_stats=True,
    )
    if ratio >= 1.0:
        for k in ["compression_ratio", "compression_scorer", "compression_level"]:
            kwargs.pop(k, None)

    try:
        llm = LLM(**kwargs)
    except Exception as e:
        return {"status": "init_fail", "error": str(e)}

    sp = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    pids = build_real_prompt_ids(tokenizer, seq_len)
    prompts = [TokensPrompt(prompt_token_ids=pids)
               for _ in range(concurrency)]

    t0 = time.time()
    try:
        outs = llm.generate(prompts, sp)
        elapsed = time.time() - t0

        valid = all(len(o.outputs[0].text.strip()) >= 10 for o in outs)
        return {
            "status": "success",
            "concurrency": concurrency,
            "elapsed_sec": round(elapsed, 2),
            "per_req_sec": round(elapsed / concurrency, 2),
            "all_valid": valid,
        }
    except Exception as e:
        return {"status": "fail", "error": str(e)}
    finally:
        del llm
        import gc; gc.collect()
        import torch; torch.cuda.empty_cache()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=32768, choices=[8192, 16384, 32768])
    args = ap.parse_args()

    seq_len = args.length
    print(f"Max concurrency @ {seq_len}")

    all_results = {}
    for label, scorer, level, ratio in CONFIGS:
        print(f"\n{'='*60}\n{label}\n{'='*60}")
        max_ok = 0
        levels = []

        for conc in CONCURRENCY_LEVELS:
            print(f"  conc={conc}...", flush=True)
            r = test_one(label, scorer, level, ratio, seq_len, conc)
            levels.append(r)

            if r["status"] == "success" and r["all_valid"]:
                max_ok = conc
                print(f"    OK {r['elapsed_sec']:.1f}s")
            else:
                print(f"    FAIL: {r.get('error', 'invalid')}")
                break

        all_results[label] = {"max_stable": max_ok, "levels": levels}
        print(f"  MAX={max_ok}")

    out_dir = "results/raw/day21_concurrency"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"concurrency_{seq_len}.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
