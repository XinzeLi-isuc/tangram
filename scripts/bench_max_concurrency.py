"""Step 4: Max concurrency test — binary search for largest batch before OOM/preemption.

Each config: increase batch until OOM. Record max stable batch + throughput.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/bench_max_concurrency.py
"""
import json, os, time, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _cake_constants import MODEL_PATH as MODEL
from _real_data import build_real_prompt_ids

OUTPUT_DIR = "results/raw/day14_perf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LEN = 32768 + 128
TGT = MAX_LEN - 128
MAX_OUT = 128
WARMUP = 2
TRIALS = 3  # fewer trials since we just need concurrency limit

CONFIGS = [
    ("Tangram_FullKV", "snapkv", "uniform", 4, 1.0),
    ("CAKE_25", "cake", "cake_layer", 4, 0.25),
    ("CAKE_50", "cake", "cake_layer", 4, 0.5),
]


def mk_ids(tokenizer):
    """Build exact TGT token_ids from real SCBench context."""
    return build_real_prompt_ids(tokenizer, TGT)


def test_batch(llm, sp, pids, bs, label):
    pl = [{"prompt_token_ids": pids}] * bs
    times, all_l = [], []
    for ti in range(TRIALS):
        t0 = time.time()
        out = llm.generate(pl, sp)
        e = time.time() - t0
        times.append(e)
        ols = [len(it.outputs[0].token_ids) for it in out]
        all_l.append(ols)
        ok = sum(1 for n in ols if n == MAX_OUT)
        print(f"    trial {ti+1}: {e:.1f}s complete={ok}/{bs}", flush=True)
    for ols in all_l:
        if any(n != MAX_OUT for n in ols):
            return None, f"incomplete output: {ols}"
    ta = np.array(times)
    return {
        "batch_size": bs, "median_s": float(np.median(ta)),
        "mean_s": float(np.mean(ta)), "times_s": ta.tolist(),
        "throughput_req_s": float(bs / np.median(ta)),
        "all_complete": True,
    }, None


def find_max(config_label, scorer, level, page_group_size, ratio):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    pids = mk_ids(tokenizer)
    sp = SamplingParams(temperature=0, max_tokens=MAX_OUT, min_tokens=MAX_OUT, ignore_eos=True)

    results = []
    bs = 1
    max_stable = 0
    max_stable_data = None

    while True:
        print(f"\n[{config_label}] batch={bs}", flush=True)
        try:
            kwargs = dict(model=MODEL, max_model_len=MAX_LEN,
                         gpu_memory_utilization=0.90, max_num_seqs=bs + 2)
            if page_group_size is not None:
                kwargs.update(page_group_size=page_group_size,
                             compression_ratio=ratio, compression_scorer=scorer,
                             compression_level=level)
            llm = LLM(**kwargs)

            for w in range(WARMUP):
                _ = llm.generate([{"prompt_token_ids": pids}] * bs, sp)
            print(f"    warmup done", flush=True)

            data, err = test_batch(llm, sp, pids, bs, config_label)
            del llm

            if err:
                print(f"    ERROR: {err}", flush=True)
                results.append({"batch_size": bs, "error": err})
                bs += 1  # try next, might be transient
                continue

            results.append(data)
            max_stable = bs
            max_stable_data = data
            print(f"    OK: median={data['median_s']:.1f}s thr={data['throughput_req_s']:.3f}req/s", flush=True)
            bs += 1

        except Exception as e:
            err_str = str(e)
            if "OOM" in err_str.upper() or "out of memory" in err_str.lower():
                print(f"    OOM at batch={bs} — max stable={max_stable}", flush=True)
                results.append({"batch_size": bs, "error": "OOM"})
                break
            else:
                print(f"    ERROR: {type(e).__name__}: {err_str[:200]}", flush=True)
                results.append({"batch_size": bs, "error": f"{type(e).__name__}: {err_str[:200]}"})
                break

    path = os.path.join(OUTPUT_DIR, f"{config_label}_maxconcurrency.json")
    out = {"config": config_label, "max_stable": max_stable,
           "max_stable_data": max_stable_data, "results": results}
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  [{config_label}] max={max_stable} saved to {path}", flush=True)
    return max_stable, max_stable_data


print("MAX CONCURRENCY TEST (32K, 128 out)", flush=True)
print("=" * 60, flush=True)

summary = {}
for label, scorer, level, pgs, ratio in CONFIGS:
    max_bs, data = find_max(label, scorer, level, pgs, ratio)
    summary[label] = {"max_batch": max_bs, "median_s": data["median_s"] if data else None}

print("\n" + "=" * 60, flush=True)
print("SUMMARY", flush=True)
print(f"{'Config':<20} {'MaxBatch':>8} {'Med(s)':>8} {'req/s':>8}", flush=True)
print("-" * 46, flush=True)
for label, info in summary.items():
    mb = info["max_batch"]
    ms = info["median_s"]
    thr = mb / ms if ms else 0
    print(f"{label:<20} {mb:>8} {ms:>8.1f} {thr:>8.3f}", flush=True)

print("\nDONE", flush=True)
