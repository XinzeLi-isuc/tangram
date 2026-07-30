"""Step 4: Measured max concurrency — increment batch until preemption/OOM.

Preemption detection: if median time > 2x linear projection from previous batch,
or any request fails to produce 128 tokens, stop and report previous batch as max.

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

CONFIGS = [
    ("Tangram_FullKV", "snapkv", "uniform", 4, 1.0),
    ("CAKE_25", "cake", "cake_layer", 4, 0.25),
    ("CAKE_50", "cake", "cake_layer", 4, 0.5),
]


def measure_one(llm, sp, pids, bs):
    """Run 2 warmup + 3 measurement. Returns median_s or None if failed."""
    pl = [{"prompt_token_ids": pids}] * bs

    # 1 warmup
    try:
        _ = llm.generate(pl, sp)
    except Exception as e:
        if "OOM" in str(e).upper():
            return None, "OOM"
        raise

    # 3 measurements
    times = []
    for ti in range(3):
        try:
            t0 = time.time()
            out = llm.generate(pl, sp)
            e = time.time() - t0
        except Exception as ex:
            if "OOM" in str(ex).upper():
                return None, "OOM"
            raise
        times.append(e)
        ols = [len(it.outputs[0].token_ids) for it in out]
        ok = sum(1 for n in ols if n == MAX_OUT)
        if ok < bs:
            return None, f"incomplete: {ok}/{bs} got {MAX_OUT} tokens, lens={ols}"
        if ti == 0:
            print(f"    bs={bs} t1={e:.1f}s {ok}/{bs} ok", flush=True)

    med = float(np.median(times))
    return med, None


def find_max(label, scorer, level, pgs, ratio):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    pids = build_real_prompt_ids(tokenizer, TGT)
    sp = SamplingParams(temperature=0, max_tokens=MAX_OUT, min_tokens=MAX_OUT, ignore_eos=True)

    results = []
    prev_med = None
    max_stable = 0
    bs = 1

    while True:
        print(f"\n[{label}] testing batch={bs}", flush=True)
        result = None
        try:
            kwargs = dict(model=MODEL, max_model_len=MAX_LEN,
                         gpu_memory_utilization=0.90, max_num_seqs=bs + 2)
            if pgs is not None:
                kwargs.update(page_group_size=pgs, compression_ratio=ratio,
                             compression_scorer=scorer, compression_level=level)
            llm = LLM(**kwargs)

            med, err = measure_one(llm, sp, pids, bs)
            del llm

            if err:
                print(f"    ERROR: {err}", flush=True)
                results.append({"batch_size": bs, "error": err, "status": "preempted_or_oom"})
                break

            # Preemption check: time > 2x linear projection
            if prev_med is not None and bs > 1:
                expected = prev_med * (bs / (bs - 1))
                if med > 2.0 * expected:
                    print(f"    PREEMPT: med={med:.1f}s > 2x expected={expected:.1f}s", flush=True)
                    results.append({"batch_size": bs, "median_s": med,
                                    "expected_s": round(expected, 2),
                                    "status": "preempted"})
                    break

            results.append({"batch_size": bs, "median_s": med, "status": "stable"})
            max_stable = bs
            prev_med = med
            print(f"    STABLE: med={med:.1f}s", flush=True)
            bs += 1

        except Exception as e:
            es = str(e)
            if "OOM" in es.upper() or "out of memory" in es.lower():
                print(f"    OOM at batch={bs}", flush=True)
                results.append({"batch_size": bs, "error": "OOM", "status": "oom"})
            else:
                print(f"    ERROR: {type(e).__name__}: {es[:200]}", flush=True)
                results.append({"batch_size": bs, "error": es[:200], "status": "error"})
            break

    # Save
    path = os.path.join(OUTPUT_DIR, f"{label}_maxconcurrency.json")
    out = {"config": label, "max_stable": max_stable, "results": results}
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  [{label}] max_stable={max_stable} -> {path}", flush=True)
    return max_stable


print("MAX CONCURRENCY TEST (32K real SCBench, preemption-aware)", flush=True)
print("=" * 60, flush=True)

summary = {}
for label, scorer, level, pgs, ratio in CONFIGS:
    mb = find_max(label, scorer, level, pgs, ratio)
    summary[label] = mb

print("\n" + "=" * 60, flush=True)
print("SUMMARY", flush=True)
for label, mb in summary.items():
    print(f"  {label}: max_stable_batch={mb}", flush=True)
print("DONE", flush=True)
