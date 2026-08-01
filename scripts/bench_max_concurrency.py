"""Max stable concurrency benchmark (v2: real preemption detection).

Each concurrency level runs in a fresh subprocess (fresh engine, no
CUDA allocator state leakage). Reads the real vLLM scheduler counter
`vllm:num_preemptions` via llm.llm_engine.get_metrics(). A level
passes ONLY if: zero preemptions, zero failed requests, all outputs
non-empty, and no OOM.

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

CONFIGS = [
    ("FullKV",  "snapkv", "uniform",    1.0),
    ("CAKE_25", "cake",   "cake_layer", 0.25),
    ("CAKE_50", "cake",   "cake_layer", 0.50),
]

CONCURRENCY_LEVELS = [1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32]

MAX_NEW_TOKENS = 64
OUTPUT_MIN_CHARS = 5


def run_level(config_name, scorer, level, ratio, seq_len, concurrency, out_path):
    """Run one concurrency level in THIS process (called via subprocess)."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    from _real_data import build_real_prompt_ids

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    pids = build_real_prompt_ids(tokenizer, seq_len)

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
        max_model_len=seq_len + MAX_NEW_TOKENS + 8,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        disable_log_stats=False,  # needed for get_metrics()
    )
    if ratio >= 1.0:
        for k in ["compression_ratio", "compression_scorer", "compression_level"]:
            kwargs.pop(k, None)

    result = {
        "config": config_name, "concurrency": concurrency,
        "seq_len": seq_len, "status": "fail", "reason": None,
        "elapsed_sec": None, "per_req_sec": None,
        "num_preemptions": None, "num_failed": None, "all_outputs_valid": None,
        "peak_running_reqs": None,
        "peak_kv_cache_usage_perc": None,
        "min_free_kv_perc": None,
    }

    llm = None
    try:
        llm = LLM(**kwargs)
        sp = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS,
                            ignore_eos=True)
        prompts = [TokensPrompt(prompt_token_ids=pids) for _ in range(concurrency)]

        # Sample num_requests_running + kv_cache_usage gauges in bg thread
        import threading
        peak_running = {"value": 0}
        peak_kv_usage = {"value": 0.0}
        min_free_kv = {"value": 1.0}
        stop_flag = {"value": False}

        def sampler():
            while not stop_flag["value"]:
                try:
                    metrics = llm.llm_engine.get_metrics()
                    for m in metrics:
                        if m.name == "vllm:num_requests_running":
                            peak_running["value"] = max(peak_running["value"],
                                                        m.value)
                        elif m.name == "vllm:kv_cache_usage_perc":
                            v = float(m.value)
                            peak_kv_usage["value"] = max(peak_kv_usage["value"], v)
                            min_free_kv["value"] = min(min_free_kv["value"], 1.0 - v)
                except Exception:
                    pass
                time.sleep(0.1)

        t_sampler = threading.Thread(target=sampler, daemon=True)
        t_sampler.start()

        t0 = time.time()
        outs = llm.generate(prompts, sp)
        elapsed = time.time() - t0
        stop_flag["value"] = True
        t_sampler.join(timeout=2)

        result["elapsed_sec"] = round(elapsed, 2)
        result["per_req_sec"] = round(elapsed / concurrency, 2)
        result["peak_running_reqs"] = peak_running["value"]
        result["peak_kv_cache_usage_perc"] = round(peak_kv_usage["value"], 4)
        result["min_free_kv_perc"] = round(min_free_kv["value"], 4)

        # Output validity
        valid = all(len(o.outputs[0].text.strip()) >= OUTPUT_MIN_CHARS
                    for o in outs)
        result["all_outputs_valid"] = bool(valid)
        result["num_failed"] = int(not valid)

        # Real preemption counter
        try:
            metrics = llm.llm_engine.get_metrics()
            preemptions = next(
                (m.value for m in metrics if m.name == "vllm:num_preemptions"),
                0,
            )
            result["num_preemptions"] = int(preemptions)
        except Exception as e:
            result["reason"] = f"metrics_error: {e}"
        if result["num_preemptions"] == 0 and valid:
            result["status"] = "success"

    except Exception as e:
        result["reason"] = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        if llm is not None:
            try:
                del llm
            except Exception:
                pass
        import gc; gc.collect()
        import torch
        torch.cuda.empty_cache()

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))
    return result


def main():
    import argparse
    import subprocess
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=32768, choices=[8192, 16384, 32768])
    ap.add_argument("--config_idx", type=int, default=None)
    ap.add_argument("--level", type=int, default=None, help="internal: run single level")
    args = ap.parse_args()

    seq_len = args.length

    # Internal mode: single level in subprocess
    if args.level is not None:
        cfg = CONFIGS[args.config_idx]
        out_dir = f"results/raw/day21_concurrency/{seq_len}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{cfg[0]}_c{args.level}.json")
        run_level(cfg[0], cfg[1], cfg[2], cfg[3], seq_len, args.level, out_path)
        return

    PY = sys.executable
    os.makedirs(f"results/raw/day21_concurrency/{seq_len}", exist_ok=True)

    all_results = {}
    for ci, (label, scorer, level, ratio) in enumerate(CONFIGS):
        print(f"\n{'='*60}\n{label}\n{'='*60}")
        max_ok = 0
        levels = []
        saw_fail = False

        for conc in CONCURRENCY_LEVELS:
            print(f"  conc={conc}...", flush=True)
            out_path = os.path.join(
                f"results/raw/day21_concurrency/{seq_len}", f"{label}_c{conc}.json")
            # Fresh subprocess per level
            proc = subprocess.run(
                [PY, __file__, "--length", str(seq_len),
                 "--config_idx", str(ci), "--level", str(conc)],
                capture_output=True, text=True, timeout=1800,
            )
            if proc.returncode != 0:
                r = {"status": "fail", "reason": proc.stderr[-300:],
                     "concurrency": conc}
            else:
                try:
                    r = json.loads(proc.stdout.strip().splitlines()[-1])
                except Exception:
                    r = {"status": "fail", "reason": "parse_error",
                         "concurrency": conc}

            levels.append(r)
            if r.get("status") == "success":
                max_ok = conc
                print(f"    OK: {r['elapsed_sec']}s, "
                      f"preemptions={r['num_preemptions']}")
            else:
                print(f"    FAIL: {r.get('reason') or r.get('status')}")
                saw_fail = True
                break  # stop at first failure level

        all_results[label] = {
            "max_stable_no_preemption": max_ok,
            "levels": levels,
            "seq_len": seq_len,
        }
        print(f"  MAX={max_ok}")

    out_path = f"results/raw/day21_concurrency/concurrency_{seq_len}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
