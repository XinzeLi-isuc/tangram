"""RULER 5-way ablation quality benchmark.

Configs:
  1. FullKV (baseline)
  2. SnapKV + uniform
  3. SnapKV + crosslayer_cluster
  4. CAKE + uniform
  5. CAKE + cake_layer

Lengths: 4096, 8192, 16384 (auto-selected by --length)
Samples: 50 per task per config (650 total per config)

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/_ruler_5way.py --length 4096
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _cake_constants import MODEL_PATH as MODEL
from _experiment_config import (
    CAKE_WINDOW_SIZE, CAKE_N_SINK_TOKENS, CAKE_FLOOR_MIN,
    CAKE_CHUNK_SIZE, CAKE_PAGE_GROUP_SIZE, GPU_MEMORY_UTILIZATION,
    CAKE_GAMMA, CAKE_TAU1, CAKE_TAU2,
)

SAMPLES_PER_TASK = 50
SEED = 42

# 5-way ablation configs
ABLATIONS = [
    ("FullKV",           "snapkv", "uniform",          1.0),
    ("SnapKV_uniform",   "snapkv", "uniform",          0.25),
    ("SnapKV_cluster",   "snapkv", "crosslayer_cluster", 0.25),
    ("CAKE_uniform",     "cake",   "uniform",          0.25),
    ("CAKE_cake_layer",  "cake",   "cake_layer",       0.25),
]

# Task list
TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "qa_1", "qa_2", "vt", "cwe", "fwe",
]


def load_ruler(length):
    """Load RULER parquet and return dict task -> list of samples."""
    import pyarrow.parquet as pq
    path = f"dataset/ruler/{length}.parquet"
    df = pq.ParquetFile(path).read().to_pandas()
    rng = np.random.default_rng(SEED)
    task_samples = {}
    for task in TASKS:
        sub = df[df["task"] == task].copy()
        if len(sub) < SAMPLES_PER_TASK:
            raise RuntimeError(f"Task {task} has {len(sub)} samples, need {SAMPLES_PER_TASK}")
        # Sample deterministically
        indices = rng.choice(len(sub), SAMPLES_PER_TASK, replace=False)
        task_samples[task] = sub.iloc[indices]
    return task_samples


def evaluate_config(config_name, scorer, level, ratio, task_samples, length):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    sp = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)

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
        compression_cake_gamma=CAKE_GAMMA,
        compression_cake_tau1=CAKE_TAU1,
        compression_cake_tau2=CAKE_TAU2,
        max_model_len=length + 256,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        disable_log_stats=True,
    )
    if ratio >= 1.0:
        for k in ["compression_ratio", "compression_scorer", "compression_level"]:
            kwargs.pop(k, None)

    llm = LLM(**kwargs)

    results = {}
    for task in TASKS:
        samples = task_samples[task]
        correct = 0
        for idx, row in samples.iterrows():
            text = row["context"] + "\n" + row["question"]
            pids = tokenizer.encode(text, add_special_tokens=True)
            prompt = TokensPrompt(prompt_token_ids=pids)
            out = llm.generate([prompt], sp)
            pred = out[0].outputs[0].text.strip()

            answers = list(row["answer"])
            match = any(str(a).lower() in pred.lower() for a in answers)
            correct += int(match)

        acc = correct / SAMPLES_PER_TASK
        results[task] = round(acc, 4)
        print(f"    {task}: {acc:.4f}")

    del llm
    import gc; gc.collect()
    import torch; torch.cuda.empty_cache()
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=4096, choices=[4096, 8192, 16384])
    ap.add_argument("--config_idx", type=int, default=None, help="Run single config (0-4)")
    args = ap.parse_args()

    length = args.length
    print(f"RULER 5-way ablation @ {length} tokens")

    task_samples = load_ruler(length)

    configs = ABLATIONS if args.config_idx is None else [ABLATIONS[args.config_idx]]

    all_results = {}
    for label, scorer, level, ratio in configs:
        print(f"\n{'='*60}")
        print(f"Config: {label} (scorer={scorer}, level={level}, ratio={ratio})")
        print(f"{'='*60}")

        t0 = time.time()
        task_accs = evaluate_config(label, scorer, level, ratio, task_samples, length)
        elapsed = time.time() - t0

        macro_avg = float(np.mean(list(task_accs.values())))
        print(f"  Macro avg: {macro_avg:.4f} ({elapsed/60:.1f} min)")

        all_results[label] = {
            "length": length,
            "config": label,
            "scorer": scorer,
            "level": level,
            "ratio": ratio,
            "task_accuracies": task_accs,
            "macro_avg": round(macro_avg, 4),
            "elapsed_sec": round(elapsed, 1),
        }

    # Summary
    print(f"\n{'='*70}")
    print("Summary: Macro Average Accuracy")
    print(f"{'='*70}")
    for label, r in all_results.items():
        print(f"  {label:<20} {r['macro_avg']:.4f}")

    out_dir = f"results/raw/day20_ruler_{length}"
    # Paper-params (non-default gamma/tau) go to a separate directory
    # with the hyperparams in the filename (gamma ablation writes 1/50/200)
    if CAKE_GAMMA != 1.0 or CAKE_TAU1 != 1.0 or CAKE_TAU2 != 1.0:
        out_dir = f"results/raw/day21_paper_params/{length}"
    os.makedirs(out_dir, exist_ok=True)

    hyper_suffix = ""
    if CAKE_GAMMA != 1.0 or CAKE_TAU1 != 1.0 or CAKE_TAU2 != 1.0:
        hyper_suffix = f"_g{CAKE_GAMMA:g}_t1{CAKE_TAU1:g}_t2{CAKE_TAU2:g}"

    for label, r in all_results.items():
        safe = label.replace(" ", "_")
        out_path = os.path.join(out_dir, f"{safe}{hyper_suffix}.json")
        with open(out_path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"Saved: {out_path}")

    # Also save combined if all configs run
    if args.config_idx is None:
        with open(os.path.join(out_dir, "ruler_5way.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Combined: {out_dir}/ruler_5way.json")


if __name__ == "__main__":
    main()
