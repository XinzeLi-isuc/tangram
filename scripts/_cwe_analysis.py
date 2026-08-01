"""CWE degradation analysis: per-layer budget dump + floor sweep.

Hypothesis: CAKE layer budgets starve some layers below what CWE needs.
Evidence to gather:
  1. Per-layer budget (kept lengths) for CAKE_cake_layer @ 25% on CWE
  2. CWE accuracy vs floor_min ∈ {0, 128, 256, 512, 1024}
  3. CWE accuracy vs requested ratio (uniform vs cake_layer)

Usage:
    CAKE_FLOOR_MIN=256 CUDA_VISIBLE_DEVICES=2 \
        python scripts/_cwe_analysis.py --length 8192 --floor 256
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _cake_constants import MODEL_PATH as MODEL
from _experiment_config import (
    CAKE_WINDOW_SIZE, CAKE_N_SINK_TOKENS, CAKE_FLOOR_MIN,
    CAKE_CHUNK_SIZE, CAKE_PAGE_GROUP_SIZE, GPU_MEMORY_UTILIZATION,
    CAKE_GAMMA, CAKE_TAU1, CAKE_TAU2,
)

OUT_DIR = "results/raw/day23_cwe"
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLES = 50


def load_cwe(length):
    import pyarrow.parquet as pq
    path = f"dataset/ruler/{length}.parquet"
    df = pq.ParquetFile(path).read().to_pandas()
    sub = df[df["task"] == "cwe"].head(N_SAMPLES)
    return sub


def run_cwe(config_name, scorer, level, ratio, samples, length, floor):
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
        compression_floor_min=floor,
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

    correct = 0
    per_sample = []
    for si, (idx, row) in enumerate(samples.iterrows()):
        text = row["context"] + "\n" + row["question"]
        pids = tokenizer.encode(text, add_special_tokens=True)
        prompt = TokensPrompt(prompt_token_ids=pids)
        out = llm.generate([prompt], sp)
        pred = out[0].outputs[0].text.strip()
        answers = list(row["answer"])
        match = any(str(a).lower() in pred.lower() for a in answers)
        correct += int(match)
        per_sample.append({
            "sample_id": int(si), "correct": int(match),
            "prediction": pred, "reference": [str(a) for a in answers],
        })
        if (si + 1) % 10 == 0:
            print(f"      {si+1}/{len(samples)} acc={correct/(si+1):.3f}",
                  flush=True)

    del llm
    import gc; gc.collect()
    import torch; torch.cuda.empty_cache()
    return correct / len(samples), per_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=8192, choices=[4096, 8192, 16384])
    ap.add_argument("--floor", type=int, default=None,
                    help="override CAKE_FLOOR_MIN (0 default)")
    args = ap.parse_args()

    length = args.length
    floor = args.floor if args.floor is not None else CAKE_FLOOR_MIN
    samples = load_cwe(length)
    print(f"CWE analysis @ {length}, floor={floor}, "
          f"gamma={CAKE_GAMMA} tau1={CAKE_TAU1} tau2={CAKE_TAU2}")

    configs = [
        ("FullKV",             "snapkv", "uniform",     1.0),
        ("CAKE25_cake_layer",  "cake",   "cake_layer",  0.25),
        ("CAKE25_uniform",     "cake",   "uniform",     0.25),
        ("CAKE50_cake_layer",  "cake",   "cake_layer",  0.5),
    ]

    results = {}
    for label, scorer, level, ratio in configs:
        print(f"\n  [{label}] floor={floor}", flush=True)
        acc, per_sample = run_cwe(
            label, scorer, level, ratio, samples, length, floor)
        print(f"    CWE acc = {acc:.4f}")
        results[label] = {
            "length": length, "floor": floor, "config": label,
            "scorer": scorer, "level": level, "ratio": ratio,
            "cwe_acc": round(acc, 4),
            "per_sample": per_sample,
        }

    # Save
    out_path = os.path.join(
        OUT_DIR, f"cwe_len{length}_floor{floor}_g{CAKE_GAMMA:g}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("CWE Accuracy Summary")
    print(f"{'='*60}")
    for label, r in results.items():
        print(f"  {label:<20} {r['cwe_acc']:.4f}")


if __name__ == "__main__":
    main()
