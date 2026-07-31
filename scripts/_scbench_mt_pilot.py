"""SCBench multi-turn quality pilot: qa_eng + vt, 30 samples.

Evaluates each turn independently (context + question) to measure
whether CAKE compression preserves per-turn answer quality.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/_scbench_mt_pilot.py
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _cake_constants import MODEL_PATH as MODEL
from _experiment_config import (
    CAKE_WINDOW_SIZE, CAKE_N_SINK_TOKENS, CAKE_FLOOR_MIN,
    CAKE_CHUNK_SIZE, CAKE_PAGE_GROUP_SIZE, GPU_MEMORY_UTILIZATION,
)

DATA = os.path.expanduser("~/dataset/scbench/datasets/microsoft--SCBench/snapshots/master/data")
OUT = "results/raw/day18_scbench_pilot"
os.makedirs(OUT, exist_ok=True)
MAX_SAMPLES = 30
MAX_CTX = 16000

CONFIGS = [
    ("FullKV", "snapkv", "uniform", 1.0),
    ("CAKE_25", "cake", "cake_layer", 0.25),
]


def parse_answer(answer):
    """Normalize string or list answer to list of strings."""
    if isinstance(answer, str):
        return [answer.lower()]
    if isinstance(answer, list):
        return [str(a).lower() for a in answer]
    return [str(answer).lower()]


def main():
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    sp = SamplingParams(temperature=0, max_tokens=50, ignore_eos=True)

    all_results = {}

    for task in ["qa_eng", "vt"]:
        print(f"\n{'='*60}")
        print(f"SCBench: {task}")
        print(f"{'='*60}")

        path = os.path.join(DATA, f"scbench_{task}.jsonl")
        samples = []
        with open(path) as f:
            for line in f:
                samples.append(json.loads(line))
                if len(samples) >= MAX_SAMPLES:
                    break

        turns = samples[0].get("multi_turns", [])
        print(f"  {len(samples)} samples, {len(turns)} turns each")

        for label, scorer, level, ratio in CONFIGS:
            print(f"\n  [{label}]", flush=True)

            kwargs = dict(
                model=MODEL, compression_ratio=ratio,
                compression_scorer=scorer, compression_level=level,
                page_group_size=CAKE_PAGE_GROUP_SIZE,
                compression_window_size=CAKE_WINDOW_SIZE,
                compression_n_sink_tokens=CAKE_N_SINK_TOKENS,
                compression_floor_min=CAKE_FLOOR_MIN,
                compression_chunk_size=CAKE_CHUNK_SIZE,
                max_model_len=MAX_CTX + 512,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                disable_log_stats=True,
            )
            if ratio >= 1.0:
                for k in ["compression_ratio", "compression_scorer",
                           "compression_level"]:
                    kwargs.pop(k, None)

            llm = LLM(**kwargs)
            turn_accs = {}  # ti → list of scores

            for si, s in enumerate(samples):
                ctx = (s.get("context") or s.get("input", ""))
                if len(ctx) > MAX_CTX:
                    ctx = ctx[:MAX_CTX]

                for ti, turn in enumerate(s.get("multi_turns", [])):
                    if ti not in turn_accs:
                        turn_accs[ti] = []
                    text = ctx + "\n" + turn["input"]
                    pids = tokenizer.encode(text, add_special_tokens=True)
                    prompt = TokensPrompt(prompt_token_ids=pids)

                    out = llm.generate([prompt], sp)
                    pred = out[0].outputs[0].text.strip()
                    refs = parse_answer(turn.get("answer", []))

                    match = 1.0 if any(r in pred.lower() for r in refs) else 0.0
                    turn_accs[ti].append(match)

                if (si + 1) % 10 == 0:
                    hits = [np.mean(v) for v in turn_accs.values()]
                    print(f"    {si+1}/{len(samples)} avg_acc={np.mean(hits):.3f}", flush=True)

            del llm
            import gc; gc.collect()
            import torch; torch.cuda.empty_cache()

            max_t = max(turn_accs.keys()) + 1
            taccs = [np.mean(turn_accs[t]) if t in turn_accs else 0.0 for t in range(max_t)]
            macro = float(np.mean([x for x in taccs if x > 0]))

            print(f"    Turns:", " ".join(f"T{t+1}={a:.3f}" for t, a in enumerate(taccs)))
            print(f"    Macro avg: {macro:.4f}")

            all_results[f"{task}_{label}"] = {
                "task": task, "config": label,
                "turn_accuracies": taccs,
                "macro_avg": round(macro, 4),
            }

    # Summary
    print(f"\n{'='*60}")
    print("SCBench Multi-Turn Summary")
    print(f"{'Task':<10} {'Config':<10} {'T1':>6} {'T2':>6} {'T3':>6} {'T4':>6} {'T5':>6} {'Avg':>8}")
    print("-" * 60)
    for task in ["qa_eng", "vt"]:
        for label in ["FullKV", "CAKE_25"]:
            r = all_results[f"{task}_{label}"]
            t = r["turn_accuracies"]
            print(f"{task:<10} {label:<10} "
                  f"{t[0]:>6.3f} {t[1]:>6.3f} {t[2]:>6.3f} "
                  f"{t[3]:>6.3f} {t[4]:>6.3f} {r['macro_avg']:>8.4f}")

    with open(os.path.join(OUT, "scbench_mt_pilot.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {OUT}/scbench_mt_pilot.json")


if __name__ == "__main__":
    main()
