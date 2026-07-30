"""RULER 4K quality pilot — validate pipeline before scaling to 8K/16K.

Loads local parquet (~/dataset/ruler/4096/test-*.parquet), runs CAKE_25
vs FullKV on niah_single_1 + vt tasks (fastest), reports string-match accuracy.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/_ruler_4k_pilot.py
"""
import json, os, sys, time
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _cake_constants import MODEL_PATH as MODEL
from _experiment_config import (
    CAKE_WINDOW_SIZE, CAKE_N_SINK_TOKENS, CAKE_FLOOR_MIN,
    CAKE_CHUNK_SIZE, CAKE_PAGE_GROUP_SIZE, GPU_MEMORY_UTILIZATION,
)

DATASET = os.path.expanduser("~/dataset/ruler/4096/test-00000-of-00001.parquet")
OUTPUT_DIR = "results/raw/day17_ruler_pilot"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MAX_SAMPLES_PER_TASK = 50


def string_match_all(refs, pred):
    """RULER recall: all gold items must appear in prediction."""
    if not refs:
        return 0.0
    return sum(1 for r in refs if r.lower() in pred.lower()) / len(refs)


def string_match_part(refs, pred):
    """RULER QA metric: any gold item present."""
    return 1.0 if any(r.lower() in pred.lower() for r in refs) else 0.0


def evaluate_task(task_name, scorer, level, ratio, samples):
    """Run one task under one config. Returns per-sample accuracy list."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    # Build prompts: context + question + answer_prefix
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts = []
    max_tokens_list = []
    answers = []

    for s in samples:
        text = s["context"] + "\n\n" + s["question"] + "\n" + s["answer_prefix"]
        pids = tokenizer.encode(text, add_special_tokens=True)
        prompts.append(TokensPrompt(prompt_token_ids=pids))
        max_tokens_list.append(min(s["max_new_tokens"], 128))
        answers.append(s["answer"])

    is_qa = task_name.startswith("qa_")
    metric = string_match_part if is_qa else string_match_all

    kwargs = dict(
        model=MODEL, compression_ratio=ratio,
        compression_scorer=scorer, compression_level=level,
        page_group_size=CAKE_PAGE_GROUP_SIZE,
        compression_window_size=CAKE_WINDOW_SIZE,
        compression_n_sink_tokens=CAKE_N_SINK_TOKENS,
        compression_floor_min=CAKE_FLOOR_MIN,
        compression_chunk_size=CAKE_CHUNK_SIZE,
        max_model_len=4096 + 128,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        disable_log_stats=True,
    )
    if ratio >= 1.0:
        del kwargs["compression_ratio"]
        del kwargs["compression_scorer"]
        del kwargs["compression_level"]

    llm = LLM(**kwargs)

    accuracies = []
    t0 = time.time()
    for i, (prompt, max_tok, refs) in enumerate(zip(prompts, max_tokens_list, answers)):
        sp = SamplingParams(temperature=0, max_tokens=max_tok, ignore_eos=True)
        out = llm.generate([prompt], sp)
        pred = out[0].outputs[0].text
        score = metric(refs, pred)
        accuracies.append(score)
        if (i + 1) % 10 == 0:
            print(f"    [{task_name}] {i+1}/{len(samples)} "
                  f"running_acc={np.mean(accuracies):.3f}", flush=True)

    del llm
    import gc; gc.collect()
    import torch; torch.cuda.empty_cache()

    elapsed = time.time() - t0
    avg_acc = np.mean(accuracies)
    print(f"  [{task_name}] {scorer}/{level} r={ratio}: "
          f"acc={avg_acc:.4f} n={len(accuracies)} {elapsed:.0f}s", flush=True)
    return accuracies


def main():
    table = pq.read_table(DATASET).to_pydict()
    print(f"Loaded {len(table['task'])} RULER samples")

    # Pilot: 2 configs × 2 tasks
    configs = [
        ("FullKV", "snapkv", "uniform", 1.0),
        ("CAKE_25", "cake", "cake_layer", 0.25),
    ]
    pilot_tasks = ["niah_single_1", "vt"]  # fastest tasks

    results = {}
    for task_name in pilot_tasks:
        # Filter samples for this task
        indices = [i for i, t in enumerate(table["task"]) if t == task_name]
        indices = indices[:MAX_SAMPLES_PER_TASK]
        samples = [{k: table[k][i] for k in table} for i in indices]
        print(f"\n{'='*50}")
        print(f"Task: {task_name} ({len(samples)} samples)")
        print(f"{'='*50}")

        for label, scorer, level, ratio in configs:
            print(f"\n  {label} (ratio={ratio})")
            accs = evaluate_task(task_name, scorer, level, ratio, samples)
            key = f"{task_name}__{label}"
            results[key] = {
                "task": task_name, "config": label,
                "scorer": scorer, "level": level, "ratio": ratio,
                "n_samples": len(samples),
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "accuracies": [float(a) for a in accs],
            }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'Task':<18} {'Config':<10} {'Acc':>8} {'Std':>8}")
    for r in results.values():
        print(f"{r['task']:<18} {r['config']:<10} "
              f"{r['mean_accuracy']:>8.4f} {r['std_accuracy']:>8.4f}")

    out_path = os.path.join(OUTPUT_DIR, "ruler_4k_pilot.json")
    with open(out_path, "w") as f:
        json.dump({"results": list(results.values())}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
