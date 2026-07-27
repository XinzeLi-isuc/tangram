"""
LongBench Quality Validation
============================
Test FullKV vs CAKE-Serve on real-world LongBench tasks.
Tasks: NarrativeQA, Qasper, MultiFieldQA-en, HotpotQA, 2WikiMultihopQA

Usage:
    conda activate cake-serve
    cd ~/cake-serve
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_longbench.py 2>&1
"""
import json, os, sys, time, re
import numpy as np

from _cake_constants import MODEL_PATH as MODEL
LONGBENCH_DIR = "/home/lixinze/dataset/longbench/data"
OUTPUT_DIR = "results/raw/day14_longbench"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tasks from the plan: 5 English long-context QA tasks
TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa"]
MAX_SAMPLES = 50  # samples per task


def load_longbench(task):
    """Load LongBench data for a task."""
    path = os.path.join(LONGBENCH_DIR, f"{task}.jsonl")
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data[:MAX_SAMPLES]


def format_prompt(sample):
    """Format LongBench sample into a prompt."""
    # LongBench has 'context', 'input', 'answers', 'length', 'all_classes'
    context = sample.get("context", "")
    question = sample.get("input", "")
    prompt = f"Read the following text and answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return prompt


def compute_metric(prediction, answers):
    """Compute F1 score for a prediction against ground truth answers."""
    prediction = prediction.strip().lower()
    answers = [a.strip().lower() for a in answers]
    
    # Exact match check first
    for a in answers:
        if prediction == a:
            return 1.0
    
    # F1 score
    pred_tokens = set(re.findall(r'\w+', prediction))
    max_f1 = 0.0
    for a in answers:
        ans_tokens = set(re.findall(r'\w+', a))
        if len(pred_tokens) == 0 or len(ans_tokens) == 0:
            continue
        common = pred_tokens & ans_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ans_tokens)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            max_f1 = max(max_f1, f1)
    return max_f1


def run_task(task, llm, sampling_params, config_name):
    """Run a single LongBench task."""
    print(f"\n  [{config_name}] Task: {task}", flush=True)
    data = load_longbench(task)
    print(f"    Samples: {len(data)}", flush=True)
    
    scores = []
    for i, sample in enumerate(data):
        prompt = format_prompt(sample)
        try:
            t0 = time.time()
            outputs = llm.generate([prompt], sampling_params)
            elapsed = time.time() - t0
            pred = outputs[0].outputs[0].text if outputs[0].outputs else ""
            answers = sample.get("answers", [])
            score = compute_metric(pred, answers)
            scores.append(score)
            if (i + 1) % 10 == 0:
                print(f"      {i+1}/{len(data)} done, avg={np.mean(scores):.3f}, "
                      f"last={score:.3f}, {elapsed:.1f}s", flush=True)
        except Exception as e:
            print(f"      Sample {i} FAILED: {str(e)[:100]}", flush=True)
            scores.append(0.0)
    
    avg = np.mean(scores) if scores else 0.0
    print(f"    Result: {task} = {avg:.3f}", flush=True)
    return float(avg), scores


def main():
    from vllm import LLM, SamplingParams
    
    configs = [
        ("FullKV", "snapkv", "uniform", 1.0),
        ("CAKE_25", "cake", "cake_layer", 0.25),
        ("CAKE_50", "cake", "cake_layer", 0.5),
    ]
    
    sp = SamplingParams(temperature=0, max_tokens=64)
    all_results = {}
    
    for name, scorer, level, ratio in configs:
        print(f"\n{'='*60}", flush=True)
        print(f"CONFIG: {name} (scorer={scorer}, level={level}, ratio={ratio})", flush=True)
        print(f"{'='*60}", flush=True)
        
        llm = LLM(model=MODEL, compression_ratio=ratio,
                  compression_scorer=scorer, compression_level=level,
                  max_model_len=32768, gpu_memory_utilization=0.90,
                  tensor_parallel_size=1, max_num_seqs=4)
        
        task_results = {}
        for task in TASKS:
            avg, scores = run_task(task, llm, sp, name)
            task_results[task] = {"avg_score": avg, "scores": scores[:10]}  # save first 10 only
        
        all_results[name] = task_results
        del llm
        
        # Save intermediate
        with open(os.path.join(OUTPUT_DIR, "longbench_results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)
    
    # Summary
    print("\n\n" + "="*60, flush=True)
    print("LONGBENCH SUMMARY", flush=True)
    print("="*60, flush=True)
    header = f"{'Task':<20}" + "".join(f"{c[0]:<12}" for c in configs)
    print(header, flush=True)
    print("-" * (20 + 12 * len(configs)), flush=True)
    for task in TASKS:
        row = f"{task:<20}"
        for name, _, _, _ in configs:
            score = all_results.get(name, {}).get(task, {}).get("avg_score", 0)
            row += f"{score:<12.3f}"
        print(row, flush=True)
    
    print(f"\nResults saved to {OUTPUT_DIR}/longbench_results.json", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()