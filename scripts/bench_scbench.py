"""
SCBench Quality Validation
Run FullKV vs CAKE-Serve on SCBench (English subsets).
Uses GPU 1 (free).

Tasks: scbench_choice_eng, scbench_qa_eng, scbench_mf, scbench_vt
"""
import json, os, sys, time, re, numpy as np

from _cake_constants import MODEL_PATH as MODEL
SCBENCH_DIR = "/home/lixinze/dataset/scbench/datasets/microsoft--SCBench/snapshots/master/data"
OUTPUT_DIR = "results/raw/day14_scbench"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# English tasks only (skip Chinese)
TASKS = ["scbench_choice_eng", "scbench_qa_eng", "scbench_mf", "scbench_vt"]
MAX_SAMPLES = 30

def load_scbench(task):
    path = os.path.join(SCBENCH_DIR, f"{task}.jsonl")
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data[:MAX_SAMPLES]

def format_prompt(sample):
    context = sample.get("context", "")
    # SCBench uses multi_turns format
    turns = sample.get("multi_turns", [])
    if turns:
        # Use first turn for simplicity
        question = turns[0].get("input", "")
        answers = turns[0].get("answer", turns[0].get("answers", ""))
    else:
        question = sample.get("input", sample.get("question", ""))
        answers = sample.get("answers", sample.get("answer", ""))
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:", answers

def compute_metric(pred, answers):
    pred = pred.strip().lower()
    if isinstance(answers, str):
        answers = [answers]
    answers = [a.strip().lower() for a in answers]
    for a in answers:
        if pred == a:
            return 1.0
    # F1
    pt = set(re.findall(r'\w+', pred))
    best = 0.0
    for a in answers:
        at = set(re.findall(r'\w+', a))
        if len(pt) == 0 or len(at) == 0:
            continue
        c = pt & at
        p = len(c)/len(pt)
        r = len(c)/len(at)
        if p + r > 0:
            best = max(best, 2*p*r/(p+r))
    return best

def run():
    from vllm import LLM, SamplingParams
    
    sp = SamplingParams(temperature=0, max_tokens=64)
    
    configs = [
        ("FullKV", "snapkv", "uniform", 1.0),
        ("CAKE_25", "cake", "cake_layer", 0.25),
    ]
    
    all_results = {}
    
    for name, scorer, level, ratio in configs:
        print(f"\n{'='*60}", flush=True)
        print(f"CONFIG: {name}", flush=True)
        print(f"{'='*60}", flush=True)
        
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        
        llm = LLM(model=MODEL, compression_ratio=ratio,
                  compression_scorer=scorer, compression_level=level,
                  max_model_len=32768, gpu_memory_utilization=0.90,
                  tensor_parallel_size=1, max_num_seqs=4)
        
        task_results = {}
        for task in TASKS:
            data = load_scbench(task)
            print(f"\n  [{task}] {len(data)} samples", flush=True)
            scores = []
            for i, s in enumerate(data):
                prompt, answers = format_prompt(s)
                try:
                    t0 = time.time()
                    out = llm.generate([prompt], sp)
                    elapsed = time.time() - t0
                    pred = out[0].outputs[0].text if out[0].outputs else ""
                    score = compute_metric(pred, answers)
                    scores.append(score)
                    if (i+1) % 10 == 0:
                        print(f"    {i+1}/{len(data)} avg={np.mean(scores):.3f}", flush=True)
                except Exception as e:
                    scores.append(0.0)
            avg = np.mean(scores) if scores else 0.0
            task_results[task] = {"avg": float(avg), "n": len(scores)}
            print(f"  → {task}: {avg:.3f}", flush=True)
        
        all_results[name] = task_results
        del llm
        
        with open(os.path.join(OUTPUT_DIR, "scbench_results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)
    
    # Summary
    print("\n\n" + "="*60, flush=True)
    print("SCBENCH SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"{'Task':<25} {'FullKV':>8} {'CAKE_25':>8}", flush=True)
    print("-"*45, flush=True)
    for task in TASKS:
        fk = all_results.get("FullKV", {}).get(task, {}).get("avg", 0)
        ck = all_results.get("CAKE_25", {}).get(task, {}).get("avg", 0)
        print(f"{task:<25} {fk:>8.3f} {ck:>8.3f}", flush=True)
    
    print(f"\nResults: {OUTPUT_DIR}/scbench_results.json", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    run()