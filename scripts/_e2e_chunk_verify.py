"""Compression chunk-size sensitivity experiment — not scheduler split test.

Compares CAKE_25 at compression_chunk_size=2048 vs 8192 (both with
matching max_num_batched_tokens). This measures sensitivity to the
compression interval, NOT scheduler request-level splits.

Phase 1: Verify cake_layer produces non-uniform layer budgets.
Phase 2: Compare layer-level retention patterns across chunk sizes.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/_e2e_chunk_verify.py
"""
import json, os, sys, tempfile, shutil, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _cake_constants import MODEL_PATH as MODEL
from _real_data import build_real_prompt_ids
from _retention_utils import load_final_decisions, summarize_retention
from _experiment_config import (
    CAKE_WINDOW_SIZE, CAKE_N_SINK_TOKENS, CAKE_FLOOR_MIN,
    CAKE_PAGE_GROUP_SIZE, GPU_MEMORY_UTILIZATION,
)

OUTPUT_DIR = "results/raw/day16_e2e"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PROMPT_LEN = 8192


def run_and_collect(chunk_size, label):
    """Run CAKE_25, return (dump_dir, final_records, engine_input_tokens)."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    pids = build_real_prompt_ids(tokenizer, PROMPT_LEN)
    prompt_input = TokensPrompt(prompt_token_ids=pids)
    sp = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)

    dump_dir = os.path.join(tempfile.mkdtemp(), "retention")
    os.makedirs(dump_dir)

    llm = LLM(
        model=MODEL, compression_ratio=0.25,
        compression_scorer="cake", compression_level="cake_layer",
        page_group_size=CAKE_PAGE_GROUP_SIZE,
        compression_window_size=CAKE_WINDOW_SIZE,
        compression_n_sink_tokens=CAKE_N_SINK_TOKENS,
        compression_floor_min=CAKE_FLOOR_MIN,
        compression_chunk_size=chunk_size,
        max_num_batched_tokens=chunk_size,
        compression_retention_dump=dump_dir,
        max_model_len=PROMPT_LEN + 128,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        disable_log_stats=True,
    )

    out = llm.generate([prompt_input], sp)
    engine_tokens = len(out[0].prompt_token_ids)
    assert engine_tokens == PROMPT_LEN, \
        f"Engine saw {engine_tokens}, expected {PROMPT_LEN}"
    num_out = len(out[0].outputs[0].token_ids)
    del llm
    import gc; gc.collect()
    import torch; torch.cuda.empty_cache()

    records = load_final_decisions(dump_dir)
    logical_by_req = {out[0].request_id: engine_tokens}
    summary = summarize_retention(records, logical_by_req)

    print(f"  [{label}] chunk={chunk_size}: {len(records)} dumps, "
          f"phys={summary['effective_physical_ratio']:.4f}, "
          f"output={num_out} tok", flush=True)
    return dump_dir, records, engine_tokens, summary


def analyze_layer_variance(records, logical_tokens):
    """Extract per-layer kept/total using logical capacity as denominator."""
    num_cells = records[0]["kept"].size if records else 1
    kept_by_layer = np.zeros(records[0]["kept"].shape[0] if records else 32,
                              dtype=np.int64)

    for rec in records:
        kept_by_layer += rec["kept"].sum(axis=1)

    # Per-layer end-to-end ratio: kept / (logical_tokens × num_groups_per_layer)
    num_groups = records[0]["kept"].shape[1] if records else 1
    layer_denom = logical_tokens * num_groups
    layer_ratios = kept_by_layer.astype(np.float64) / layer_denom
    mean_ratio = float(layer_ratios.mean())
    std_ratio = float(layer_ratios.std())
    cv = std_ratio / mean_ratio if mean_ratio > 0 else 0

    n_deviant = int(np.sum(np.abs(layer_ratios - mean_ratio) > 0.05 * mean_ratio))

    return {
        "num_layers": len(layer_ratios),
        "mean_ratio": round(mean_ratio, 6),
        "std_ratio": round(std_ratio, 6),
        "cv": round(cv, 4),
        "n_deviant_layers": n_deviant,
        "is_uniform": n_deviant < 2,
        "layer_ratios": [round(float(r), 6) for r in layer_ratios],
        "layer_kept": kept_by_layer.tolist(),
    }


def compare_chunks(a_stats, b_stats):
    """Compare layer budget patterns between two chunk sizes."""
    a_ratios = np.array(a_stats["layer_ratios"])
    b_ratios = np.array(b_stats["layer_ratios"])
    from scipy.stats import spearmanr
    spear, _ = spearmanr(a_ratios, b_ratios)
    a_top5 = set(np.argsort(a_ratios)[-5:])
    b_top5 = set(np.argsort(b_ratios)[-5:])
    mae = float(np.mean(np.abs(a_ratios - b_ratios)))
    return {
        "spearman_r": round(float(spear), 4),
        "top5_overlap": len(a_top5 & b_top5) / 5.0,
        "budget_mae": round(mae, 6),
    }


def main():
    print("Compression chunk-size sensitivity (CAKE_25, 8K)", flush=True)
    print("=" * 60, flush=True)

    results = {}
    all_phys = {}

    for chunk_size in [2048, 8192]:
        print(f"\n--- chunk={chunk_size} ---", flush=True)
        d, records, eng_tok, summary = run_and_collect(chunk_size, "cake_layer")
        stats = analyze_layer_variance(records, eng_tok)
        label = f"chunk_{chunk_size}"
        results[label] = {"summary": summary, "layer_stats": stats}
        all_phys[chunk_size] = summary["effective_physical_ratio"]
        shutil.rmtree(os.path.dirname(d), ignore_errors=True)

        print(f"  phys={summary['effective_physical_ratio']:.4f} "
              f"final-step={summary['final_step_shrink_ratio']:.4f} "
              f"layers={stats['num_layers']} "
              f"CV={stats['cv']:.4f} "
              f"uniform={stats['is_uniform']}", flush=True)
        if not stats["is_uniform"]:
            top3 = np.argsort(stats["layer_ratios"])[-3:].tolist()
            bot3 = np.argsort(stats["layer_ratios"])[:3].tolist()
            print(f"  Top-3 layers: {top3}  Bot-3: {bot3}", flush=True)

    # Comparison
    comparison = compare_chunks(results["chunk_2048"]["layer_stats"],
                                results["chunk_8192"]["layer_stats"])
    results["comparison"] = comparison
    results["phys_ratios"] = all_phys

    is_non_uniform = not results["chunk_2048"]["layer_stats"]["is_uniform"]
    both_close = all(abs(r - 0.25) < 0.03 for r in all_phys.values())

    verdict = {
        "budget_correctness": "PASS" if both_close else "FAIL",
        "layer_adaptivity": "PASS" if is_non_uniform else "FAIL",
        "chunk_size_stability": "PASS" if comparison["budget_mae"] < 0.05 else (
            "UNSTABLE" if comparison["budget_mae"] > 0.1 else "MODERATE"
        ),
    }
    results["verdict"] = verdict

    # Add metadata
    import subprocess
    results["meta"] = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            cwd=os.path.dirname(__file__) + "/.."
        ).strip(),
        "model": MODEL,
        "requested_ratio": 0.25,
        "prompt_length": PROMPT_LEN,
        "compression_window_size": CAKE_WINDOW_SIZE,
        "compression_n_sink_tokens": CAKE_N_SINK_TOKENS,
        "compression_floor_min": CAKE_FLOOR_MIN,
        "page_group_size": CAKE_PAGE_GROUP_SIZE,
        "chunk_sizes_tested": [2048, 8192],
    }

    print(f"\n  Budget correctness: {verdict['budget_correctness']}", flush=True)
    print(f"  Layer adaptivity:  {verdict['layer_adaptivity']}", flush=True)
    print(f"  Chunk stability:   {verdict['chunk_size_stability']}", flush=True)
    print(f"  Spearman={comparison['spearman_r']} top5={comparison['top5_overlap']} "
          f"MAE={comparison['budget_mae']}", flush=True)
    print(f"  Phys: 2048={all_phys[2048]:.4f} 8192={all_phys[8192]:.4f}", flush=True)

    out_path = os.path.join(OUTPUT_DIR, "e2e_verify.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Verdict: {verdict}", flush=True)
    print(f"  Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
