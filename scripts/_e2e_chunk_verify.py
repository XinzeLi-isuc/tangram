"""Step 3: GPU E2E — verify cake_layer produces non-uniform layer budgets
and that chunk-split approximation doesn't break scoring.

1. Run CAKE_25 with max_num_batched_tokens=2048 (baseline chunk).
2. Analyze retention dump: per-layer variance, budget spread.
3. Compare with max_num_batched_tokens=8192 (larger chunk, fewer splits).
4. Report Spearman, top-k overlap, budget MAE between chunk sizes.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/_e2e_chunk_verify.py
"""
import json, os, sys, tempfile, shutil, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _cake_constants import MODEL_PATH as MODEL
from _real_data import build_real_prompt_ids
from _experiment_config import (
    CAKE_WINDOW_SIZE, CAKE_N_SINK_TOKENS, CAKE_FLOOR_MIN,
    CAKE_CHUNK_SIZE, CAKE_PAGE_GROUP_SIZE, GPU_MEMORY_UTILIZATION,
)

OUTPUT_DIR = "results/raw/day16_e2e"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PROMPT_LEN = 8192


def run_and_collect(chunk_size, label):
    """Run one CAKE_25 generation, collect retention dump .npz files."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    pids = build_real_prompt_ids(tokenizer, PROMPT_LEN)
    prompt = tokenizer.decode(pids)
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
        compression_chunk_size=chunk_size,       # MUST match max_num_batched_tokens
        max_num_batched_tokens=chunk_size,       # vLLM constraint when compression is on
        compression_retention_dump=dump_dir,
        max_model_len=PROMPT_LEN + 128,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        disable_log_stats=True,
    )

    out = llm.generate([prompt], sp)
    num_out = len(out[0].outputs[0].token_ids)
    del llm
    import gc; gc.collect()
    import torch; torch.cuda.empty_cache()

    npz_files = sorted(f for f in os.listdir(dump_dir) if f.endswith(".npz"))
    print(f"  [{label}] chunk={chunk_size}: {len(npz_files)} dumps, "
          f"output={num_out} tok", flush=True)
    return dump_dir, npz_files


def analyze_layer_variance(dump_dir, npz_files):
    """Extract per-layer kept/total from retention dumps, compute variance."""
    kept_by_layer = None
    total_by_layer = None
    seen = 0

    for fn in npz_files:
        d = np.load(os.path.join(dump_dir, fn), allow_pickle=False)
        kept = d["kept"].astype(np.int64)    # shape: (num_layers, heads_or_groups)
        total = d["total"].astype(np.int64)

        if kept_by_layer is None:
            kept_by_layer = kept.sum(axis=1)
            total_by_layer = total.sum(axis=1)
        else:
            kept_by_layer += kept.sum(axis=1)
            total_by_layer += total.sum(axis=1)
        seen += 1

    # Per-layer ratio
    eps = 1e-9
    layer_ratios = kept_by_layer / (total_by_layer + eps)
    mean_ratio = kept_by_layer.sum() / (total_by_layer.sum() + eps)
    std_ratio = np.std(layer_ratios)
    cv = std_ratio / (mean_ratio + eps)

    # How many layers deviate > 5% from mean?
    n_deviant = int(np.sum(np.abs(layer_ratios - mean_ratio) > 0.05 * mean_ratio))
    is_uniform = n_deviant < 2  # fewer than 2 layers deviate → effectively uniform

    return {
        "num_dumps": seen,
        "num_layers": len(layer_ratios),
        "mean_ratio": float(mean_ratio),
        "std_ratio": float(std_ratio),
        "cv": float(cv),
        "n_deviant_layers": n_deviant,
        "is_uniform": is_uniform,
        "layer_ratios": layer_ratios.tolist(),
        "layer_kept": kept_by_layer.tolist(),
        "layer_total": total_by_layer.tolist(),
    }


def compare_chunks(a_stats, b_stats):
    """Compare retention patterns between two chunk sizes."""
    a_ratios = np.array(a_stats["layer_ratios"])
    b_ratios = np.array(b_stats["layer_ratios"])

    # Spearman rank correlation
    from scipy.stats import spearmanr
    spear, _ = spearmanr(a_ratios, b_ratios)

    # Top-5 layer overlap
    a_top5 = set(np.argsort(a_ratios)[-5:])
    b_top5 = set(np.argsort(b_ratios)[-5:])
    top5_overlap = len(a_top5 & b_top5) / 5.0

    # Budget MAE
    mae = float(np.mean(np.abs(a_ratios - b_ratios)))

    return {
        "spearman_r": round(float(spear), 4),
        "top5_overlap": top5_overlap,
        "budget_mae": round(mae, 6),
    }


def main():
    print("GPU E2E: cake_layer budget verification + chunk-split validation",
          flush=True)
    print("=" * 60, flush=True)

    results = {}

    # Phase 1: Single run at chunk=2048 to verify non-uniformity
    print("\n--- Phase 1: Budget non-uniformity (chunk=2048) ---", flush=True)
    d1, f1 = run_and_collect(2048, "cake_layer")
    stats = analyze_layer_variance(d1, f1)
    results["budget_analysis"] = stats
    shutil.rmtree(os.path.dirname(d1), ignore_errors=True)

    print(f"  Layers: {stats['num_layers']}", flush=True)
    print(f"  Mean ratio: {stats['mean_ratio']:.4f}", flush=True)
    print(f"  Std ratio:  {stats['std_ratio']:.4f}", flush=True)
    print(f"  CV:         {stats['cv']:.4f}", flush=True)
    print(f"  Deviant layers (>5% from mean): {stats['n_deviant_layers']}",
          flush=True)

    if stats["is_uniform"]:
        print("  ⚠️  WARNING: Budget appears UNIFORM — cake_layer may not be "
              "producing layer-adaptive budgets!", flush=True)
    else:
        top3 = np.argsort(stats["layer_ratios"])[-3:].tolist()
        bot3 = np.argsort(stats["layer_ratios"])[:3].tolist()
        print(f"  Top-3 budget layers: {top3}", flush=True)
        print(f"  Bot-3 budget layers: {bot3}", flush=True)

    # Phase 2: Compare chunk=8192 vs chunk=2048
    print("\n--- Phase 2: Chunk-split fidelity (2048 vs 8192) ---", flush=True)
    d_small, f_small = run_and_collect(2048, "small")
    stats_small = analyze_layer_variance(d_small, f_small)
    shutil.rmtree(os.path.dirname(d_small), ignore_errors=True)

    d_large, f_large = run_and_collect(8192, "large")
    stats_large = analyze_layer_variance(d_large, f_large)
    shutil.rmtree(os.path.dirname(d_large), ignore_errors=True)

    comparison = compare_chunks(stats_small, stats_large)
    results["chunk_comparison"] = {
        "small_chunk_2048": stats_small,
        "large_chunk_8192": stats_large,
        "comparison": comparison,
    }

    print(f"  Spearman r: {comparison['spearman_r']}", flush=True)
    print(f"  Top-5 overlap: {comparison['top5_overlap']}", flush=True)
    print(f"  Budget MAE: {comparison['budget_mae']}", flush=True)
    print(f"  Small chunk CV: {stats_small['cv']:.4f}", flush=True)
    print(f"  Large chunk CV: {stats_large['cv']:.4f}", flush=True)

    verdict = "PASS" if (
        not stats["is_uniform"]
        and comparison["spearman_r"] > 0.8
        and comparison["budget_mae"] < 0.1
    ) else "NEEDS_INVESTIGATION"
    results["verdict"] = verdict

    out_path = os.path.join(OUTPUT_DIR, "e2e_verify.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Verdict: {verdict}", flush=True)
    print(f"  Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
