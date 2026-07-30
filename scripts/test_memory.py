"""
Retention Verification: measure actual KV token retention via
compression_retention_dump. Reads saved .npz keep-decision files
to compute effective_ratio = kept_tokens / total_tokens.

Usage:
    cd ~/cake-serve
    CUDA_VISIBLE_DEVICES=0 python scripts/test_memory.py
"""
import json, os, shutil, subprocess, sys, tempfile, time
from datetime import datetime, timezone

import numpy as np

from _cake_constants import MODEL_PATH
from _real_data import build_real_prompt_ids
from _experiment_config import (
    CAKE_WINDOW_SIZE, CAKE_N_SINK_TOKENS, CAKE_FLOOR_MIN,
    CAKE_CHUNK_SIZE, CAKE_PAGE_GROUP_SIZE,
    RETENTION_PROMPT_LENGTH, MAX_OUTPUT_TOKENS, MAX_MODEL_LEN,
    GPU_MEMORY_UTILIZATION,
)

OUTPUT_DIR = "results/raw/day10_memory"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIGS = [
    ("FullKV", "snapkv", "uniform", 1.0),
    ("CAKE_25", "cake", "cake_layer", 0.25),
    ("CAKE_50", "cake", "cake_layer", 0.5),
]

LENGTHS_DEFAULT = [8192, 16384]  # default: test at both 8K and 16K

# All compression parameters from _experiment_config
COMPRESSION_WINDOW_SIZE = CAKE_WINDOW_SIZE
COMPRESSION_N_SINK_TOKENS = CAKE_N_SINK_TOKENS
COMPRESSION_FLOOR_MIN = CAKE_FLOOR_MIN
COMPRESSION_CHUNK_SIZE = CAKE_CHUNK_SIZE
PAGE_GROUP_SIZE = CAKE_PAGE_GROUP_SIZE


def _get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=os.path.dirname(__file__) + "/..",
        ).strip()
    except Exception:
        return "unknown"


def _get_gpu_name():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip()
        return out.split("\n")[0] if out else "unknown"
    except Exception:
        return "unknown"


def _parse_retention_dump(dump_dir: str, requested_ratio: float):
    """Parse retention dump .npz files and compute effective ratios.

    Schema (from vllm/v1/attention/compression/profiling.py):
        kept, total, sink, win, eval_len, req, rank

    Filename format: {req_id}_r{rank}_{seq}.npz where seq increments
    per decision within the same rank. Only the highest-seq record per
    (req, rank) contributes — earlier decisions are intermediate.

    Returns (physical_ratio, context_ratio, total_kept, total_seen, num_dumps).
    """
    REQUIRED = {"kept", "total", "sink", "win", "eval_len", "req", "rank"}

    npz_files = sorted(
        f for f in os.listdir(dump_dir) if f.endswith(".npz"))

    if not npz_files:
        if requested_ratio < 1.0:
            raise RuntimeError(
                f"ratio={requested_ratio} < 1 but no retention dump .npz files "
                f"found in {dump_dir} — compression may not have triggered. "
                "Check max_model_len, prompt length, and compression config."
            )
        # FullKV (ratio=1.0): no compression dumps is expected
        return 1.0, 1.0, 0, 0, 0

    # Deduplicate by (req, rank): pick record with highest sequence number.
    # Filename: {req_id}_r{rank}_{seq}.npz — seq comes from Profiler._seq counter.
    # Multiple decisions at same eval_len are possible (multi-chunk prefill);
    # only the highest seq is the final decision for that (req, rank).
    import re
    _SEQ_RE = re.compile(r"_r\d+_(\d+)\.npz$")

    by_key: dict = {}
    for fn in npz_files:
        path = os.path.join(dump_dir, fn)
        d = np.load(path, allow_pickle=False)

        missing = REQUIRED - set(d.files)
        if missing:
            raise RuntimeError(
                f"Invalid retention dump {fn}: missing fields {sorted(missing)}. "
                f"Found: {sorted(d.files)}"
            )

        key = (str(d["req"]), int(d["rank"]))
        m = _SEQ_RE.search(fn)
        seq = int(m.group(1)) if m else -1
        if key not in by_key or seq > by_key[key]["seq"]:
            by_key[key] = {
                "kept": d["kept"].astype(np.int64),
                "total": d["total"].astype(np.int64),
                "sink": int(d["sink"]),
                "win": int(d["win"]),
                "eval_len": int(d["eval_len"]),
                "seq": seq,
            }

    # Aggregate across all deduplicated (req, rank) records
    kept_all = np.concatenate([rec["kept"].ravel() for rec in by_key.values()])
    total_all = np.concatenate([rec["total"].ravel() for rec in by_key.values()])
    physical_ratio = float(kept_all.sum() / total_all.sum()) if total_all.sum() > 0 else 1.0

    # Context-only ratio: exclude always-kept sink + recent window
    ctx_kept_parts = []
    ctx_total_parts = []
    for rec in by_key.values():
        k = rec["kept"].ravel()
        t = rec["total"].ravel()
        sink = rec["sink"]
        win = rec["win"]
        ctx_k = np.clip(k - sink - win, 0, None)
        ctx_t = np.maximum(t - sink, 1)
        ctx_kept_parts.append(ctx_k)
        ctx_total_parts.append(ctx_t)
    context_kept = np.concatenate(ctx_kept_parts)
    context_total = np.concatenate(ctx_total_parts)
    context_ratio = float(context_kept.sum() / context_total.sum())

    return (
        physical_ratio,
        context_ratio,
        int(kept_all.sum()),
        int(total_all.sum()),
        len(by_key),
    )


def run_one_length(prompt_length, output_dir, bytes_per_token, dtype_name,
                  num_layers, num_kv_heads, head_dim):
    """Run retention verification for a single prompt length."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt_ids = build_real_prompt_ids(tokenizer, prompt_length)
    prompt = tokenizer.decode(prompt_ids)
    actual_tokens = len(prompt_ids)
    estimated_full_kv_gib = bytes_per_token * actual_tokens / (1024 ** 3)
    model_len = prompt_length + MAX_OUTPUT_TOKENS

    print(f"\n  Model: L={num_layers} H_kv={num_kv_heads} D={head_dim} "
          f"dtype={dtype_name} page_group_size={PAGE_GROUP_SIZE}")
    print(f"  Prompt: {actual_tokens} tokens → estimated full KV ≈ "
          f"{estimated_full_kv_gib:.3f} GiB")
    sys.stdout.flush()

    sp = SamplingParams(temperature=0, max_tokens=MAX_OUTPUT_TOKENS,
                        ignore_eos=True)
    results = []

    for label, scorer, level, ratio in CONFIGS:
        print(f"\n  {'='*50}")
        print(f"  {label}  ratio={ratio}  len={prompt_length}")
        print(f"  {'='*50}")
        sys.stdout.flush()

        dump_dir = os.path.join(tempfile.mkdtemp(), "retention")
        os.makedirs(dump_dir)
        success = False

        try:
            llm = LLM(
                model=MODEL_PATH,
                compression_ratio=ratio,
                compression_scorer=scorer,
                compression_level=level,
                page_group_size=PAGE_GROUP_SIZE,
                compression_window_size=COMPRESSION_WINDOW_SIZE,
                compression_n_sink_tokens=COMPRESSION_N_SINK_TOKENS,
                compression_floor_min=COMPRESSION_FLOOR_MIN,
                compression_chunk_size=COMPRESSION_CHUNK_SIZE,
                compression_retention_dump=dump_dir,
                max_model_len=model_len,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                disable_log_stats=True,
            )

            t0 = time.time()
            out = llm.generate([prompt], sp)
            elapsed = time.time() - t0
            num_out = len(out[0].outputs[0].token_ids)
            del llm

            phys_r, ctx_r, tot_kept, tot_seen, n_dumps = _parse_retention_dump(
                dump_dir, ratio)

            estimated_retained_kv_gib = estimated_full_kv_gib * phys_r
            estimated_saved_kv_gib = estimated_full_kv_gib - estimated_retained_kv_gib

            results.append({
                "config": label,
                "ratio": ratio,
                "input_tokens": actual_tokens,
                "output_tokens": num_out,
                "time_s": round(elapsed, 2),
                "physical_ratio": round(phys_r, 4),
                "context_ratio": round(ctx_r, 4),
                "total_kept": tot_kept,
                "total_seen": tot_seen,
                "num_dumps": n_dumps,
                "estimated_full_kv_gib": round(estimated_full_kv_gib, 4),
                "estimated_retained_kv_gib": round(estimated_retained_kv_gib, 4),
                "estimated_saved_kv_gib": round(estimated_saved_kv_gib, 4),
                "status": "OK",
            })
            print(f"    OK: {num_out} tok, {elapsed:.1f}s, "
                  f"phys={phys_r:.4f}, ctx={ctx_r:.4f} ({n_dumps} dumps)")
            print(f"    Est.KV: full={estimated_full_kv_gib:.3f} GiB, "
                  f"retained={estimated_retained_kv_gib:.3f} GiB, "
                  f"saved={estimated_saved_kv_gib:.3f} GiB")
            success = True
        except Exception as e:
            results.append({
                "config": label, "ratio": ratio,
                "error": f"{type(e).__name__}: {str(e)[:300]}",
                "status": "FAIL",
            })
            print(f"    FAIL: {type(e).__name__}: {e}")
            raise
        finally:
            if success:
                shutil.rmtree(dump_dir, ignore_errors=True)
            else:
                print(f"    (dump retained at {dump_dir} for debugging)")

    meta = {
        "git_commit": _get_git_commit(),
        "model": MODEL_PATH,
        "gpu": _get_gpu_name(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dtype": dtype_name,
        "page_group_size": PAGE_GROUP_SIZE,
        "compression_window_size": COMPRESSION_WINDOW_SIZE,
        "compression_n_sink_tokens": COMPRESSION_N_SINK_TOKENS,
        "compression_floor_min": COMPRESSION_FLOOR_MIN,
        "compression_chunk_size": COMPRESSION_CHUNK_SIZE,
        "max_model_len": model_len,
        "prompt_tokens": actual_tokens,
        "bytes_per_token": bytes_per_token,
        "command": " ".join(sys.argv),
    }

    out_path = os.path.join(output_dir, f"memory_results_{prompt_length}.json")
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)

    print(f"\n  {'Config':<10} {'Ratio':<6} {'PhysR':<8} {'CtxR':<8} "
          f"{'EstFull':<10} {'EstSaved':<10} {'Status'}")
    print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*6}")
    for r in results:
        phys = f"{r.get('physical_ratio', 0):.4f}" if 'physical_ratio' in r else "N/A"
        ctx  = f"{r.get('context_ratio', 0):.4f}" if 'context_ratio' in r else "N/A"
        full = f"{r.get('estimated_full_kv_gib', 0):.3f}" if 'estimated_full_kv_gib' in r else "N/A"
        saved = f"{r.get('estimated_saved_kv_gib', 0):.3f}" if 'estimated_saved_kv_gib' in r else "N/A"
        print(f"  {r['config']:<10} {r['ratio']:<6.2f} {phys:<8} {ctx:<8} "
              f"{full:<10} {saved:<10} {r['status']}")

    print(f"\n  Saved: {out_path}")


def main():
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(MODEL_PATH)
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads

    import torch
    dt = getattr(config, "torch_dtype", None) or getattr(config, "dtype", None)
    if dt is not None:
        dtype_bytes = torch.tensor([], dtype=dt).element_size()
        dtype_name = str(dt).split(".")[-1]
    else:
        dtype_bytes = 2
        dtype_name = "fp16"

    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes

    lengths = [int(l) for l in sys.argv[1:]] if len(sys.argv) > 1 else LENGTHS_DEFAULT
    for length in lengths:
        print(f"\n{'#'*60}")
        print(f"# RETENTION TEST: {length} tokens")
        print(f"{'#'*60}")
        sys.stdout.flush()
        run_one_length(length, OUTPUT_DIR, bytes_per_token, dtype_name,
                       num_layers, num_kv_heads, head_dim)

    print("\nDONE")


if __name__ == "__main__":
    main()
