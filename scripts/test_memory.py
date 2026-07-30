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
from _retention_utils import load_final_decisions, summarize_retention
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




def run_one_length(prompt_length, output_dir, bytes_per_token, dtype_name,
                  num_layers, num_kv_heads, head_dim):
    """Run retention verification for a single prompt length.

    Uses TokensPrompt (no decode-retokenize) and logical context capacity
    for end-to-end ratio computation.
    """
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt_ids = build_real_prompt_ids(tokenizer, prompt_length)
    prompt_input = TokensPrompt(prompt_token_ids=prompt_ids)
    actual_tokens = len(prompt_ids)
    estimated_full_kv_gib = bytes_per_token * actual_tokens / (1024 ** 3)
    model_len = prompt_length + MAX_OUTPUT_TOKENS
    gpu_util = GPU_MEMORY_UTILIZATION if prompt_length <= 8192 else 0.85

    print(f"\n  Model: L={num_layers} H_kv={num_kv_heads} D={head_dim} "
          f"dtype={dtype_name} page_group_size={PAGE_GROUP_SIZE}")
    print(f"  Prompt: {actual_tokens} tokens → estimated full KV ≈ "
          f"{estimated_full_kv_gib:.3f} GiB")
    sys.stdout.flush()

    sp = SamplingParams(temperature=0, max_tokens=32,  # small output for retention
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
                gpu_memory_utilization=gpu_util,
                disable_log_stats=True,
            )

            t0 = time.time()
            out = llm.generate([prompt_input], sp)
            elapsed = time.time() - t0
            engine_input_tokens = len(out[0].prompt_token_ids)
            num_out = len(out[0].outputs[0].token_ids)
            del llm
            import gc; gc.collect()
            import torch; torch.cuda.empty_cache()

            # Verify no decode-retokenize drift
            assert engine_input_tokens == prompt_length, \
                f"Engine saw {engine_input_tokens} tokens, expected {prompt_length}"

            # End-to-end metrics using logical context capacity
            num_decisions_total = len([
                f for f in os.listdir(dump_dir) if f.endswith(".npz")
            ])
            records = load_final_decisions(dump_dir)
            num_final_records = len(records)

            if ratio < 1.0 and num_final_records == 0:
                raise RuntimeError(
                    f"ratio={ratio} < 1 but no retention dump in {dump_dir}")
            elif num_final_records == 0:
                # FullKV: no compression, all cells kept = logical cells
                num_groups = num_kv_heads // PAGE_GROUP_SIZE
                logical_cells = engine_input_tokens * num_layers * num_groups
                summary = {
                    "effective_physical_ratio": 1.0,
                    "effective_evictable_ratio": 1.0,
                    "final_step_shrink_ratio": 1.0,
                    "kept_token_cells": logical_cells,
                    "logical_token_cells": logical_cells,
                    "resident_before_final_cells": logical_cells,
                    "num_unique_requests": 1,
                }
            else:
                logical_by_req = {out[0].request_id: engine_input_tokens}
                summary = summarize_retention(records, logical_by_req)

            phys_r = summary["effective_physical_ratio"]
            ctx_r = summary["effective_evictable_ratio"]
            final_step = summary["final_step_shrink_ratio"]
            kept_cells = summary["kept_token_cells"]
            logical_cells = summary["logical_token_cells"]

            estimated_retained_kv_gib = estimated_full_kv_gib * phys_r
            estimated_saved_kv_gib = estimated_full_kv_gib - estimated_retained_kv_gib

            results.append({
                "config": label,
                "requested_ratio": ratio,
                "effective_physical_ratio": round(phys_r, 6),
                "effective_evictable_ratio": round(ctx_r, 6),
                "final_step_shrink_ratio": round(final_step, 6),
                "input_tokens": actual_tokens,
                "engine_input_tokens": engine_input_tokens,
                "output_tokens": num_out,
                "time_s": round(elapsed, 2),
                "kept_token_cells": kept_cells,
                "logical_token_cells": logical_cells,
                "num_decisions_total": num_decisions_total,
                "num_final_records": num_final_records,
                "estimated_full_kv_gib": round(estimated_full_kv_gib, 4),
                "estimated_retained_kv_gib": round(estimated_retained_kv_gib, 4),
                "estimated_saved_kv_gib": round(estimated_saved_kv_gib, 4),
                "status": "OK",
            })
            print(f"    OK: {num_out} tok, {elapsed:.1f}s, "
                  f"phys={phys_r:.4f} final-step={final_step:.4f} "
                  f"({num_decisions_total} total / {num_final_records} final)")
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

    print(f"\n  {'Config':<10} {'PhysR':>8} {'FinalR':>8} {'EstFull':>10} {'EstSaved':>10} {'Status'}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*6}")
    for r in results:
        if 'effective_physical_ratio' in r:
            phys = f"{r['effective_physical_ratio']:.4f}"
            final = f"{r.get('final_step_shrink_ratio', 0):.4f}"
        else:
            phys = final = "N/A"
        full = f"{r.get('estimated_full_kv_gib', 0):.3f}" if 'estimated_full_kv_gib' in r else "N/A"
        saved = f"{r.get('estimated_saved_kv_gib', 0):.3f}" if 'estimated_saved_kv_gib' in r else "N/A"
        print(f"  {r['config']:<10} {phys:>8} {final:>8} {full:>10} {saved:>10} {r['status']}")

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
