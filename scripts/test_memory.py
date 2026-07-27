"""
Retention Verification: measure actual KV token retention via
compression_retention_dump. Reads saved .npz keep-decision files
to compute effective_ratio = kept_tokens / total_tokens.

Usage:
    cd ~/cake-serve
    CUDA_VISIBLE_DEVICES=0 python scripts/test_memory.py
"""
import json, os, shutil, sys, tempfile, time
import numpy as np

from _cake_constants import MODEL_PATH

OUTPUT_DIR = "results/raw/day10_memory"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIGS = [
    ("FullKV", "snapkv", "uniform", 1.0),
    ("CAKE_25", "cake", "cake_layer", 0.25),
    ("CAKE_50", "cake", "cake_layer", 0.5),
]

LENGTH = 8192
MAX_MODEL_LEN = 16384


def _parse_retention_dump(dump_dir: str, requested_ratio: float):
    """Parse retention dump .npz files and compute effective ratios.

    Schema (from vllm/v1/attention/compression/profiling.py):
        kept, total, sink, win, eval_len, req, rank

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

    # Deduplicate by (req, rank): pick record with max eval_len.
    # A single request can trigger multiple compression decisions
    # (e.g. chunked prefill); only the final decision matters.
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
        eval_len = int(d["eval_len"])
        if key not in by_key or eval_len > by_key[key]["eval_len"]:
            by_key[key] = {
                "kept": d["kept"].astype(np.int64),
                "total": d["total"].astype(np.int64),
                "sink": int(d["sink"]),
                "win": int(d["win"]),
                "eval_len": eval_len,
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


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer, AutoConfig

    config = AutoConfig.from_pretrained(MODEL_PATH)
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    num_groups = num_kv_heads // 4  # page_group_size=4

    # Estimate KV cache GiB (fp16): 2 * L * H * D * 2 bytes
    kv_bytes = 2 * num_layers * num_kv_heads * head_dim * 2
    kv_gib = kv_bytes / (1024 ** 3)
    print(f"Model: L={num_layers} H_kv={num_kv_heads} D={head_dim} "
          f"groups={num_groups} KV_cache≈{kv_gib:.1f} GiB/token")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    ids = tokenizer.encode("KV cache compression reduces GPU memory. ")
    repeat_len = len(ids) - 1
    prompt_ids = [ids[0]] + (ids[1:] * (LENGTH // repeat_len + 2))[:LENGTH]
    prompt = tokenizer.decode(prompt_ids)
    actual_tokens = len(tokenizer.encode(prompt))

    sp = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
    results = []

    for label, scorer, level, ratio in CONFIGS:
        print(f"\n{'='*60}")
        print(f"{label}  ratio={ratio}")
        print(f"{'='*60}")
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
                page_group_size=4,
                compression_retention_dump=dump_dir,
                max_model_len=MAX_MODEL_LEN,
                gpu_memory_utilization=0.85,
                disable_log_stats=True,
            )

            t0 = time.time()
            out = llm.generate([prompt], sp)
            elapsed = time.time() - t0
            num_out = len(out[0].outputs[0].token_ids)
            del llm

            phys_r, ctx_r, tot_kept, tot_seen, n_dumps = _parse_retention_dump(
                dump_dir, ratio)

            results.append({
                "config": label, "ratio": ratio,
                "input_tokens": actual_tokens,
                "output_tokens": num_out,
                "time_s": round(elapsed, 2),
                "physical_ratio": round(phys_r, 4),
                "context_ratio": round(ctx_r, 4),
                "total_kept": tot_kept,
                "total_seen": tot_seen,
                "num_dumps": n_dumps,
                "status": "OK",
            })
            print(f"  OK: {num_out} tok, {elapsed:.1f}s, "
                  f"physical_ratio={phys_r:.4f}, context_ratio={ctx_r:.4f} "
                  f"({n_dumps} dumps)")
            success = True
        except Exception as e:
            results.append({
                "config": label, "ratio": ratio,
                "error": f"{type(e).__name__}: {str(e)[:300]}",
                "status": "FAIL",
            })
            print(f"  FAIL: {type(e).__name__}: {e}")
            raise
        finally:
            if success:
                shutil.rmtree(dump_dir, ignore_errors=True)
            else:
                print(f"  (dump retained at {dump_dir} for debugging)")

    with open(os.path.join(OUTPUT_DIR, "memory_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"{'Config':<12} {'Ratio':<6} {'PhysR':<8} {'CtxR':<8} {'Toks':<8} {'Status'}")
    print(f"{'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for r in results:
        phys = f"{r.get('physical_ratio', 0):.4f}" if 'physical_ratio' in r else "N/A"
        ctx = f"{r.get('context_ratio', 0):.4f}" if 'context_ratio' in r else "N/A"
        out_tok = str(r.get("output_tokens", "ERR"))
        print(f"{r['config']:<12} {r['ratio']:<6.2f} {phys:<8} {ctx:<8} {out_tok:<8} {r['status']}")

    print(f"\nSaved: {OUTPUT_DIR}/memory_results.json")
    print("DONE")


if __name__ == "__main__":
    main()
