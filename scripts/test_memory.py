"""
Retention Verification: measure actual KV token retention via
compression_retention_dump. Reads saved .npz keep-decision files
to compute effective_ratio = kept_tokens / total_tokens.

Usage:
    cd ~/cake-serve
    CUDA_VISIBLE_DEVICES=0 python scripts/test_memory.py
"""
import json, os, shutil, sys, tempfile, time, numpy as np

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


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer, AutoConfig

    config = AutoConfig.from_pretrained(MODEL_PATH)
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    num_groups = num_kv_heads // 4  # page_group_size=4

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

            # Read retention dump to compute effective_ratio
            npz_files = sorted(
                f for f in os.listdir(dump_dir) if f.endswith(".npz"))
            total_kept = 0
            total_full = 0
            for fn in npz_files:
                data = np.load(os.path.join(dump_dir, fn))
                total_kept += int(data.get("kept_lengths", np.array(0)).sum())
                total_full += int(data.get("total_seen", np.array(0)).sum())

            eff_ratio = (total_kept / total_full) if total_full > 0 else 1.0

            results.append({
                "config": label, "ratio": ratio,
                "input_tokens": actual_tokens,
                "output_tokens": num_out,
                "time_s": round(elapsed, 2),
                "effective_ratio": round(eff_ratio, 4),
                "total_kept": total_kept,
                "total_seen": total_full,
                "status": "OK",
            })
            print(f"  OK: {num_out} tok, {elapsed:.1f}s, "
                  f"eff_ratio={eff_ratio:.4f}")
        except Exception as e:
            results.append({
                "config": label, "ratio": ratio,
                "error": str(e)[:300], "status": "FAIL",
            })
            print(f"  FAIL: {type(e).__name__}")
            raise
        finally:
            shutil.rmtree(dump_dir, ignore_errors=True)

    with open(os.path.join(OUTPUT_DIR, "memory_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"{'Config':<12} {'Ratio':<6} {'EffRatio':<10} {'Toks':<8} {'Status'}")
    print(f"{'-'*12} {'-'*6} {'-'*10} {'-'*8} {'-'*6}")
    for r in results:
        eff = f"{r.get('effective_ratio', 0):.4f}" if 'effective_ratio' in r else "N/A"
        out = str(r.get("output_tokens", "ERR"))
        print(f"{r['config']:<12} {r['ratio']:<6.2f} {eff:<10} {out:<8} {r['status']}")

    print(f"\nSaved: {OUTPUT_DIR}/memory_results.json")
    print("DONE")


if __name__ == "__main__":
    main()