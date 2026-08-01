"""Dump per-layer kept budget for CAKE_cake_layer on a CWE prompt.

Shows the actual per-layer kept counts to test whether any layer is
starved (e.g. budget < 100 tokens) — the suspected cause of CWE drop.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/_dump_layer_budget.py \
        --length 8192 --out results/raw/day23_cwe/budget_8192.json
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _cake_constants import MODEL_PATH as MODEL
from _experiment_config import (
    CAKE_WINDOW_SIZE, CAKE_N_SINK_TOKENS, CAKE_FLOOR_MIN,
    CAKE_CHUNK_SIZE, CAKE_PAGE_GROUP_SIZE, GPU_MEMORY_UTILIZATION,
    CAKE_GAMMA, CAKE_TAU1, CAKE_TAU2,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=8192)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq

    length = args.length
    df = pq.ParquetFile(f"dataset/ruler/{length}.parquet").read().to_pandas()
    cwe = df[df["task"] == "cwe"].head(1).iloc[0]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    kwargs = dict(
        model=MODEL,
        compression_ratio=0.25,
        compression_scorer="cake",
        compression_level="cake_layer",
        page_group_size=CAKE_PAGE_GROUP_SIZE,
        compression_window_size=CAKE_WINDOW_SIZE,
        compression_n_sink_tokens=CAKE_N_SINK_TOKENS,
        compression_floor_min=CAKE_FLOOR_MIN,
        compression_chunk_size=CAKE_CHUNK_SIZE,
        compression_cake_gamma=CAKE_GAMMA,
        compression_cake_tau1=CAKE_TAU1,
        compression_cake_tau2=CAKE_TAU2,
        compression_retention_dump=f"{args.out}.dump" if args.out else None,
        max_model_len=length + 256,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        disable_log_stats=True,
    )
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
    text = cwe["context"] + "\n" + cwe["question"]
    prompt = TokensPrompt(prompt_token_ids=tokenizer.encode(text))
    _ = llm.generate([prompt], sp)

    # Read per-layer budgets from the dump via _retention_utils
    from _retention_utils import load_final_decisions, summarize_retention
    dump_dir = f"{args.out}.dump" if args.out else None
    if dump_dir and os.path.isdir(dump_dir):
        decisions = load_final_decisions(dump_dir)
        kept = []
        for req_id, dec in decisions.items():
            if dec.kept_lengths is not None:
                kept = np.asarray(dec.kept_lengths)
        if len(kept):
            kept = kept.reshape(-1, kept.shape[-1])  # (layers, groups)
            per_layer = kept.mean(axis=1)
            print(f"\nPer-layer kept tokens (mean over groups), "
                  f"num_layers={len(per_layer)}:")
            for l in range(len(per_layer)):
                bar = "#" * int(per_layer[l] / max(per_layer.max(), 1) * 40)
                print(f"  L{l:>2}: {per_layer[l]:>7.1f} {bar}")
            print(f"\n  min={per_layer.min():.1f} "
                  f"max={per_layer.max():.1f} "
                  f"mean={per_layer.mean():.1f} "
                  f"ratio={per_layer.min()/per_layer.max():.3f}")
            if args.out:
                with open(args.out, "w") as f:
                    json.dump({
                        "length": length,
                        "per_layer_kept": per_layer.tolist(),
                        "min": float(per_layer.min()),
                        "max": float(per_layer.max()),
                        "mean": float(per_layer.mean()),
                    }, f, indent=2)
                print(f"\nSaved: {args.out}")

    del llm
    import gc; gc.collect()
    import torch; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
