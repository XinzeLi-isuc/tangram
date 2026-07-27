"""
GPU Smoke Test: verify CAKE-Serve pipeline works end-to-end.
Tests: FullKV → CAKE+uniform → CAKE+cake_layer.
"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from _cake_constants import MODEL_PATH
from vllm import LLM, SamplingParams

PROMPT = "KV cache compression is a critical technique for efficient LLM inference. " * 400
sp = SamplingParams(temperature=0, max_tokens=64)

tests = [
    ("FullKV", None, None, 1.0),
    ("CAKE+uniform", "cake", "uniform", 0.25),
    ("CAKE+cake_layer", "cake", "cake_layer", 0.25),
]

print("SMOKE TEST: CAKE-Serve end-to-end\n")
for name, scorer, level, ratio in tests:
    print(f"  [{name}] ratio={ratio}...", end=" ", flush=True)
    kwargs = dict(
        model=MODEL_PATH, compression_ratio=ratio,
        max_model_len=8192, gpu_memory_utilization=0.85,
        enforce_eager=True, disable_log_stats=True,
        page_group_size=4,
    )
    if scorer:
        kwargs["compression_scorer"] = scorer
        kwargs["compression_level"] = level
    try:
        llm = LLM(**kwargs)
        out = llm.generate([PROMPT], sp)
        text = out[0].outputs[0].text[:80]
        n_tok = len(out[0].outputs[0].token_ids)
        print(f"OK ({n_tok} tokens): {text}")
        del llm
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {str(e)[:200]}")

print("\nDONE")