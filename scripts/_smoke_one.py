"""
Single-config smoke runner (spawned by smoke_test.py).
Runs one vLLM config in isolation to avoid CUDA context contamination.
Usage: python _smoke_one.py '<json_config>'
"""
import json, os, sys, time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from _cake_constants import MODEL_PATH
from vllm import LLM, SamplingParams

PROMPT = (
    "KV cache compression is a critical technique for efficient LLM inference. "
    * 400
)
sp = SamplingParams(temperature=0, max_tokens=64)

config = json.loads(sys.argv[1])

kwargs = dict(
    model=MODEL_PATH,
    compression_ratio=config["ratio"],
    max_model_len=8192,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    disable_log_stats=True,
    page_group_size=4,
)
if config["scorer"]:
    kwargs["compression_scorer"] = config["scorer"]
    kwargs["compression_level"] = config["level"]

t0 = time.time()
llm = LLM(**kwargs)
out = llm.generate([PROMPT], sp)
elapsed = time.time() - t0
text = out[0].outputs[0].text[:120]
n_tok = len(out[0].outputs[0].token_ids)
del llm

print(
    json.dumps(
        {
            "status": "OK",
            "output_tokens": n_tok,
            "text_preview": text,
            "time_s": round(elapsed, 2),
        }
    )
)
