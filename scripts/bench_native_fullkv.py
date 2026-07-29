"""Native-vLLM FullKV baseline: no Tangram ragged paging, no compression.
Separates vLLM-native overhead from Tangram page-group overhead.

Usage:  CUDA_VISIBLE_DEVICES=0 python scripts/bench_native_fullkv.py
"""
import json, os, time, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _cake_constants import MODEL_PATH as MODEL
from _real_data import build_real_prompt_ids

OUTPUT_DIR = "results/raw/day14_perf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_MODEL_LEN = 32768 + 128
TARGET_INPUT_TOKENS = MAX_MODEL_LEN - 128
MAX_OUTPUT_TOKENS = 128
BATCH_SIZES = [1, 2, 4]
WARMUP = 2
TRIALS = 5


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = build_real_prompt_ids(tokenizer, TARGET_INPUT_TOKENS)
    sp = SamplingParams(temperature=0, max_tokens=MAX_OUTPUT_TOKENS,
                        min_tokens=MAX_OUTPUT_TOKENS, ignore_eos=True)
    results = {}

    for bs in BATCH_SIZES:
        print(f"\n[Native-vLLM] batch={bs}", flush=True)
        try:
            llm = LLM(model=MODEL, max_model_len=MAX_MODEL_LEN,
                      gpu_memory_utilization=0.90, max_num_seqs=bs + 2)
            prompt_list = [{"prompt_token_ids": prompt_ids}] * bs

            for w in range(WARMUP):
                _ = llm.generate(prompt_list, sp)
            print(f"  warmup done", flush=True)

            times = []
            all_output_lens = []
            for t_idx in range(TRIALS):
                t0 = time.time()
                out = llm.generate(prompt_list, sp)
                elapsed = time.time() - t0
                times.append(elapsed)
                output_lens = [len(item.outputs[0].token_ids) for item in out]
                all_output_lens.append(output_lens)
                print(f"  trial {t_idx+1}: {elapsed:.1f}s out={output_lens}", flush=True)

            for t_idx, olen in enumerate(all_output_lens):
                if any(n != MAX_OUTPUT_TOKENS for n in olen):
                    raise RuntimeError(f"Trial {t_idx+1}: expected all {MAX_OUTPUT_TOKENS}, got {olen}")

            t_arr = np.array(times)
            results[bs] = {
                "batch_size": bs, "config": "Native-vLLM",
                "input_tokens": len(prompt_ids),
                "output_tokens_actual": all_output_lens,
                "output_tokens_target": MAX_OUTPUT_TOKENS,
                "median_s": float(np.median(t_arr)),
                "mean_s": float(np.mean(t_arr)),
                "std_s": float(np.std(t_arr)),
                "times_s": t_arr.tolist(),
                "throughput_req_s": float(bs / np.median(t_arr)),
            }
            del llm
        except Exception as e:
            if "OOM" in str(e).upper():
                print(f"  OOM at batch={bs}", flush=True)
                results[bs] = {"batch_size": bs, "error": "OOM"}
                break
            raise

    path = os.path.join(OUTPUT_DIR, "Native_vLLM_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {path}")
    print("DONE")


if __name__ == "__main__":
    main()
