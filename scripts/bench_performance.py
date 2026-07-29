"""
Day 14: Performance Benchmark (32K context, 128 output tokens)
FullKV vs CAKE 25% vs CAKE 50%
Each config: warmup 2x, measurement 5x

Fixes:
  - Use prompt_token_ids for precise token count (no encode→truncate→decode→encode drift)
  - Record actual input/output token counts per trial
  - Write per-config JSON before OOM break (was silently skipped)
  - Verify actual_input_tokens + 128 <= max_model_len
"""
import json, os, time
import numpy as np

from _cake_constants import MODEL_PATH as MODEL
from _real_data import build_real_prompt_ids

OUTPUT_DIR = "results/raw/day14_perf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_MODEL_LEN = 32768 + 128  # room for output
TARGET_INPUT_TOKENS = MAX_MODEL_LEN - 128
MAX_OUTPUT_TOKENS = 128


def run_config(name, scorer, level, ratio, batch_sizes, warmup=2, trials=5):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = build_real_prompt_ids(tokenizer, TARGET_INPUT_TOKENS)
    sp = SamplingParams(
        temperature=0, max_tokens=MAX_OUTPUT_TOKENS,
        min_tokens=MAX_OUTPUT_TOKENS, ignore_eos=True,
    )
    results = {}

    for bs in batch_sizes:
        print(f"\n  [{name}] batch={bs}", flush=True)
        result_entry = None

        try:
            llm = LLM(
                model=MODEL, compression_ratio=ratio,
                compression_scorer=scorer, compression_level=level,
                page_group_size=4,
                max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.90,
                max_num_seqs=bs + 2,
            )

            prompt_dict = {"prompt_token_ids": prompt_ids}
            prompt_list = [prompt_dict] * bs

            # Warmup with same batch size as measurement
            for w in range(warmup):
                _ = llm.generate(prompt_list, sp)
                print(f"    warmup {w+1}/{warmup} done", flush=True)

            # Measurement
            times = []
            all_output_lens = []
            for t_idx in range(trials):
                t0 = time.time()
                out = llm.generate(prompt_list, sp)
                elapsed = time.time() - t0
                times.append(elapsed)
                # Record all output lengths, verify all hit target
                output_lens = [
                    len(item.outputs[0].token_ids) for item in out
                ]
                all_output_lens.append(output_lens)
                print(f"    trial {t_idx+1}/{trials}: {elapsed:.1f}s "
                      f"(output={output_lens})",
                      flush=True)

            # Fail-fast if any batch member didn't produce target output length
            for t_idx, output_lens in enumerate(all_output_lens):
                if any(n != MAX_OUTPUT_TOKENS for n in output_lens):
                    raise RuntimeError(
                        f"Trial {t_idx+1}: expected all outputs to be "
                        f"{MAX_OUTPUT_TOKENS} tokens, got {output_lens}"
                    )

            t_arr = np.array(times)
            result_entry = {
                "batch_size": bs,
                "config": name,
                "input_tokens": len(prompt_ids),
                "output_tokens_actual": all_output_lens,
                "output_tokens_target": MAX_OUTPUT_TOKENS,
                "median_s": float(np.median(t_arr)),
                "p50_s": float(np.percentile(t_arr, 50)),
                "p95_s": float(np.percentile(t_arr, 95)),
                "mean_s": float(np.mean(t_arr)),
                "std_s": float(np.std(t_arr)),
                "times_s": t_arr.tolist(),
                "throughput_req_s": float(bs / np.median(t_arr)),
            }
            results[bs] = result_entry
            del llm

        except Exception as e:
            if "OOM" in str(e).upper() or "out of memory" in str(e).lower():
                print(f"    OOM at batch={bs}", flush=True)
                result_entry = {"batch_size": bs, "error": "OOM"}
                results[bs] = result_entry
                # Write per-config JSON before breaking so OOM cell is preserved
                _write_config_json(name, results)
                break
            raise  # fail-fast on other errors
        finally:
            # Write per-config JSON after every batch size (even on success)
            if result_entry is not None and "error" not in result_entry:
                _write_config_json(name, results)

    return results


def _write_config_json(name, results):
    path = os.path.join(OUTPUT_DIR, f"{name}_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


print("Day 14: Performance Benchmark (32K)", flush=True)
print("=" * 60, flush=True)

batch_sizes = [1, 2, 4, 6, 8, 10]

r1 = run_config("FullKV", "snapkv", "uniform", 1.0, batch_sizes)
r2 = run_config("CAKE_25", "cake", "cake_layer", 0.25, batch_sizes)
r3 = run_config("CAKE_50", "cake", "cake_layer", 0.5, batch_sizes)

print("\n\n" + "=" * 60, flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"{'Config':<12} {'Batch':>5} {'Med(s)':>8} {'P95(s)':>8} {'req/s':>8}", flush=True)
print("-" * 45, flush=True)
for name, r in [("FullKV", r1), ("CAKE_25", r2), ("CAKE_50", r3)]:
    for bs in sorted(r.keys()):
        d = r[bs]
        if "median_s" in d:
            print(f"{name:<12} {bs:>5} {d['median_s']:>8.1f} "
                  f"{d['p95_s']:>8.1f} {d['throughput_req_s']:>8.1f}", flush=True)
        else:
            print(f"{name:<12} {bs:>5} {'OOM':>8}", flush=True)

all_results = {"FullKV": r1, "CAKE_25": r2, "CAKE_50": r3}
with open(os.path.join(OUTPUT_DIR, "perf_results.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nResults saved", flush=True)
print("DONE", flush=True)
