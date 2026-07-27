"""
GPU Smoke Test: verify CAKE-Serve pipeline works end-to-end.
Each config runs in its own subprocess to avoid CUDA context
contamination. Accumulates failures and exits non-zero on any
failure so CI can catch regressions.

Output: results/raw/smoke/smoke_results.json
"""
import json, os, subprocess, sys, time
from datetime import datetime, timezone

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from _cake_constants import MODEL_PATH

OUTPUT_DIR = "results/raw/smoke"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SMOKE_SCRIPT = os.path.join(os.path.dirname(__file__), "_smoke_one.py")

SMOKE_ONE_CONTENT = '''"""
Single-config smoke runner (spawned by smoke_test.py).
"""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from _cake_constants import MODEL_PATH
from vllm import LLM, SamplingParams

PROMPT = "KV cache compression is a critical technique for efficient LLM inference. " * 400
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

print(json.dumps({
    "status": "OK", "output_tokens": n_tok,
    "text_preview": text, "time_s": round(elapsed, 2),
}))
'''


def _get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=os.path.dirname(__file__) + "/.."
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


def main():
    # Write the subprocess runner script
    with open(SMOKE_SCRIPT, "w") as f:
        f.write(SMOKE_ONE_CONTENT)

    tests = [
        {"name": "FullKV", "scorer": None, "level": None, "ratio": 1.0},
        {"name": "CAKE+uniform", "scorer": "cake", "level": "uniform", "ratio": 0.25},
        {"name": "CAKE+cake_layer", "scorer": "cake", "level": "cake_layer", "ratio": 0.25},
    ]

    failures = []
    results = []
    print("SMOKE TEST: CAKE-Serve end-to-end\n")

    for cfg in tests:
        name = cfg["name"]
        print(f"  [{name}] ratio={cfg['ratio']}...", end=" ", flush=True)
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, SMOKE_SCRIPT, json.dumps(cfg)],
                capture_output=True, text=True, timeout=300,
                cwd=os.path.dirname(__file__) + "/..",
            )
            elapsed = time.time() - t0
            if proc.returncode != 0:
                msg = proc.stderr.strip()[-300:] if proc.stderr else f"exit={proc.returncode}"
                print(f"FAIL (exit={proc.returncode}): {msg}")
                failures.append({"name": name, "reason": msg, "exit_code": proc.returncode})
                results.append({"name": name, "status": "FAIL", "error": msg})
            else:
                try:
                    info = json.loads(proc.stdout.strip().split("\n")[-1])
                except json.JSONDecodeError:
                    info = {"status": "UNKNOWN", "raw": proc.stdout[-200:]}
                if info.get("status") == "OK":
                    print(f"OK ({info.get('output_tokens', '?')} tokens, "
                          f"{elapsed:.1f}s): {info.get('text_preview', '')[:60]}")
                else:
                    print(f"FAIL: {info}")
                    failures.append({"name": name, "reason": str(info)})
                info["name"] = name
                info["elapsed_s"] = round(elapsed, 2)
                results.append(info)
        except subprocess.TimeoutExpired:
            print("FAIL (timeout)")
            failures.append({"name": name, "reason": "timeout 300s"})
            results.append({"name": name, "status": "FAIL", "error": "timeout"})
        except Exception as e:
            print(f"FAIL: {e}")
            failures.append({"name": name, "reason": str(e)})
            results.append({"name": name, "status": "FAIL", "error": str(e)})

    # Write audit-ready JSON result
    meta = {
        "git_commit": _get_git_commit(),
        "model": MODEL_PATH,
        "gpu": _get_gpu_name(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(tests),
        "passed": len(tests) - len(failures),
        "failed": len(failures),
    }
    out_path = os.path.join(OUTPUT_DIR, "smoke_results.json")
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)

    # Cleanup
    try:
        os.unlink(SMOKE_SCRIPT)
    except OSError:
        pass

    print(f"\n{'='*60}")
    print(f"SMOKE SUMMARY: {meta['passed']}/{meta['total']} passed")
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f['name']}: {f['reason']}")
    print(f"Results: {out_path}")
    print("DONE")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
