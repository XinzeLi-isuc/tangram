"""Online serving saturation benchmark with repetition.

Scans offered QPS around the saturation point for 8K/16K contexts,
3 independent repetitions per (config, length, qps), then plots
throughput-TTFT/TPOT curves.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_serving_sat.py \
        --lengths 8192 16384 --repeats 3 --num-prompts 100
"""
import argparse, json, os, socket, subprocess, sys, time, urllib.request

from _cake_constants import MODEL_PATH
from _experiment_config import (
    CAKE_WINDOW_SIZE, CAKE_N_SINK_TOKENS, CAKE_FLOOR_MIN,
    CAKE_CHUNK_SIZE, CAKE_PAGE_GROUP_SIZE,
)

OUTPUT_DIR = "results/raw/day22_serving_sat"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

CONFIGS = [
    ("FullKV",  ["--compression-ratio", "1.0"], 1.0),
    ("CAKE_25", [
        "--compression-ratio", "0.25",
        "--compression-scorer", "cake",
        "--compression-level", "cake_layer",
    ], 0.25),
    ("CAKE_50", [
        "--compression-ratio", "0.5",
        "--compression-scorer", "cake",
        "--compression-level", "cake_layer",
    ], 0.5),
]

# QPS grids around saturation (inferred from pilot: 16K saturates ~0.8-1.2,
# 8K ~1.5-2.5 on A6000 with Llama-3.1-8B)
QPS_GRID = {
    8192:  [0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0],
    16384: [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
}


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _wait_ready(base_url, timeout_s=240):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            proxy_handler = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(proxy_handler)
            r = opener.open(f"{base_url}/v1/models", timeout=3)
            if r.status == 200:
                return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"Server at {base_url} did not become ready")


def run_one_bench(base_url, model, input_len, output_len, qps,
                  num_prompts, seed, out_path):
    """Run vllm bench serve once; returns metrics dict or None."""
    bench_cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "vllm", "--model", model,
        "--endpoint", "/v1/completions", "--base-url", base_url,
        "--dataset-name", "random", "--num-prompts", str(num_prompts),
        "--request-rate", str(qps), "--tokenizer", model,
        "--random-input-len", str(input_len), "--random-output-len", str(output_len),
        "--random-range-ratio", "0.1", "--seed", str(seed),
        "--save-result", "--result-dir", os.path.dirname(out_path),
        "--result-filename", os.path.basename(out_path),
    ]
    timeout = int(num_prompts / max(qps, 0.01) + 1200)
    r = subprocess.run(bench_cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        print(f"    bench FAILED rc={r.returncode}: {r.stderr[-300:]}")
        return None
    if not os.path.exists(out_path):
        print(f"    result file missing: {out_path}")
        return None
    with open(out_path) as f:
        return json.load(f)


def start_server(config_args, length, port):
    server_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH, "--dtype", "auto",
        "--max-model-len", str(length + 256), "--gpu-memory-utilization", "0.85",
        "--tensor-parallel-size", "1", "--disable-log-requests",
        "--page-group-size", str(CAKE_PAGE_GROUP_SIZE),
        "--compression-window-size", str(CAKE_WINDOW_SIZE),
        "--compression-n-sink-tokens", str(CAKE_N_SINK_TOKENS),
        "--compression-floor-min", str(CAKE_FLOOR_MIN),
        "--compression-chunk-size", str(CAKE_CHUNK_SIZE),
        "--port", str(port),
    ] + config_args
    log_path = os.path.join(OUTPUT_DIR, f"server_{port}.log")
    with open(log_path, "w") as lf:
        server = subprocess.Popen(server_cmd, stdout=lf, stderr=subprocess.STDOUT)
    return server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", nargs="+", type=int, default=[8192, 16384])
    ap.add_argument("--configs", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--num-prompts", type=int, default=100)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--seed-base", type=int, default=42)
    args = ap.parse_args()

    results = {}  # results[length][config][qps] = [rep1, rep2, rep3]

    for length in args.lengths:
        results[length] = {}
        qps_list = QPS_GRID.get(length, [1.0, 2.0])
        for ci in args.configs:
            name, extra_args, ratio = CONFIGS[ci]
            print(f"\n{'#'*70}")
            print(f"# {name} @ {length} (ratio={ratio})")
            print(f"# QPS grid: {qps_list}, repeats={args.repeats}")
            print(f"{'#'*70}", flush=True)

            port = _free_port()
            server = start_server(extra_args, length, port)
            base_url = f"http://localhost:{port}"
            results[length][name] = {}
            try:
                _wait_ready(base_url)
                print("Server ready")

                for qps in qps_list:
                    reps = []
                    for rep in range(args.repeats):
                        seed = args.seed_base + rep
                        out_name = f"{name}_len{length}_qps{qps}_rep{rep}"
                        out_path = os.path.join(OUTPUT_DIR, f"{out_name}.json")
                        print(f"  qps={qps} rep={rep} seed={seed} ...", flush=True)
                        t0 = time.time()
                        d = run_one_bench(
                            base_url, MODEL_PATH, length, args.output_len,
                            qps, args.num_prompts, seed, out_path)
                        print(f"    ({time.time()-t0:.0f}s) "
                              f"{'OK' if d else 'FAIL'}", flush=True)
                        if d is not None:
                            reps.append(d)
                        # cool-down between reps to drain the queue
                        time.sleep(10)
                    if reps:
                        results[length][name][str(qps)] = reps
            finally:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                time.sleep(2)

    # Save summary
    summary = {
        "lengths": args.lengths,
        "configs": [CONFIGS[ci][0] for ci in args.configs],
        "repeats": args.repeats,
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "results": results,
    }
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_path}")

    # Aggregated table
    print(f"\n{'='*95}")
    print(f"{'Config':<10} {'Len':>5} {'QPS':>5} "
          f"{'tok/s(med)':>11} {'TTFT50(med)':>12} {'TPOT50(med)':>12} "
          f"{'req/s(med)':>11} {'reps':>4}")
    print("-" * 95)
    for length in args.lengths:
        for name in results[length]:
            for qps_str, reps in results[length][name].items():
                qps = float(qps_str)
                toks = [r.get("total_token_throughput", 0) for r in reps]
                ttft = [r.get("median_ttft_ms", 0) for r in reps]
                tpot = [r.get("median_tpot_ms", 0) for r in reps]
                reqs = [r.get("request_throughput", 0) for r in reps]
                import statistics
                print(f"{name:<10} {length:>5} {qps:>5.1f} "
                      f"{statistics.median(toks):>11.0f} "
                      f"{statistics.median(ttft)/1000:>12.1f} "
                      f"{statistics.median(tpot):>12.1f} "
                      f"{statistics.median(reqs):>11.3f} {len(reps):>4}")
    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("DONE")


if __name__ == "__main__":
    main()
