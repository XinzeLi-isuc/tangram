"""
Online Serving Benchmark using vllm bench serve.
True continuous batching with async request arrival.

Usage:
    python scripts/bench_serving.py 2>&1 | tee results/raw/day17_serving.log
"""
import json, os, subprocess, sys, time

from _cake_constants import MODEL_PATH

OUTPUT_DIR = "results/raw/day17_serving"
os.makedirs(OUTPUT_DIR, exist_ok=True)
GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "0")


def run_bench(config_name, extra_args, ratio, request_rate, num_prompts=100):
    """Start vllm serve, run bench, collect metrics."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {config_name} (ratio={ratio}, rate={request_rate})")
    print(f"{'='*60}")

    # Kill any existing server
    subprocess.run(["pkill", "-f", "vllm.entrypoints.openai"], capture_output=True)
    time.sleep(2)

    # Start server in background
    server_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--dtype", "auto",
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.85",
        "--tensor-parallel-size", "1",
        "--disable-log-requests",
        "--port", "8000",
    ] + extra_args

    server = subprocess.Popen(server_cmd,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    print(f"Server PID: {server.pid}")

    # Wait for server to be ready
    for _ in range(60):
        try:
            import urllib.request
            r = urllib.request.urlopen("http://localhost:8000/health", timeout=2)
            if r.status == 200:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("ERROR: Server did not start")
        server.kill()
        return None

    print("Server ready")

    # Run benchmark
    bench_cmd = [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--model", MODEL_PATH,
        "--endpoint", "/v1/completions",
        "--base-url", "http://localhost:8000",
        "--dataset-name", "random",
        "--dataset-path", "",
        "--num-prompts", str(num_prompts),
        "--request-rate", str(request_rate),
        "--tokenizer", MODEL_PATH,
        "--save-result",
        "--result-dir", OUTPUT_DIR,
        "--result-filename", f"{config_name}_r{ratio}_qps{request_rate}.json",
    ]
    result = subprocess.run(bench_cmd, capture_output=True, text=True, timeout=600)
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

    # Kill server
    server.terminate()
    server.wait(timeout=10)
    time.sleep(2)


if __name__ == "__main__":
    configs = [
        ("FullKV", ["--compression-ratio", "1.0"], 1.0),
        ("CAKE_25", [
            "--compression-ratio", "0.25",
            "--compression-scorer", "cake",
            "--compression-level", "cake_layer",
            "--page-group-size", "4",
        ], 0.25),
        ("CAKE_50", [
            "--compression-ratio", "0.5",
            "--compression-scorer", "cake",
            "--compression-level", "cake_layer",
            "--page-group-size", "4",
        ], 0.5),
    ]

    for name, args, ratio in configs:
        for rate in [0.5, 1.0, 2.0]:
            run_bench(name, args, ratio, rate, num_prompts=50)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("DONE")