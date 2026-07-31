"""
Online Serving Benchmark using vllm bench serve.
True continuous batching with async request arrival.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_serving.py 2>&1 | tee results/raw/day17_serving.log
"""
import json, os, socket, subprocess, sys, time, urllib.request

from _cake_constants import MODEL_PATH
from _experiment_config import (
    CAKE_WINDOW_SIZE, CAKE_N_SINK_TOKENS, CAKE_FLOOR_MIN,
    CAKE_CHUNK_SIZE, CAKE_PAGE_GROUP_SIZE,
)

OUTPUT_DIR = "results/raw/day17_serving"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Bypass HTTP proxy for localhost (required in restricted network env)
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")


def run_bench(config_name, extra_args, ratio, request_rate, num_prompts=200):
    """Start vllm serve, run bench, collect metrics."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {config_name} (ratio={ratio}, rate={request_rate})")
    print(f"{'='*60}")

    port = _free_port()
    base_url = f"http://localhost:{port}"
    out_name = f"{config_name}_r{ratio}_qps{request_rate}"

    server_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH, "--dtype", "auto",
        "--max-model-len", "16384", "--gpu-memory-utilization", "0.85",
        "--tensor-parallel-size", "1", "--disable-log-requests",
        "--page-group-size", str(CAKE_PAGE_GROUP_SIZE),
        "--compression-window-size", str(CAKE_WINDOW_SIZE),
        "--compression-n-sink-tokens", str(CAKE_N_SINK_TOKENS),
        "--compression-floor-min", str(CAKE_FLOOR_MIN),
        "--compression-chunk-size", str(CAKE_CHUNK_SIZE),
        "--port", str(port),
    ] + extra_args

    server_log = os.path.join(OUTPUT_DIR, f"{out_name}_server.log")
    with open(server_log, "w") as lf:
        server = subprocess.Popen(server_cmd, stdout=lf, stderr=subprocess.STDOUT)
    print(f"Server PID={server.pid} port={port}")

    try:
        _wait_ready(base_url)
        print("Server ready")

        bench_cmd = [
            sys.executable, "-m", "vllm.entrypoints.cli.main",
            "bench", "serve",
            "--backend", "vllm", "--model", MODEL_PATH,
            "--endpoint", "/v1/completions", "--base-url", base_url,
            "--dataset-name", "random", "--num-prompts", str(num_prompts),
            "--request-rate", str(request_rate), "--tokenizer", MODEL_PATH,
            "--random-input-len", "8192", "--random-output-len", "128",
            "--random-range-ratio", "0.1", "--seed", "42",
            "--save-result", "--result-dir", OUTPUT_DIR,
            "--result-filename", f"{out_name}.json",
        ]
        r = subprocess.run(bench_cmd, capture_output=True, text=True,
                           timeout=max(1800, int(num_prompts / max(request_rate, 0.01) + 900)),
                           check=True)
        tail = r.stdout[-800:] if len(r.stdout) > 800 else r.stdout
        print(tail)

    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        time.sleep(1)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _wait_ready(base_url, timeout_s=180):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{base_url}/v1/models")
            req.set_proxy("localhost:0", "http")  # bypass proxy
            r = urllib.request.urlopen(req, timeout=3)
            if r.status == 200:
                return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"Server at {base_url} did not become ready")


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
            run_bench(name, args, ratio, rate)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("DONE")