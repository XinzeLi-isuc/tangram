"""
Shared constants for CAKE-Serve scripts.
Consolidates hardcoded paths into a single source of truth.

Override via environment variables:
  CAKE_MODEL_PATH     — model directory (default: Local Llama-3.1-8B-Instruct)
  CAKE_SCBENCH_DIR    — SCBench data directory (default: ~/dataset/scbench/...)
"""
import os

_MODEL_DEFAULT = "meta-llama/Llama-3.1-8B-Instruct"

# Resolve model path: env var > default HF model name
MODEL_PATH = os.environ.get("CAKE_MODEL_PATH", _MODEL_DEFAULT)

# SCBench data directory
SCBENCH_DIR = os.environ.get(
    "CAKE_SCBENCH_DIR",
    os.path.expanduser(
        "~/dataset/scbench/datasets/microsoft--SCBench/snapshots/master/data"),
)
