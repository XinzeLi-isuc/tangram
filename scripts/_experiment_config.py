"""
Standardized experiment configuration for CAKE-Serve.

All scripts MUST import config values from here, not hardcode their own.
Overrides via environment variables are documented per field.

Values are production defaults unless otherwise noted.
"""
import os

# ── Compression parameters (production default) ──────────────────────
CAKE_WINDOW_SIZE = int(os.environ.get("CAKE_WINDOW_SIZE", 32))
CAKE_N_SINK_TOKENS = int(os.environ.get("CAKE_N_SINK_TOKENS", 4))
CAKE_FLOOR_MIN = int(os.environ.get("CAKE_FLOOR_MIN", 0))
CAKE_CHUNK_SIZE = int(os.environ.get("CAKE_CHUNK_SIZE", 2048))
CAKE_PAGE_GROUP_SIZE = int(os.environ.get("CAKE_PAGE_GROUP_SIZE", 4))

# ── Model / dataset ───────────────────────────────────────────────────
# Override CAKE_MODEL_PATH env var if your model is elsewhere
_MODEL_DEFAULT = os.path.expanduser(
    "~/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
)
MODEL_PATH = os.environ.get("CAKE_MODEL_PATH", _MODEL_DEFAULT)

# ── Benchmark geometry ────────────────────────────────────────────────
# Prompt lengths used in different experiments
RETENTION_PROMPT_LENGTH = int(os.environ.get("CAKE_RETENTION_LENGTH", 8192))
PERF_PROMPT_LENGTH = int(os.environ.get("CAKE_PERF_LENGTH", 32768))
MAX_OUTPUT_TOKENS = int(os.environ.get("CAKE_MAX_OUTPUT", 128))
MAX_MODEL_LEN = int(os.environ.get("CAKE_MAX_MODEL_LEN", 32768 + 128))

# ── GPU ───────────────────────────────────────────────────────────────
GPU_MEMORY_UTILIZATION = float(os.environ.get("CAKE_GPU_MEM_UTIL", 0.90))
