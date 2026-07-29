"""
Shared real-data prompt builder for CAKE-Serve benchmarks.
Uses SCBench (scbench_qa_eng) context as real, non-synthetic text.

Usage:
    from _real_data import build_real_prompt_ids
    pids = build_real_prompt_ids(tokenizer, target_tokens=32768)
"""
import json, os

# First SCBench qa_eng entry has 381K chars — more than enough for 32K tokens
_DEFAULT_DATASET = os.path.expanduser(
    "~/dataset/scbench/datasets/microsoft--SCBench/snapshots/master/data/scbench_qa_eng.jsonl"
)

_CACHED_TEXT = None


def _load_context(dataset_path=None):
    """Load first SCBench context (lazy-cached per process)."""
    global _CACHED_TEXT
    if _CACHED_TEXT is not None:
        return _CACHED_TEXT
    path = dataset_path or _DEFAULT_DATASET
    with open(path) as f:
        first = json.loads(f.readline())
    _CACHED_TEXT = first["context"]
    return _CACHED_TEXT


def build_real_prompt_ids(tokenizer, target_tokens, dataset_path=None):
    """Build exact `target_tokens` prompt_token_ids from real SCBench text.

    Uses tokenizer.encode on real context, then truncates or pads.
    Returns list[int] of length exactly `target_tokens`.
    """
    text = _load_context(dataset_path)
    all_ids = tokenizer.encode(text)

    if len(all_ids) >= target_tokens:
        # Truncate to exact target (keeping BOS if present)
        return all_ids[:target_tokens]

    # Pad by repeating if shorter (shouldn't happen with SCBench 380K+ char contexts)
    repeat_unit = all_ids[1:] if all_ids and all_ids[0] == tokenizer.bos_token_id else all_ids
    while len(all_ids) < target_tokens:
        all_ids.extend(repeat_unit)
    return all_ids[:target_tokens]
