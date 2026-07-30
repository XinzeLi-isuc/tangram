"""
Shared retention dump parsing utilities.

Correct end-to-end metric: kept_token_cells / logical_context_cells,
NOT kept / resident_before_final (which is the final-step shrink ratio).

Usage:
    from _retention_utils import load_final_decisions, summarize_retention
    records = load_final_decisions(dump_dir)
    summary = summarize_retention(records, logical_tokens_by_req)
"""
import os, re
import numpy as np

# Required schema from profiling.py:
#   kept, total, sink, win, eval_len, req, rank
# Filename: {req_id}_r{rank}_{seq}.npz
REQUIRED_FIELDS = {"kept", "total", "sink", "win", "eval_len", "req", "rank"}
_SEQ_RE = re.compile(r"_r\d+_(\d+)\.npz$")


def load_final_decisions(dump_dir: str):
    """Return list of dicts — one per (req_id, rank) with highest seq.

    Each dict: {kept, total, sink, win, eval_len, req, rank, filename}
    """
    npz_files = sorted(f for f in os.listdir(dump_dir) if f.endswith(".npz"))
    if not npz_files:
        return []

    by_key: dict = {}
    for fn in npz_files:
        path = os.path.join(dump_dir, fn)
        d = np.load(path, allow_pickle=False)
        missing = REQUIRED_FIELDS - set(d.files)
        if missing:
            raise RuntimeError(
                f"Invalid dump {fn}: missing {sorted(missing)}. Found: {sorted(d.files)}"
            )

        key = (str(d["req"]), int(d["rank"]))
        m = _SEQ_RE.search(fn)
        seq = int(m.group(1)) if m else -1
        if key not in by_key or seq > by_key[key][1]:
            by_key[key] = (
                {
                    "kept": d["kept"].astype(np.int64),
                    "total": d["total"].astype(np.int64),
                    "sink": int(d["sink"]),
                    "win": int(d["win"]),
                    "eval_len": int(d["eval_len"]),
                    "req": str(d["req"]),
                    "rank": int(d["rank"]),
                    "filename": fn,
                },
                seq,
            )

    return [rec for rec, _ in by_key.values()]


def summarize_retention(records, logical_tokens_by_req):
    """Compute end-to-end retention metrics.

    Args:
        records: list of dicts from load_final_decisions()
        logical_tokens_by_req: dict[str, int] — original prompt length per req

    Returns dict with:
        effective_physical_ratio: kept_capacity / logical_capacity
        effective_evictable_ratio: (kept - sink - win) / (logical - sink - win) per cell
        final_step_shrink_ratio: kept_capacity / resident_before_final
        kept_token_cells: total kept cells (sum over all layers × groups)
        logical_token_cells: total capacity cells (= logical_tokens × num_cells)
        resident_before_final_cells: cells resident at final compression step
        num_unique_requests
    """
    kept_cells = 0
    resident_before_final = 0
    logical_cells = 0

    ctx_kept = np.int64(0)
    ctx_logical = np.int64(0)

    for rec in records:
        k = rec["kept"]
        r = rec["total"]
        req_id = rec["req"]
        logical_tokens = logical_tokens_by_req.get(req_id, 0)
        if logical_tokens <= 0:
            raise ValueError(
                f"Missing logical_tokens for req={req_id}. "
                f"Known keys: {sorted(logical_tokens_by_req.keys())}"
            )

        num_cells = k.size  # = num_layers × num_page_groups

        kept_cells += k.sum()
        resident_before_final += r.sum()
        logical_cells += logical_tokens * num_cells

        # Evictable region: exclude always-kept sink + recent window
        sink = rec["sink"]
        win = rec["win"]
        ctx_kept += np.maximum(k - sink - win, 0).sum()
        evictable_tokens = max(logical_tokens - sink - win, 0)
        ctx_logical += evictable_tokens * num_cells

    physical = kept_cells / logical_cells if logical_cells > 0 else 1.0
    final_step = kept_cells / resident_before_final if resident_before_final > 0 else 1.0
    evictable = float(ctx_kept / ctx_logical) if ctx_logical > 0 else 1.0

    return {
        "effective_physical_ratio": round(physical, 6),
        "effective_evictable_ratio": round(evictable, 6),
        "final_step_shrink_ratio": round(final_step, 6),
        "kept_token_cells": int(kept_cells),
        "logical_token_cells": int(logical_cells),
        "resident_before_final_cells": int(resident_before_final),
        "num_unique_requests": len({rec["req"] for rec in records}),
    }
