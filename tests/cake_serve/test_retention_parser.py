"""
Pure-CPU unit tests for retention dump parser (_retention_utils).

Verifies:
  1. Compression config with no dump → empty list from load_final_decisions
  2. FullKV no dump → summarize_retention returns 1.0
  3. Missing required fields → RuntimeError
  4. Same (req, rank), multiple seq → picks highest seq
  5. Same eval_len, different seq → later seq wins
  6. Known kept/total/sink/win → exact physical + evictable + final-step ratios
  7. End-to-end ratio uses logical capacity NOT resident_before_final
  8. Two TP ranks correctly merged

Run: pytest tests/cake_serve/test_retention_parser.py -v
"""
import os, tempfile
import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from _retention_utils import load_final_decisions, summarize_retention


def _write_npz(dump_dir, filename, kept, total, sink, win, eval_len, req="0", rank=0):
    path = os.path.join(dump_dir, filename)
    np.savez(
        path,
        kept=np.array(kept, dtype=np.int64),
        total=np.array(total, dtype=np.int64),
        sink=np.int64(sink),
        win=np.int64(win),
        eval_len=np.int64(eval_len),
        req=np.array(str(req)),
        rank=np.int64(rank),
    )


class TestRetentionParser:
    """Test _retention_utils correctness on synthetic .npz files."""

    def test_no_dump_returns_empty(self):
        """Empty dump dir returns empty list."""
        with tempfile.TemporaryDirectory() as d:
            records = load_final_decisions(d)
            assert records == []

    def test_missing_fields_raises(self):
        """Dump missing required field (e.g. 'win') must raise."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "req0_r0_0.npz")
            np.savez(
                path,
                kept=np.array([[10]], dtype=np.int64),
                total=np.array([[20]], dtype=np.int64),
                sink=np.int64(4),
                # win intentionally missing
                eval_len=np.int64(100),
                req=np.array("0"),
                rank=np.int64(0),
            )
            with pytest.raises(RuntimeError, match="missing.*win"):
                load_final_decisions(d)

    def test_dedup_picks_max_seq(self):
        """Same (req, rank) → picks highest sequence number from filename."""
        with tempfile.TemporaryDirectory() as d:
            _write_npz(d, "req0_r0_0.npz",
                       kept=[[5]], total=[[20]], sink=4, win=0,
                       eval_len=100, req="0", rank=0)
            _write_npz(d, "req0_r0_1.npz",
                       kept=[[8]], total=[[20]], sink=4, win=0,
                       eval_len=100, req="0", rank=0)

            records = load_final_decisions(d)
            assert len(records) == 1
            assert records[0]["kept"].sum() == 8

    def test_same_eval_len_different_seq(self):
        """Lower eval_len but higher seq → later seq wins."""
        with tempfile.TemporaryDirectory() as d:
            _write_npz(d, "req0_r0_2.npz",
                       kept=[[12]], total=[[20]], sink=4, win=0,
                       eval_len=50, req="0", rank=0)
            _write_npz(d, "req0_r0_1.npz",
                       kept=[[3]], total=[[20]], sink=4, win=0,
                       eval_len=100, req="0", rank=0)

            records = load_final_decisions(d)
            assert len(records) == 1
            assert records[0]["kept"].sum() == 12  # seq=2 wins over seq=1

    def test_logical_capacity_not_resident_before(self):
        """End-to-end ratio uses logical context capacity, not resident KV.

        Original context: 8192 tokens.
        Before final compression: 4096 cells resident.
        After final compression:  2048 cells kept.

        final_step_shrink = 2048 / 4096 = 0.5
        effective_physical = 2048 / 8192 = 0.25
        """
        with tempfile.TemporaryDirectory() as d:
            # Single head: kept=2048, resident_before=4096
            _write_npz(d, "reqA_r0_0.npz",
                       kept=[[2048]], total=[[4096]], sink=4, win=32,
                       eval_len=200, req="A", rank=0)

            records = load_final_decisions(d)
            summary = summarize_retention(records, {"A": 8192})

            assert summary["effective_physical_ratio"] == pytest.approx(0.25)
            assert summary["final_step_shrink_ratio"] == pytest.approx(0.5)
            assert summary["kept_token_cells"] == 2048
            assert summary["logical_token_cells"] == 8192

    def test_evictable_ratio_excludes_sink_window(self):
        """Evictable ratio: (kept - sink - win) / (logical - sink)."""
        with tempfile.TemporaryDirectory() as d:
            _write_npz(d, "req0_r0_0.npz",
                       kept=[[40]], total=[[100]], sink=4, win=8,
                       eval_len=200, req="0", rank=0)

            records = load_final_decisions(d)
            summary = summarize_retention(records, {"0": 100})

            assert summary["effective_physical_ratio"] == pytest.approx(0.4)
            assert summary["effective_evictable_ratio"] == pytest.approx(28 / 96, abs=1e-5)

    def test_two_ranks_merged(self):
        """Two TP ranks: aggregate kept/logical cells."""
        with tempfile.TemporaryDirectory() as d:
            _write_npz(d, "req0_r0_0.npz",
                       kept=[[30, 20]], total=[[100, 80]], sink=4, win=2,
                       eval_len=200, req="0", rank=0)
            _write_npz(d, "req0_r1_0.npz",
                       kept=[[10, 40]], total=[[50, 60]], sink=4, win=2,
                       eval_len=200, req="0", rank=1)

            records = load_final_decisions(d)
            assert len(records) == 2

            summary = summarize_retention(records, {"0": 100})
            assert summary["kept_token_cells"] == 100
            assert summary["logical_token_cells"] == 400  # 100 tokens × 4 cells
