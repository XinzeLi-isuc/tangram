"""
Pure-CPU unit tests for retention dump parser (_parse_retention_dump).

Verifies:
  1. Compression config with no dump → RuntimeError
  2. FullKV (ratio=1.0) no dump → returns 1.0, 1.0, 0, 0, 0
  3. Missing required fields → RuntimeError
  4. Same (req, rank), multiple eval_len → picks max
  5. Known kept/total/sink/win → exact physical + context ratios

Run: pytest tests/cake_serve/test_retention_parser.py -v
"""
import os, tempfile
import numpy as np
import pytest

# Import the parser function from test_memory.py
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from test_memory import _parse_retention_dump


def _write_npz(dump_dir, filename, kept, total, sink, win, eval_len, req="0", rank=0):
    """Helper: write a single retention dump .npz file."""
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
    """Test _parse_retention_dump correctness on synthetic .npz files."""

    def test_compression_no_dump_raises(self):
        """Compression config (ratio<1) with no .npz files must raise."""
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(RuntimeError, match="no retention dump"):
                _parse_retention_dump(d, requested_ratio=0.5)

    def test_fullkv_no_dump_returns_one(self):
        """FullKV (ratio=1.0) with no dumps returns 1.0 ratios, zero counts."""
        with tempfile.TemporaryDirectory() as d:
            phys, ctx, kept, total, n = _parse_retention_dump(d, requested_ratio=1.0)
            assert phys == 1.0
            assert ctx == 1.0
            assert kept == 0
            assert total == 0
            assert n == 0

    def test_missing_fields_raises(self):
        """Dump file missing required fields (e.g. 'win') must raise."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "req0_0_0.npz")
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
                _parse_retention_dump(d, requested_ratio=0.5)

    def test_dedup_picks_max_seq(self):
        """Same (req, rank) → picks highest sequence number from filename."""
        with tempfile.TemporaryDirectory() as d:
            # seq=0: eval_len=100, kept=5 (earlier intermediate decision)
            _write_npz(d, "req0_r0_0.npz",
                       kept=[[5]], total=[[20]], sink=4, win=0,
                       eval_len=100, req="0", rank=0)
            # seq=1: same eval_len=100, kept=8 (final decision)
            _write_npz(d, "req0_r0_1.npz",
                       kept=[[8]], total=[[20]], sink=4, win=0,
                       eval_len=100, req="0", rank=0)

            phys, ctx, kept, total, n = _parse_retention_dump(d, requested_ratio=0.5)

            # Only the highest-seq record should contribute, not max eval_len
            assert n == 1
            assert kept == 8   # from seq=1, not seq=0
            assert total == 20

    def test_same_eval_len_different_seq(self):
        """Two records with identical eval_len, different seq → picks higher seq."""
        with tempfile.TemporaryDirectory() as d:
            # Higher seq=2 but lower eval_len=50, kept=12
            _write_npz(d, "req0_r0_2.npz",
                       kept=[[12]], total=[[20]], sink=4, win=0,
                       eval_len=50, req="0", rank=0)
            # Lower seq=1 but higher eval_len=100, kept=3
            _write_npz(d, "req0_r0_1.npz",
                       kept=[[3]], total=[[20]], sink=4, win=0,
                       eval_len=100, req="0", rank=0)

            phys, ctx, kept, total, n = _parse_retention_dump(d, requested_ratio=0.5)

            # seq=2 wins despite lower eval_len — final decision matters
            assert n == 1
            assert kept == 12
            assert total == 20

    def test_exact_ratios_known_data(self):
        """Known kept/total/sink/win → exact physical + context ratios."""
        with tempfile.TemporaryDirectory() as d:
            # Single head: total=100, kept=40, sink=4, win=8
            # physical_ratio = 40/100 = 0.4
            # context_ratio = (40-4-8)/(100-4) = 28/96 ≈ 0.291666...
            _write_npz(d, "req0_r0_0.npz",
                       kept=[[40]], total=[[100]], sink=4, win=8,
                       eval_len=200, req="0", rank=0)

            phys, ctx, kept, total, n = _parse_retention_dump(d, requested_ratio=0.5)

            assert n == 1
            assert kept == 40
            assert total == 100
            assert phys == pytest.approx(0.4)
            assert ctx == pytest.approx(28 / 96)

    def test_multiple_ranks_aggregated(self):
        """Multiple ranks on same req → summed aggregation."""
        with tempfile.TemporaryDirectory() as d:
            _write_npz(d, "req0_r0_0.npz",
                       kept=[[30, 20]], total=[[100, 80]], sink=4, win=2,
                       eval_len=200, req="0", rank=0)
            _write_npz(d, "req0_r1_0.npz",
                       kept=[[10, 40]], total=[[50, 60]], sink=4, win=2,
                       eval_len=200, req="0", rank=1)

            phys, ctx, kept, total, n = _parse_retention_dump(d, requested_ratio=0.5)

            # 2 unique keys: (0, 0) and (0, 1)
            assert n == 2
            # total kept = 30+20+10+40 = 100
            assert kept == 100
            # total seen = 100+80+50+60 = 290
            assert total == 290
            # phys = 100/290 ≈ 0.3448
            assert phys == pytest.approx(100 / 290)
            # context: for r0: ctx_kept = [30-4-2, 20-4-2] = [24, 14], ctx_total = [96, 76]
            #          for r1: ctx_kept = [10-4-2, 40-4-2] = [4, 34], ctx_total = [46, 56]
            # context_kept = 24+14+4+34 = 76, context_total = 96+76+46+56 = 274
            # context_ratio = 76/274
            assert ctx == pytest.approx(76 / 274)
