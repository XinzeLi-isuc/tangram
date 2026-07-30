# Legacy / Invalidated Results

These results were produced before critical fixes or had methodological
issues. They should NOT be cited as current CAKE-Serve results.

| Item | Issue | Date Invalidated |
|------|-------|------------------|
| day12_report.md | Offline `llm.generate` batch mislabeled as "continuous batching" | 2026-07-30 |
| day13_report.md | RULER results from before preference lifecycle fix; data-loader bugs | 2026-07-30 |
| day15_report.md | 2K-token prompts claimed as "long context"; timing without cuda sync | 2026-07-30 |
| day11_phase1_comparison/ | Paper equivalence: budget sum 28248 vs target 31744 — failed | 2026-07-30 |
| day11_phase2_chunked/ | Simulation logic stale; duplicated old CAKE impl, not prod code | 2026-07-30 |
| day12_batching/ | Offline batch experiment, not online serving | 2026-07-30 |
| day13_ablation/ | Pre-fix RULER; incorrect data paths | 2026-07-30 |
| day15_ablation/ | False claims about context length; unverified Spearman/ms metrics | 2026-07-30 |

For current valid results, see:
- `results/raw/smoke/` — smoke test (3/3 PASS)
- `results/raw/day10_memory/` — retention verification
- `results/raw/day14_perf/` — 32K performance benchmark
