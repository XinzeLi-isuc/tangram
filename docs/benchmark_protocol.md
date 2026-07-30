# CAKE-Serve Benchmark Protocol

## Quality Evaluation

### Datasets
- **RULER**: 13 tasks (NIAH + multi-key/multi-value/query + QA + variable tracking),
  50 samples each. Lengths: 8K, 16K, 32K.
- **SCBench** (recommended for multi-turn): scbench_choice_eng, scbench_qa_eng,
  scbench_mf, scbench_vt.

### Configurations (5-way ablation)
| # | Scorer | Level | Purpose |
|---|--------|-------|---------|
| 1 | FullKV | — | Upper bound |
| 2 | SnapKV | uniform | Tangram baseline |
| 3 | SnapKV | crosslayer_cluster | Tangram strongest |
| 4 | CAKE | uniform | Token scorer only |
| 5 | CAKE | cake_layer | Full CAKE-Serve |

Note: SnapKV + cake_layer is NOT "budget level only" — SnapKV
produces no CAKE preference, so cake_layer falls back to uniform.
That row was removed from the ablation matrix.

### Ratios
`requested_ratio` (e.g. 0.25) vs measured `effective_physical_ratio`
after sink/window/alignment. Both must be reported.

### Commands
```bash
# RULER
cd ~/cake-serve
MODEL=meta-llama/Llama-3.1-8B-Instruct \
SCORER=cake LEVEL=cake_layer RATIOS="0.5 0.25" LENGTHS="8192" \
bash benchmarks/tangram/benchmark_ruler.sh
```

## Performance Evaluation

### Offline Batch
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/bench_offline_batch.py
```
Measures throughput scaling across batch sizes with fixed prompts.

### Online Serving (recommended)
```bash
# Terminal 1: server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --compression-ratio 0.25 --compression-scorer cake \
  --compression-level cake_layer --page-group-size 4

# Terminal 2: benchmark
vllm bench serve \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random --request-rate 2.0 \
  --num-prompts 500
```

## Metrics

### Quality
- RULER average accuracy per task
- Per-category breakdown (NIAH, Multi-key, QA, Variable tracking)

### Memory
- Peak GPU memory (nvidia-smi)
- KV block count (engine log)
- Effective retention ratio

### Serving
- Request throughput (req/s)
- Input/output token throughput (tok/s)
- TTFT P50/P95/P99
- TPOT P50/P95/P99
- Max no-OOM concurrency
- Scorer overhead (ms/layer)

## Reproducibility

Every result file must include:
```json
{
  "git_commit": "...",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "gpu": "RTX A6000 48GB",
  "created_at": "2026-07-25T12:00:00",
  "command": "SCORER=cake LEVEL=cake_layer ... bash benchmark_ruler.sh",
  "requested_ratio": 0.25,
  "effective_ratio": 0.268,
  "seed": 42
}
```