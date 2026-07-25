# 实验报告：Day 2 — Tangram RULER 8K Baseline (SnapKV)

## 实验目的
在 RULER 8K 数据集上运行 SnapKV + crosslayer_cluster 作为 baseline，记录准确率、耗时和显存信息。

## 环境
- 同 Day 1
- 额外依赖: `pip install cloudpickle datasets pyarrow`

## 复现步骤

### 1. 安装依赖
```bash
conda activate cake-serve
pip install cloudpickle datasets pyarrow
pip install "huggingface-hub<1.0" "numpy<2.3,>=1.24"
```

### 2. 运行 RULER 8K benchmark
```bash
cd ~/cake-serve
CUDA_VISIBLE_DEVICES=0 \
MODEL=/home/lixinze/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
SCORER=snapkv \
LEVEL=crosslayer_cluster \
RATIOS="0.5 0.25" \
LENGTHS=8192 \
WINDOW_SIZE=32 \
FLOOR_MIN=0 \
GPU_MEM_UTIL=0.90 \
MAX_LEN=32768 \
NUM=50 \
PYTHON="/home/lixinze/miniconda3/envs/cake-serve/bin/python" \
bash benchmarks/tangram/benchmark_ruler.sh 2>&1 | tee results/raw/day02_ruler_run.log
```

### 3. 查看结果汇总
```bash
cd ~/cake-serve
# 脚本会自动打印汇总表
# 原始 JSON 结果在:
ls benchmarks/tangram/results_ruler/snapkv_crosslayer_cluster/len8192/*/Meta-Llama-3.1-8B-Instruct_r*.json
```

## 结果

### 13 个 RULER 任务的准确率

| 任务 | ratio=0.5 | ratio=0.25 |
|------|:---------:|:----------:|
| niah_single_1 | 98.0% | 88.0% |
| niah_single_2 | 84.0% | 78.0% |
| niah_single_3 | 96.0% | 82.0% |
| niah_multikey_1 | 94.0% | 78.0% |
| niah_multikey_2 | 88.0% | 74.0% |
| niah_multikey_3 | 72.0% | 20.0% |
| niah_multivalue | 92.0% | 90.0% |
| niah_multiquery | 95.0% | 92.5% |
| vt | 99.6% | 99.6% |
| cwe | 94.6% | 66.8% |
| fwe | 84.0% | 82.0% |
| qa_1 | 76.0% | 66.0% |
| qa_2 | 60.0% | 62.0% |
| **平均** | **87.2%** | **75.2%** |

### 运行时间
- 每个任务 50 样本: ~65s
- 总共 13 任务 × 2 比例: ~28 分钟

## 产出文件
- 原始 JSON: `results/raw/snapkv_ruler_8k_baseline/` (26 个文件)
- 运行日志: `results/raw/day02_ruler_run.log` (如使用 tee)
- 环境修复日志: `results/raw/day02_logs/env_fixes.md`
- 日报: `results/processed/day02_report.md`