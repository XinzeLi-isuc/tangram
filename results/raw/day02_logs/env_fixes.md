# Day 2 环境修复日志

## 问题
- `benchmark_ruler.py` 需要 `cloudpickle`, `datasets`, `pyarrow` 等依赖
- 安装 `datasets` 时升级了 `numpy` 到 2.5.1，导致 `numba` 报错（numba 需要 numpy <= 2.2）
- `datasets` 安装时也升级了 `huggingface-hub` 到 1.24.0，导致 `transformers` 4.57.6 报错（需要 huggingface-hub < 1.0）

## 修复
1. `pip install cloudpickle` — 安装缺失依赖
2. `pip install "huggingface-hub<1.0"` — 降级 huggingface-hub 到 0.36.2
3. `pip install "numpy<2.3,>=1.24" --force-reinstall` — 降级 numpy 到 2.2.6

## 最终环境
- numpy: 2.2.6
- numba: 0.61.2
- huggingface-hub: 0.36.2
- transformers: 4.57.6
- datasets: 2.14.4
- pyarrow: 25.0.0
- vllm: 0.1.dev11642+g1172122f8
- torch: 2.9.0+cu130