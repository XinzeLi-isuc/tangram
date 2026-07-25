# Day 2: Tangram RULER Baseline — SnapKV + crosslayer_cluster

**日期**: 2026-07-24  
**工作时段**: 20:45 ~ 21:25  
**状态**: ✅ 完成

---

## 今日完成

### 1. Tangram benchmark 结构探索
- 阅读 `benchmarks/tangram/benchmark_ruler.sh` — 主调脚本，支持 SCORER/LEVEL/LENGTHS/RATIOS 环境变量
- 阅读 `benchmarks/tangram/benchmark_ruler.py` — 13 个 RULER 任务，支持 50 样本/任务
- 阅读 `benchmarks/tangram/bench_common.py` — `build_llm()` 引擎构造 + `add_compression_args()`
- RULER 任务: niah_single_{1,2,3}, niah_multikey_{1,2,3}, niah_multivalue, niah_multiquery, vt, cwe, fwe, qa_1, qa_2
- RULER 长度: 4096, 8192, 16384

### 2. 环境依赖修复
- 安装 `cloudpickle` — benchmark_ruler.py 依赖
- 安装 `datasets` + `pyarrow` — RULER 数据加载依赖，降级 `huggingface-hub` 到 0.36.2
- 降级 `numpy` 到 2.2.6 — 解决 numba 兼容性问题
- 详细日志见 `results/raw/day02_logs/env_fixes.md`

### 3. RULER 8K Baseline 结果

**配置**: SnapKV + crosslayer_cluster, Llama-3.1-8B-Instruct, 50 samples/task, WINDOW_SIZE=32, FLOOR_MIN=0

| Task | ratio=0.5 | ratio=0.25 |
|------|----------:|-----------:|
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

### 4. 关键观察
- **ratio=0.5**: 大部分任务保持 80%+ 准确率，niah_multikey_3 最低 (72%)
- **ratio=0.25**: 精度下降明显，niah_multikey_3 崩溃到 20%，但 vt 保持不变 (99.6%)
- **vt (visual tracking)**: 几乎不受压缩影响（99.6% 在两个比例下）
- **qa_2**: 最困难的任务，两个比例都只有 60% 左右
- 每个任务 50 样本耗时约 65 秒，总共约 28 分钟完成两个比例
- 原始 JSON 结果保存在 `benchmarks/tangram/results_ruler/snapkv_crosslayer_cluster/`

### 5. 环境快照更新
- `results/raw/environment.txt` 已更新

---

## 代码变更

**本次无代码修改** — 仅运行 Tangram 原有 benchmark 建立基线。

## 环境变更
| 依赖 | 变更 |
|------|------|
| cloudpickle | 新增安装 |
| datasets 2.14.4 | 新增安装 |
| pyarrow 25.0.0 | 新增安装 |
| huggingface-hub 0.36.2 | 降级 (从 1.24.0) |
| numpy 2.2.6 | 降级 (从 2.5.1) |

## 明天计划（第3天）

CAKE 官方复现：
1. 安装 CAKE 官方环境（`cake-ref` conda env）
2. 克隆 CAKE 官方仓库
3. 跑 Llama-3.1-8B，导出层偏好、预算、token score
4. 与 Tangram RULER 结果对照

---

## 问题与风险

| 问题 | 状态 |
|------|------|
| 环境依赖兼容性 | ✅ 已修复 |
| RULER 8K baseline 完成 | ✅ |
| 数据可用作 CAKE 对比基线 | ✅ |