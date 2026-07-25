# Day 9: CAKE-Serve MVP 验收 — RULER 8K 完整结果

**日期**: 2026-07-24  
**状态**: ✅ 完成（MVP 里程碑达成）

## 今日完成

### 1. 代码修改
- `benchmark_ruler.sh` — 添加 cake scorer 支持
- `bench_common.py` — 添加 cake/cake_layer 到 choices
- `ruler_local.py` — 修复 datasets 兼容性，改用 pyarrow 直接读取缓存

### 2. RULER 8K 完整结果

**CAKE-Serve vs SnapKV 对比**:

| 指标 | SnapKV r=0.5 | CAKE-Serve r=0.5 | SnapKV r=0.25 | CAKE-Serve r=0.25 |
|------|:-----------:|:----------------:|:-------------:|:-----------------:|
| 平均准确率 | 87.2% | **87.3%** | 75.2% | **77.5%** |
| 最佳任务 | vt 99.6% | niah_single_1/3 **100%** | vt 99.6% | vt 99.6% |
| 最差任务 | qa_2 60% | qa_2 64% | niah_multikey_3 20% | niah_multikey_3 40% |

### 3. 验收标准

| 标准 | 状态 |
|------|------|
| CAKE-Serve 完整管道通过 | ✅ |
| ratio=0.5 质量 ≥ SnapKV | ✅ 87.3% vs 87.2% |
| ratio=0.25 质量 ≥ SnapKV | ✅ 77.5% vs 75.2% (+2.3%) |
| RULER 完整跑通 | ✅ 26 个 JSON 文件 |
| 结果可复现 | ✅ |

### 面试准备

**面试官问：CAKE-Serve 的主要成果是什么？**
> 在 50% retention 下与 SnapKV 持平（87.3%），在 25% retention 下提升 2.3 个百分点（77.5% vs 75.2%）。更重要的是，CAKE-Serve 提供了 SnapKV 不具备的层偏好能力——它知道哪些层需要更多预算、哪些层可以更激进压缩。这种能力在显存受限场景下可以直接转化为系统收益（更高并发、更低延迟）。

**面试官问：为什么有些任务 CAKE 更好，有些 SnapKV 更好？**
> CAKE 的时序方差评分会保留那些"重要性随时间波动"的 token。这在 niah_multikey_3（多 key 检索）上效果显著（+8%~+20%），因为多 key 场景下 token 的重要性随时间变化很大。但在 niah_multivalue（多值检索）上反而下降（-11%），因为多值场景需要的是"稳定关注"的 token，而不是"波动"的 token。

**自建项目可能遇到的难题：**
1. **RULER 数据集加载**：`datasets` 库与 Python 3.12 不兼容，需要直接使用 pyarrow 读取缓存
2. **benchmark_ruler.sh 不支持自定义 scorer**：需要修改 case 分支和 choices 列表
3. **GPU 显存竞争**：同时运行 benchmark 和验证脚本会 OOM，需要用不同 GPU 或顺序执行

---

## 明天计划（第10天）

物理显存验证：
- FullKV vs Uniform vs CAKE-Serve 的 KV Page 数对比
- 32K 下观察明确物理内存差异