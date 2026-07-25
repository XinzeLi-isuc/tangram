# 实验报告：Day 6 — SnapKV vs CAKE Scorer 对比验证

## 实验目的
在 4K 上下文上对比 SnapKV 和 CAKE 两种 scorer 的评分分布、top-k 重叠、计算耗时和可重复性。

## 环境
- 同 Day 1
- 使用合成 Q/K 数据（无需加载模型）

## 复现步骤
```bash
cd ~/cake-serve
CUDA_VISIBLE_DEVICES=0 \
/home/lixinze/miniconda3/envs/cake-serve/bin/python scripts/compare_snapkv_cake.py
```

## 结果

### 1. Score Shape、Dtype、NaN/Inf
| 指标 | SnapKV | CAKE |
|------|--------|------|
| Shape | [8, 4096] | [8, 4064] |
| Dtype | float32 | float32 |
| NaN | False | False |
| Inf | False | False |

注: CAKE shape 比 SnapKV 少 window_size=32 个位置（CAKE 不评分 observation window 区域）

### 2. Head-wise 分布
- 两个 scorer 的 head 间分布均匀（合成数据）
- SnapKV 均值 0.000315，CAKE 均值 0.000244（尺度不同，但都在同一量级）

### 3. Top-k 重叠

| ratio | head=0 | head=4 | head=7 |
|-------|:------:|:------:|:------:|
| 1.00 | 100.0% | 100.0% | 100.0% |
| 0.50 | 70.5% | 67.9% | 70.4% |
| 0.25 | 56.2% | 50.4% | 55.0% |

分析:
- ratio=1.0 全部保留，重叠 100%
- ratio=0.5 时 67-70% 重叠，说明两种评分策略有显著差异
- ratio=0.25 时降到 50-56%，差异更大
- 差异来自: SnapKV 使用 amax + softmax，CAKE 使用 mean + gamma * var

### 4. 计算耗时

| Scorer | 耗时 (ms) | 相对比 |
|--------|:---------:|:------:|
| SnapKV | 0.25 ± 0.10 | 1.00x |
| CAKE | 0.43 ± 0.06 | 1.69x |

CAKE 慢 1.69x 的原因: 计算 mean 和 var 两个统计量（SnapKV 只计算 mean）

### 5. 可重复性
- 3 次运行 preference 完全一致 (109.351898)
- 结果确定性的

### 6. 预算分配

| ratio | 总预算 | 每层范围 |
|-------|:------:|:--------:|
| 0.50 | 15,872 | [67, 851] |
| 0.25 | 7,936 | [34, 426] |
| 0.10 | 3,174 | [13, 170] |

预算总和精确匹配目标值。

## 结论
- CAKE scorer 通路完整可用
- 与 SnapKV 有 50-70% 的 top-k 重叠（差异合理，评分策略不同）
- 计算开销约 1.69x（0.43ms 对 0.25ms），在预填充阶段可接受
- 可重复性完美

## 产出文件
- 对比脚本: `scripts/compare_snapkv_cake.py`
- 实验报告: `results/raw/experiments/day06_scorer_comparison.md`
- 日报: `results/processed/day06_report.md`