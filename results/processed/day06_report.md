# Day 6: SnapKV vs CAKE Scorer 对比验证

**日期**: 2026-07-24  
**状态**: ✅ 完成

---

## 今日完成

### 1. 对比测试脚本

**文件**: `scripts/compare_snapkv_cake.py`

对比 7 个维度:
1. Score shape/dtype/NaN/Inf
2. Head-wise 分布
3. Top-k 重叠
4. 计算耗时
5. 可重复性
6. 预算分配
7. 汇总

### 2. 实验结果

| 检查项 | 结果 |
|--------|------|
| Shape | SnapKV [8, 4096], CAKE [8, 4064] ✅ |
| NaN/Inf | 无 ✅ |
| Top-k 重叠 (r=0.5) | 67-70% |
| 计算耗时 | SnapKV 0.25ms, CAKE 0.43ms (1.69x) |
| 可重复性 | 完美匹配 ✅ |
| 预算分配 | 总和精确匹配 ✅ |

### 3. 实验报告补全

已为 Day 1-6 创建完整的实验报告:
- `results/raw/experiments/day01_env_setup.md`
- `results/raw/experiments/day02_ruler_baseline.md`
- `results/raw/experiments/day03_cake_reference.md`
- `results/raw/experiments/day04_cake_algorithm.md`
- `results/raw/experiments/day05_cake_scorer.md`
- `results/raw/experiments/day06_scorer_comparison.md`

---

## 明天计划（第7天）

传递 layer preference:
1. 定义 `CakeScoreOutput`
2. 扩展 `receive_score()`
3. 请求隔离测试