# Day 15: 消融与负结果

**日期**: 2026-07-25  
**状态**: ✅ 完成

## 消融实验

### 1. window_size 消融

| window | mean_pref | 范围 | 时间 | Spearman vs 32 |
|:------:|:---------:|:----:|:---:|:--------------:|
| 16 | 491 | [91, 970] | 4.5ms | 0.988 |
| 32 | 741 | [150, 1647] | 0.55ms | 1.000 |
| 64 | 1249 | [270, 2763] | 0.65ms | 0.988 |

**结论**: window_size 影响偏好值大小，但几乎不影响层间排序（Spearman > 0.97）。window=32 是计算效率与精度平衡的最佳选择。

### 2. gamma 消融

| gamma | mean_pref | Spearman vs 0.0 |
|:----:|:---------:|:---------------:|
| 0.0 | 740.57 | 1.000 |
| 0.5 | 740.57 | 1.000 |
| 1.0 | 740.57 | 1.000 |
| 2.0 | 740.57 | 1.000 |

**结论**: gamma 不影响层偏好分数（偏好公式不含 gamma）。gamma 仅影响 token 级别评分。

## 负结果

### 1. CAKE Scorer 计算开销
- SnapKV: 0.112ms/层
- CAKE: **0.576ms/层**
- 开销比: **5.13x**
- 含义: 32 层模型每步 prefill 增加约 18ms 额外开销。短 prompt 下开销占比显著。

### 2. 短 prompt 压缩收益有限
- 1K prompt 的 token 分数平均 0.00126
- 2K prompt 的 token 分数平均 0.00041
- 短 prompt 的 KV 占用本身很小，压缩收益有限

### 3. 10% retention 质量下降
- 极低 retention 下层偏好波动变大，预算分配被量化噪声主导
- 建议适用区间 25%~50%

### 4. page_group_size 权衡
- page_group_size=1: 最细粒度，精度最高，元数据开销最大
- page_group_size=4: 粗粒度，精度略低，元数据开销小
- 此为 Tangram Paged KV Cache 的固有权衡

## 面试准备

**面试官问：CAKE 的计算开销比 SnapKV 大多少？**
> CAKE 的计算开销约为 SnapKV 的 5 倍（0.58ms vs 0.11ms/层）。这是因为 CAKE 需要计算 attention 的 mean 和 var 两个统计量，而 SnapKV 只需要 mean。对于 32 层模型，每步 prefill 增加约 18ms，在长上下文场景下可以接受，但在短 prompt 下可能不值得。

**面试官问：window_size 和 gamma 对结果有多大影响？**
> window_size 影响偏好值大小但不影响层间排序（Spearman > 0.97），说明 CAKE 的层偏好对窗口大小不敏感。gamma 不影响层偏好（只影响 token 评分），所以默认值 1.0 是安全的。

## 产出
- `results/raw/day15_ablation/ablation_results.json` — 完整结果
- `scripts/ablation_day15.py` — 测试脚本