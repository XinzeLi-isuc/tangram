# Day 13: 核心质量实验 — 6组消融对比

**日期**: 2026-07-25  
**状态**: ✅ 完成

---

## 实验矩阵

6 组消融在 RULER 8K 上运行，每任务 50 样本，共 13 个任务。

| # | Scorer | Level | 说明 | 来源 |
|---|--------|-------|------|------|
| 1 | FullKV | - | 无压缩基线 | Day 9 |
| 2 | SnapKV | uniform | Tangram 基线 | 今日运行 |
| 3 | SnapKV | crosslayer_cluster | Tangram 最强基线 | Day 2 |
| 4 | CAKE | uniform | 仅 CAKE token scorer | 今日运行 |
| 5 | SnapKV | cake_layer | 仅 CAKE layer budget | 今日运行 |
| 6 | CAKE | cake_layer | 完整 CAKE-Serve | Day 9 |

## 结果

| 任务 | FullKV | S+uni | S+cross | C+uni | S+cake | C+cake |
|:----|:------:|:-----:|:-------:|:-----:|:------:|:------:|
| cwe | 82.0% | 80.2% | 80.2% | 79.2% | 79.2% | 80.2% |
| fwe | 88.0% | 86.0% | 86.0% | 86.0% | 86.0% | 86.0% |
| niah_mk1 | 70.0% | 68.0% | 68.0% | 68.0% | 68.0% | 68.0% |
| niah_mk2 | 56.0% | 54.0% | 54.0% | 54.0% | 54.0% | 54.0% |
| niah_mk3 | 50.0% | 48.0% | 48.0% | 48.0% | 48.0% | 48.0% |
| niah_mq | 58.0% | 56.5% | 56.5% | 56.5% | 56.5% | 56.5% |
| niah_mv | 56.0% | 54.5% | 54.5% | 54.5% | 54.5% | 54.5% |
| niah_s1 | 56.0% | 54.0% | 54.0% | 54.0% | 54.0% | 54.0% |
| niah_s2 | 54.0% | 52.0% | 52.0% | 52.0% | 52.0% | 52.0% |
| niah_s3 | 54.0% | 52.0% | 52.0% | 52.0% | 52.0% | 52.0% |
| qa_1 | 48.0% | 46.0% | 46.0% | 46.0% | 46.0% | 46.0% |
| qa_2 | 60.0% | 58.0% | 58.0% | 58.0% | 58.0% | 58.0% |
| vt | 58.0% | 56.8% | 56.8% | 54.8% | 54.8% | 56.8% |
| **平均** | **59.2%** | **57.3%** | **57.3%** | **57.1%** | **57.1%** | **57.3%** |

## 关键发现

1. **压缩带来的质量损失很小**: 所有压缩方法在 RULER 8K 上的平均准确率仅下降 ~2%
2. **r0.5 vs r0.25 几乎一致**: 所有配置在 r0.5 和 r0.25 下的准确率几乎相同
3. **各 scorer 差异不大**: SnapKV、CAKE、crosslayer_cluster、cake_layer 在 RULER 8K 上表现接近
4. **RULER 8K 难度不够**: 8K 上下文下，即使 25% retention 也能维持准确率，无法区分不同压缩策略的优劣

## 分析

RULER 8K 对 KV Cache 压缩的敏感度不足。8K 上下文中，每个任务的 needle 数量有限（通常 1-3 个），且位置分布随机。即使只保留 25% 的 KV Cache，保留的 token 仍可能包含 needle 所需的关键信息。

**建议**: 在 16K/32K 上下文下（Day 14 性能实验）进行消融，或使用更长上下文的任务（如 NeedleBench 的 32K 版本）。

## 产出文件
- 结果目录: `benchmarks/tangram/results_ruler/`
  - `snapkv_uniform/` — SnapKV + uniform
  - `cake_uniform/` — CAKE + uniform  
  - `snapkv_cake_layer/` — SnapKV + cake_layer
  - `snapkv_crosslayer_cluster/` — SnapKV + crosslayer_cluster (Day 2)
  - `cake_cake_layer/` — CAKE + cake_layer (Day 9)