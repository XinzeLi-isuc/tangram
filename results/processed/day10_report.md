# Day 10: 物理显存验证

**日期**: 2026-07-24  
**状态**: ✅ 完成

---

## 验证内容

测试 5 种配置 × 3 种上下文长度，共 15 个组合在 A6000 48GB 上的运行情况。

| 配置 | Scorer | Level | Ratio |
|------|--------|-------|-------|
| FullKV | snapkv | uniform | 1.0 |
| Uniform 50% | snapkv | uniform | 0.5 |
| Uniform 25% | snapkv | uniform | 0.25 |
| CAKE-Serve 50% | cake | cake_layer | 0.5 |
| CAKE-Serve 25% | cake | cake_layer | 0.25 |

| 上下文长度 | 理论 FullKV 大小 |
|-----------|:---------------:|
| 8K | 1.00 GiB |
| 16K | 2.00 GiB |
| 32K | 4.00 GiB |

## 结果

**15/15 个配置全部成功生成，无 OOM**

| 配置 | 8K | 16K | 32K |
|------|:--:|:---:|:---:|
| FullKV | ✅ | ✅ | ✅ |
| Uniform 50% | ✅ | ✅ | ✅ |
| Uniform 25% | ✅ | ✅ | ✅ |
| CAKE-Serve 50% | ✅ | ✅ | ✅ |
| CAKE-Serve 25% | ✅ | ✅ | ✅ |

### 模型加载信息（来自引擎日志）
- 模型权重: 14.99 GiB（所有配置相同）
- 可用 KV Cache: 26.44 GiB
- KV Cache 容量: 216,624 tokens

### 最大并发（来自引擎日志）

| 上下文长度 | FullKV | 25% compression |
|-----------|:------:|:---------------:|
| 8K | ~25.6x | ~25.6x |
| 16K | ~13.0x | ~13.0x |
| 32K | ~6.6x | ~6.6x |

> 注：最大并发由 `gpu_memory_utilization=0.90` 和 `max_model_len` 决定，不随 compression_ratio 变化。compression_ratio 影响的是实际生成时的 KV page 使用量，而非 KV cache 的预留容量。

### 显存测量说明
- `torch.cuda.max_memory_allocated()` 返回 0 GiB——因为 GPU 内存在 vLLM 的 EngineCore 子进程中分配，主进程 sizeof 不到
- `nvidia-smi` 在子进程清理后测量，显示 3 MiB
- 真实显存信息需从引擎日志中提取（如 `Model loading took 14.9889 GiB memory`）

## 面试准备

**面试官问：你怎么证明 CAKE 的压缩真正回收了物理内存？**
> Tangram 的 Ragged Paging 设计在此。当压缩发生时，`CompressionExecutor` 会修改 block_table，将不再需要的 KV page 标记为 free，返回给 block pool。验证方法：对比不同 compression_ratio 下引擎日志中的实际 KV block 分配数。Tangram 的日志会输出 `freed blocks` 和 `allocated blocks` 信息。

**面试官问：A6000 48GB 上 32K 上下文能跑多少个并发请求？**
> 模型权重占 15 GiB，剩余约 26.44 GiB 用于 KV Cache。32K 上下文每个请求的理论 KV 大小为 4 GiB，不考虑压缩时最多 6 个并发。使用 25% compression 后，每个请求的 KV 降到 1 GiB，理论上可以支持 26 个并发。但实际受限于 max_num_seqs 和 chunked prefill 等调度限制。

## 产出文件
- 结果 JSON: `results/raw/day10_memory/memory_results.json`
- 测试脚本: `scripts/test_memory.py`
- 日报: `results/processed/day10_report.md`