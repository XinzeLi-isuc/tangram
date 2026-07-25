# 实验报告：Day 10 — 物理显存验证

## 实验目的
验证 CAKE-Serve 在不同上下文长度和压缩比例下，能否在 A6000 48GB 上稳定运行而无 OOM。确认压缩后的物理 KV Page 回收机制有效。

## 环境
- GPU: NVIDIA RTX A6000 48GB
- 模型: Llama-3.1-8B-Instruct (bfloat16)
- 理论 KV Cache: 32层 × 8KV heads × 128dim × 2(K+V) × 2bytes = 128 KiB/token

## 复现步骤
```bash
cd ~/cake-serve
CUDA_VISIBLE_DEVICES=0 python scripts/test_memory.py
```

## 结果
15/15 配置全部成功运行，无 OOM。

## 产出文件
- 结果: `results/raw/day10_memory/memory_results.json`