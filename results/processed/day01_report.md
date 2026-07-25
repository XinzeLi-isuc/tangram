# CAKE-Serve 第1天日报：环境搭建与 FullKV Baseline

**日期**: 2026-07-24  
**工作时段**: 20:12 ~ 20:45  
**状态**: ✅ 完成

---

## 今日完成

### 1. 项目目录搭建

```
cake-serve/
├── vllm/                          # Tangram fork (vLLM)
├── benchmarks/cake_serve/configs/ # CAKE-Serve 专属 benchmark
├── tests/cake_serve/              # CAKE-Serve 单元测试
├── scripts/                       # 工具脚本
├── results/raw/                   # 原始结果
├── results/processed/             # 处理后的数据
├── results/figures/               # 图表
├── third_party/                   # 第三方引用
├── ATTRIBUTION.md                 # (待创建)
└── README.md                      # (待创建)
```

### 2. Tangram 环境安装

| 项目 | 值 |
|------|-----|
| 仓库 | https://github.com/aiha-lab/tangram.git |
| Commit | `1172122f820f5a6bcd152189b9708b9e10dca3e8` |
| Python 环境 | `cake-serve` (conda, Python 3.12) |
| vLLM 版本 | `0.1.dev11642+g1172122f8.precompiled` |
| PyTorch 版本 | `2.9.0+cu130` |
| CUDA 版本 | `13.0` (Driver 580.142) |
| Transformers 版本 | `4.57.6` |
| 安装方式 | `VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto` |
| 编译耗时 | ~8.5 分钟 |

### 3. FullKV Baseline 测试

**模型**: Llama-3.1-8B-Instruct（本地路径，ModelScope 下载）
**模型参数**:
- 32 layers, 32 Q heads, 8 KV heads (GQA)
- hidden_size=4096, head_size=128
- max_position_embeddings=131072
- bfloat16 dtype

**测试 1: FullKV (compression_ratio=1.0, snapkv scorer)**

| 指标 | 值 |
|------|-------|
| 模型加载 | ~4s safetensors (4 shards) |
| CUDA Graph 捕获 | ~5s (51 graphs) |
| Generation | ~3.4s 生成 128 tokens |
| 输出长度 | 661 字符 |
| 吞吐 | ~37.6 tok/s |

**测试 2: SnapKV (compression_ratio=0.5, snapkv scorer, uniform level)**

| 指标 | 值 |
|------|-------|
| 模型加载 | ~4s safetensors |
| Generation | ~3.4s 生成 128 tokens |
| 输出长度 | 661 字符 |
| 吞吐 | ~37.9 tok/s |

> 注意：短 prompt + 短 output 下，FullKV 和 SnapKV 的输出长度相同（661 chars），这是预期的——128 token 生成在短上下文中区别不大。

### 4. 环境快照

已保存到 `results/raw/`：
- `tangram_commit.txt`
- `environment.txt`（含 Python、Torch、vLLM、nvidia-smi、pip freeze）

---

## 代码结构理解

第1天阅读了 Tangram 核心压缩代码，确认了集成架构：

```
压缩轴 2 (scorer / token 评分)
  └─ scorer.py: build_qk_scorer() 工厂
  └─ snapkv.py: SnapKVScorer → [num_kv_heads, T]
  └─ qk_scorer_base.py: QKScorer 基类
  
压缩轴 1 (selection level / 预算分配)
  └─ selection_level.py: make_selection_level() 工厂
  └─ UniformLevel, CrossLayerHeadLevel, PerLayerClusterLevel, ...

压缩控制器
  └─ compressor.py: KVCompressor
  └─ receive_score() → pending_score 累积
  └─ prepare_keep_decision() → 调用 level.compute_counts()
  └─ _make_qk_scorer() → 通过 attention op 分发 scorer

执行器
  └─ executor.py: CompressionExecutor.run_request()
  └─ 物理 KV Page 回收
```

**CAKE 集成点已确认**:
1. `cake.py` — 新增 scorer，返回 `CakeScoreOutput`
2. `scorer.py` — 注册 CakeScorer
3. `selection_level.py` — 新增 `CakeLayerLevel` + `SelectionContext`
4. `compressor.py` — 扩展 `receive_score()` 传递 layer preference

---

## 明天计划（第2天）

1. 运行 Tangram 原有 RULER benchmark 脚本
2. 记录 SnapKV baseline 多比例结果（0.5, 0.25）
3. 熟悉 RULER 数据集格式和评估流程
4. 为后续 CAKE 算法移植做准备

---

## 问题与风险

| 问题 | 状态 |
|------|------|
| Llama-3.1-8B 本地模型可用 | ✅ |
| Tangram 安装成功 | ✅ |
| FullKV 生成正常 | ✅ |
| SnapKV 生成正常 | ✅ |
| 环境快照已保存 | ✅ |