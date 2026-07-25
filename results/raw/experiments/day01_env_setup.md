# 实验报告：Day 1 — 环境搭建与 FullKV/SnapKV Baseline

## 实验目的
搭建 Tangram/vLLM 开发环境，验证 Llama-3.1-8B-Instruct 模型加载和 FullKV/SnapKV 生成通路。

## 环境
- GPU: NVIDIA RTX A6000 48GB
- CUDA: 13.0 (Driver 580.142)
- Conda env: `cake-serve` (Python 3.12)
- 项目目录: `~/cake-serve/`

## 复现步骤

### 1. 克隆 Tangram
```bash
git clone https://github.com/aiha-lab/tangram.git ~/cake-serve
cd ~/cake-serve
git rev-parse HEAD  # 1172122f820f5a6bcd152189b9708b9e10dca3e8
```

### 2. 创建环境并安装
```bash
conda create -n cake-serve python=3.12 -y
conda activate cake-serve
pip install uv
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto
```

### 3. 运行测试
```bash
cd ~/cake-serve
CUDA_VISIBLE_DEVICES=0 python scripts/test_fullkv.py
```

### 4. 保存环境快照
```bash
cd ~/cake-serve
git rev-parse HEAD > results/raw/tangram_commit.txt
python -V > results/raw/environment.txt
python -c "import torch; print(torch.__version__, torch.version.cuda)" >> results/raw/environment.txt
python -c "import vllm; print(vllm.__version__)" >> results/raw/environment.txt
nvidia-smi >> results/raw/environment.txt
pip freeze >> results/raw/environment.txt
```

## 结果

### 环境版本
| 组件 | 版本 |
|------|------|
| vLLM | 0.1.dev11642+g1172122f8 |
| PyTorch | 2.9.0+cu130 |
| Transformers | 4.57.6 |
| CUDA | 13.0 |

### 模型
- 模型: Llama-3.1-8B-Instruct (本地路径: `/home/lixinze/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct`)
- 参数: 32 layers, 32 Q heads, 8 KV heads (GQA), head_dim=128
- 精度: bfloat16

### 测试结果
| 配置 | 生成时间 | 吞吐 | 输出长度 |
|------|---------|------|---------|
| FullKV (ratio=1.0) | ~3.4s | 37.6 tok/s | 661 chars |
| SnapKV (ratio=0.5) | ~3.4s | 37.9 tok/s | 661 chars |

## 产出文件
- 环境快照: `results/raw/environment.txt`
- Tangram commit: `results/raw/tangram_commit.txt`
- 日报: `results/processed/day01_report.md`