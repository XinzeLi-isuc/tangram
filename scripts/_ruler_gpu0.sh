#!/bin/bash
cd /home/lixinze/cake-serve
export CUDA_VISIBLE_DEVICES=0
PY=/home/lixinze/miniconda3/envs/cake-serve/bin/python
for i in 0 1 2; do
  echo "[GPU0] Starting config $i at $(date)"
  $PY scripts/_ruler_5way.py --length 4096 --config_idx $i
  echo "[GPU0] Config $i done at $(date)"
done
echo "[GPU0] ALL DONE"
