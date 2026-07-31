#!/bin/bash
cd /home/lixinze/cake-serve
export CUDA_VISIBLE_DEVICES=1
PY=/home/lixinze/miniconda3/envs/cake-serve/bin/python
for i in 3 4; do
  echo "[GPU1] Starting config $i at $(date)"
  $PY scripts/_ruler_5way.py --length 4096 --config_idx $i
  echo "[GPU1] Config $i done at $(date)"
done
echo "[GPU1] ALL DONE"
