#!/usr/bin/env python3
"""Run RULER configs 0,1,2 sequentially on GPU0."""
import subprocess, sys, os
os.chdir("/home/lixinze/cake-serve")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PY = "/home/lixinze/miniconda3/envs/cake-serve/bin/python"
for i in [0, 1, 2]:
    print(f"[GPU0] Config {i} start", flush=True)
    subprocess.run([PY, "scripts/_ruler_5way.py", "--length", "4096", "--config_idx", str(i)], check=True)
    print(f"[GPU0] Config {i} done", flush=True)
print("[GPU0] ALL DONE")
