#!/usr/bin/env python3
"""Run RULER configs 3,4 sequentially on GPU1 @ 16K."""
import subprocess, sys, os
os.chdir("/home/lixinze/cake-serve")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
PY = "/home/lixinze/miniconda3/envs/cake-serve/bin/python"
for i in [3, 4]:
    print(f"[GPU1] Config {i} @ 16K start", flush=True)
    subprocess.run([PY, "scripts/_ruler_5way.py", "--length", "16384", "--config_idx", str(i)], check=True)
    print(f"[GPU1] Config {i} @ 16K done", flush=True)
print("[GPU1] ALL DONE")
