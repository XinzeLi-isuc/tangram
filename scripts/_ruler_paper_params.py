#!/usr/bin/env python3
"""RULER with paper CAKE scorer params (gamma=200, tau1=1.6, tau2=0.4).

Runs configs 3 (CAKE_uniform) and 4 (CAKE_cake_layer) for given length.
Usage: CAKE_GAMMA=200 CAKE_TAU1=1.6 CAKE_TAU2=0.4 CUDA_VISIBLE_DEVICES=2 \
       python scripts/_ruler_paper_params.py --length 4096
"""
import subprocess, sys, os
os.chdir("/home/lixinze/cake-serve")
PY = "/home/lixinze/miniconda3/envs/cake-serve/bin/python"

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=4096, choices=[4096, 8192, 16384])
    args = ap.parse_args()

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    g = os.environ.get("CAKE_GAMMA", "1")
    t1 = os.environ.get("CAKE_TAU1", "1")
    t2 = os.environ.get("CAKE_TAU2", "1")
    print(f"[GPU{g}] RULER {args.length} paper params "
          f"gamma={g} tau1={t1} tau2={t2}", flush=True)

    for i in [3, 4]:  # CAKE_uniform, CAKE_cake_layer
        print(f"[GPU{g}] Config {i} @ {args.length} start", flush=True)
        subprocess.run(
            [PY, "scripts/_ruler_5way.py", "--length", str(args.length),
             "--config_idx", str(i)],
            check=True,
        )
        print(f"[GPU{g}] Config {i} @ {args.length} done", flush=True)
    print(f"[GPU{g}] ALL DONE")


if __name__ == "__main__":
    main()
