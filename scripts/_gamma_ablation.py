#!/usr/bin/env python3
"""Gamma ablation: run CAKE_cake_layer (config 4) with gamma in {1, 50, 200}.

Usage: CUDA_VISIBLE_DEVICES=2 python scripts/_gamma_ablation.py --length 4096
"""
import subprocess, sys, os
os.chdir("/home/lixinze/cake-serve")
PY = "/home/lixinze/miniconda3/envs/cake-serve/bin/python"

GAMMAS = [1, 50, 200]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=4096, choices=[4096, 8192])
    args = ap.parse_args()

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    for g in GAMMAS:
        print(f"[GPU{gpu}] gamma={g} @ {args.length} start", flush=True)
        env = os.environ.copy()
        env["CAKE_GAMMA"] = str(g)
        env["CAKE_TAU1"] = "1.6"
        env["CAKE_TAU2"] = "0.4"
        subprocess.run(
            [PY, "scripts/_ruler_5way.py", "--length", str(args.length),
             "--config_idx", "4"],
            check=True, env=env,
        )
        print(f"[GPU{gpu}] gamma={g} done", flush=True)
    print(f"[GPU{gpu}] ALL DONE")


if __name__ == "__main__":
    main()
