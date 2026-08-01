"""Paired bootstrap + McNemar tests on RULER per-sample records.

Reads per_sample.json from a day20_ruler_<len>/ directory and compares
configs pairwise (CAKE_cake_layer vs SnapKV_cluster, CAKE_cake_layer vs
CAKE_uniform, etc.).

Usage: python scripts/_ruler_stats.py --length 16384
"""
import argparse, json, os
import numpy as np

rng = np.random.default_rng(42)
N_BOOT = 10000


def load_per_sample(length):
    path = f"results/raw/day20_ruler_{length}/per_sample.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing — run _ruler_5way.py first")
    with open(path) as f:
        return json.load(f)


def build_aligned(per_sample, cfg_a, cfg_b):
    """Return (a_correct, b_correct) aligned by (task, sample_id)."""
    a_rows = per_sample.get(cfg_a, {})
    b_rows = per_sample.get(cfg_b, {})
    pairs = []
    for task in a_rows:
        by_id = {r["sample_id"]: r["correct"] for r in a_rows[task]}
        for r in b_rows.get(task, []):
            if r["sample_id"] in by_id:
                pairs.append((by_id[r["sample_id"]], r["correct"]))
    if not pairs:
        raise ValueError(f"No aligned pairs between {cfg_a} and {cfg_b}")
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    return a, b


def paired_bootstrap(a, b, n_boot=N_BOOT):
    """95% CI for mean difference (a - b) via paired bootstrap."""
    n = len(a)
    diff = a - b
    obs = diff.mean()
    samples = rng.choice(diff, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    p_lt0 = (samples < 0).mean()
    p_gt0 = (samples > 0).mean()
    p_value = 2 * min(p_lt0, p_gt0)
    return obs, lo, hi, p_value


def mcnemar(a, b):
    """McNemar exact test on paired binary outcomes.

    b01: a correct, b wrong; b10: a wrong, b correct.
    p = 2 * min(b01, b10) via binomial / exact.
    """
    b01 = int(((a == 1) & (b == 0)).sum())  # a wins
    b10 = int(((a == 0) & (b == 1)).sum())  # b wins
    n_disc = b01 + b10
    if n_disc == 0:
        return 1.0, b01, b10
    # exact binomial two-sided
    from math import comb
    p = 0.0
    for k in range(min(b01, b10), n_disc + 1):
        p += 2 * comb(n_disc, k) * 0.5 ** n_disc
    return min(p, 1.0), b01, b10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=16384, choices=[4096, 8192, 16384])
    args = ap.parse_args()

    per_sample = load_per_sample(args.length)
    configs = list(per_sample.keys())
    print(f"Configs with per-sample data: {configs}")
    print(f"Length: {args.length}")

    pairs = [
        ("CAKE_cake_layer", "SnapKV_cluster"),
        ("CAKE_cake_layer", "SnapKV_uniform"),
        ("CAKE_cake_layer", "CAKE_uniform"),
    ]
    for a, b in pairs:
        if a not in per_sample or b not in per_sample:
            print(f"\n  SKIP {a} vs {b}: missing config")
            continue
        aa, bb = build_aligned(per_sample, a, b)
        n = len(aa)
        acc_a, acc_b = aa.mean(), bb.mean()
        obs, lo, hi, p_boot = paired_bootstrap(aa, bb)
        p_mcn, b01, b10 = mcnemar(aa, bb)

        print(f"\n  {a} vs {b} (n={n})")
        print(f"    acc: {a}={acc_a:.4f}  {b}={acc_b:.4f}  diff={acc_a-acc_b:+.4f}")
        print(f"    paired bootstrap 95% CI: [{lo:+.4f}, {hi:+.4f}]  p={p_boot:.4f}")
        print(f"    McNemar: a-win={b01}, b-win={b10}, p={p_mcn:.4f} "
              f"{'***' if p_mcn < 0.001 else '**' if p_mcn < 0.01 else '*' if p_mcn < 0.05 else 'ns'}")


if __name__ == "__main__":
    main()
