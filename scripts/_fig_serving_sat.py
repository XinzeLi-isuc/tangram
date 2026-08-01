"""Plot throughput-TTFT/TPOT saturation curves from day22_serving_sat.

Usage: python scripts/_fig_serving_sat.py
"""
import json, os, statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = "results/figures"
os.makedirs(FIG_DIR, exist_ok=True)
DATA_DIR = "results/raw/day22_serving_sat"

COLORS = {"FullKV": "#4C72B0", "CAKE_25": "#8172B3", "CAKE_50": "#937860"}
MARKERS = {"FullKV": "o", "CAKE_25": "s", "CAKE_50": "^"}

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def main():
    summary_path = os.path.join(DATA_DIR, "summary.json")
    if not os.path.exists(summary_path):
        print(f"NO DATA: {summary_path}")
        return
    with open(summary_path) as f:
        summary = json.load(f)

    lengths = summary.get("lengths", [])
    configs = summary.get("configs", [])
    results = summary["results"]

    for length in lengths:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for name in configs:
            if name not in results[str(length)]:
                continue
            qps_list, toks, ttft50, tpot50 = [], [], [], []
            for qps_str in sorted(results[str(length)][name], key=float):
                reps = results[str(length)][name][qps_str]
                qps = float(qps_str)
                qps_list.append(qps)
                toks.append(statistics.median(
                    r.get("total_token_throughput", 0) for r in reps))
                ttft50.append(statistics.median(
                    r.get("median_ttft_ms", 0) for r in reps) / 1000)
                tpot50.append(statistics.median(
                    r.get("median_tpot_ms", 0) for r in reps))
            ax = axes[0]
            ax.plot(qps_list, toks, marker=MARKERS[name], linewidth=2,
                    label=name.replace("_", " "), color=COLORS[name])
            ax = axes[1]
            ax.plot(qps_list, ttft50, marker=MARKERS[name], linewidth=2,
                    label=f"{name.replace('_',' ')} TTFT p50", color=COLORS[name])
            ax.plot(qps_list, tpot50, marker=MARKERS[name], linewidth=1.5,
                    linestyle="--", label=f"{name.replace('_',' ')} TPOT p50",
                    color=COLORS[name], alpha=0.7)

        ax = axes[0]
        ax.set_xlabel("Offered QPS")
        ax.set_ylabel("Token Throughput (tok/s)")
        ax.set_title(f"Throughput vs Offered Load ({length//1024}K ctx)")
        ax.legend()

        ax = axes[1]
        ax.set_xlabel("Offered QPS")
        ax.set_ylabel("Latency (s; TTFT solid / TPOT dashed)")
        ax.set_title(f"Latency vs Offered Load ({length//1024}K ctx)")
        ax.legend(fontsize=9)

        out = f"{FIG_DIR}/serving_sat_{length}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
