"""Generate publication-quality figures from all CAKE-Serve experiments.

Outputs to results/figures/*.png (150 dpi).

Figures:
  1. ruler_5way.png      — grouped bar: 5-way ablation @ 4K/8K/16K
  2. gamma_ablation.png  — line: CAKE_cake_layer quality vs gamma (4K/8K)
  3. offline_perf.png    — line: 32K batch latency (FullKV/CAKE25/CAKE50)
  4. serving_throughput.png — bar: online tok/s + TTFT p50 @ QPS=2.0
  5. serving_latency.png — bar: online TTFT p50/p99 @ QPS=2.0
  6. retention.png       — bar: physical retention ratio 8K/16K/32K
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = "results/figures"
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {
    "FullKV": "#4C72B0", "SnapKV_uniform": "#DD8452",
    "SnapKV_cluster": "#55A868", "CAKE_uniform": "#C44E52",
    "CAKE_cake_layer": "#8172B3",
    "CAKE_25": "#8172B3", "CAKE_50": "#937860",
}
HATCH = {"FullKV": "", "SnapKV_uniform": "//", "SnapKV_cluster": "\\\\",
         "CAKE_uniform": "xx", "CAKE_cake_layer": ""}

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ── 1. RULER 5-way ───────────────────────────────────────────────────
def fig_ruler_5way():
    lengths = [4096, 8192, 16384]
    cfgs = ["FullKV", "SnapKV_uniform", "SnapKV_cluster",
            "CAKE_uniform", "CAKE_cake_layer"]
    data = {}
    for L in lengths:
        data[L] = {}
        for cfg in cfgs:
            p = f"results/raw/day20_ruler_{L}/ruler_{cfg}.json"
            if os.path.exists(p):
                data[L][cfg] = load_json(p)["macro_avg"]

    x = np.arange(len(lengths))
    width = 0.16
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, cfg in enumerate(cfgs):
        vals = [data[L].get(cfg, np.nan) for L in lengths]
        bars = ax.bar(x + (i - 2) * width, vals, width,
                      label=cfg.replace("_", " "), color=COLORS[cfg],
                      hatch=HATCH[cfg], edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.008,
                        f"{v:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{L//1024}K" for L in lengths])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Macro Average Accuracy")
    ax.set_xlabel("Context Length (tokens)")
    ax.set_title("RULER 5-Way Ablation (13 tasks × 50 samples)")
    ax.legend(ncol=2, framealpha=0.9)
    out = f"{FIG_DIR}/ruler_5way.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ── 2. Gamma ablation ────────────────────────────────────────────────
def fig_gamma():
    gammas = [1, 50, 200]
    lengths = [4096, 8192]
    data = {L: [] for L in lengths}
    for L in lengths:
        for g in gammas:
            p1 = f"results/raw/day21_paper_params/{L}/CAKE_cake_layer_g{g}_t11.6_t20.4.json"
            p2 = f"results/raw/day21_paper_params/{L}/CAKE_cake_layer_g{g}.json"
            p = p1 if os.path.exists(p1) else p2
            data[L].append(load_json(p)["macro_avg"])

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for L, marker in zip(lengths, ["o", "s"]):
        ax.plot(gammas, data[L], marker=marker, linewidth=2,
                label=f"{L//1024}K context", color=COLORS["CAKE_cake_layer"])
        for g, v in zip(gammas, data[L]):
            ax.annotate(f"{v:.4f}", (g, v), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9)

    ax.set_xscale("log")
    ax.set_xticks(gammas)
    ax.set_xticklabels([str(g) for g in gammas])
    ax.set_xlabel("CAKE scorer gamma (tau1=1.6, tau2=0.4 fixed)")
    ax.set_ylabel("Macro Average Accuracy")
    ax.set_title("Gamma Sensitivity: CAKE + cake_layer (RULER)")
    ax.legend()
    out = f"{FIG_DIR}/gamma_ablation.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ── 3. Offline batch perf ────────────────────────────────────────────
def fig_offline_perf():
    path = "results/raw/day14_perf/perf_results_32768.json"
    with open(path) as f:
        data = json.load(f)
    batches = [1, 4, 8, 10]
    cfgs = ["FullKV", "CAKE_25", "CAKE_50"]
    series = {c: [] for c in cfgs}
    for c in cfgs:
        results = data.get(c, {}).get("results", {})
        for b in batches:
            item = results.get(str(b), {})
            if isinstance(item, dict):
                series[c].append(item.get("median_s", np.nan))
            else:
                series[c].append(np.nan)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for c in cfgs:
        ax.plot(batches, series[c], marker="o", linewidth=2,
                label=c.replace("_", " "), color=COLORS[c])
    ax.set_xticks(batches)
    ax.set_xlabel("Batch Size (32K context, 128 output tokens)")
    ax.set_ylabel("Median Latency (s)")
    ax.set_title("32K Offline Batch Performance (A6000, Llama-3.1-8B)")
    ax.legend()
    out = f"{FIG_DIR}/offline_perf.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ── 4. Serving throughput ────────────────────────────────────────────
def fig_serving():
    cfgs = ["FullKV", "CAKE_25", "CAKE_50"]
    qps = 2.0
    tok_s, ttft50, ttft99, req_s = [], [], [], []
    for c in cfgs:
        ratio = 1.0 if c == "FullKV" else 0.25 if c == "CAKE_25" else 0.5
        p = f"results/raw/day17_serving/{c}_r{ratio}_qps{qps}.json"
        r = load_json(p)
        tok_s.append(r["total_token_throughput"])
        ttft50.append(r["median_ttft_ms"] / 1000)
        ttft99.append(r["p99_ttft_ms"] / 1000)
        req_s.append(r["request_throughput"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # throughput
    ax = axes[0]
    bars = ax.bar(cfgs, tok_s, color=[COLORS[c] for c in cfgs],
                  edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, tok_s):
        ax.text(b.get_x() + b.get_width() / 2, v + 40, f"{v:.0f}",
                ha="center", fontsize=10)
    ax.set_ylabel("Token Throughput (tok/s)")
    ax.set_title("Online Serving: Token Throughput @ QPS=2.0")
    ax.set_ylim(0, max(tok_s) * 1.15)

    # TTFT
    ax = axes[1]
    x = np.arange(len(cfgs))
    w = 0.35
    b1 = ax.bar(x - w / 2, ttft50, w, label="TTFT p50",
                color="#4C72B0", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w / 2, ttft99, w, label="TTFT p99",
                color="#C44E52", edgecolor="black", linewidth=0.5)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 4,
                f"{b.get_height():.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(cfgs)
    ax.set_ylabel("Seconds")
    ax.set_title("Online Serving: TTFT @ QPS=2.0 (16K ctx)")
    ax.legend()
    ax.set_ylim(0, max(ttft99) * 1.15)

    fig.suptitle("Online Serving (16K input / 128 output, 200 requests)",
                 fontsize=13)
    out = f"{FIG_DIR}/serving.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ── 5. Retention ─────────────────────────────────────────────────────
def fig_retention():
    lengths = [8192, 16384, 32768]
    cfgs = ["FullKV", "CAKE_25", "CAKE_50"]
    data = {c: [] for c in cfgs}
    for L in lengths:
        p = f"results/raw/day10_memory/memory_results_{L}.json"
        r = load_json(p)
        results = r.get("results", []) if isinstance(r, dict) else r
        for c in cfgs:
            v = None
            for item in results:
                if item.get("config") == c:
                    v = item.get("effective_physical_ratio")
                    break
            data[c].append(v if v is not None else np.nan)

    x = np.arange(len(lengths))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, c in enumerate(cfgs):
        bars = ax.bar(x + (i - 1) * width, data[c], width, label=c,
                      color=COLORS[c], edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, data[c]):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                        f"{v:.3f}", ha="center", fontsize=9)

    ax.axhline(0.25, color="gray", linestyle=":", linewidth=1)
    ax.text(2.4, 0.258, "requested 25%", color="gray", fontsize=9, ha="right")
    ax.axhline(0.50, color="gray", linestyle=":", linewidth=1)
    ax.text(2.4, 0.508, "requested 50%", color="gray", fontsize=9, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{L//1024}K" for L in lengths])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Effective Physical Retention Ratio")
    ax.set_xlabel("Context Length (tokens)")
    ax.set_title("KV Cache Physical Retention (unified config)")
    ax.legend()
    out = f"{FIG_DIR}/retention.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    print("Generating figures...")
    fig_ruler_5way()
    fig_gamma()
    fig_offline_perf()
    fig_serving()
    fig_retention()
    print(f"\nAll figures saved to {FIG_DIR}/")
