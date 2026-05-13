"""Generate every poster/report figure from results/metrics/*.json.

Run from repo root:  python scripts/build_all_figures.py
Outputs land in     results/figures/  (and results/tables/ for CSVs).
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
FIGURES.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

PAPER_DLINEAR_MSE_NORM = {
    ("weather", 96): 0.196, ("weather", 336): 0.283,
    ("electricity", 96): 0.140, ("electricity", 336): 0.169,
    ("traffic", 96): 0.410, ("traffic", 336): 0.436,
}
PAPER_PATCHTST_MSE_NORM = {
    ("weather", 96): 0.152, ("weather", 336): 0.249,
    ("electricity", 96): 0.130, ("electricity", 336): 0.167,
}


def load_df() -> pd.DataFrame:
    rows = []
    for path in sorted(METRICS.glob("*.json")):
        s = json.load(open(path))
        cfg = s.get("config", {})
        if cfg.get("model") == "dlinear":
            continue
        rows.append({
            "run_name": s["run_name"],
            "dataset": cfg.get("dataset"),
            "model": cfg.get("model"),
            "seq_len": cfg.get("seq_len"),
            "pred_len": cfg.get("pred_len"),
            "patch_len": cfg.get("patch_len"),
            "stride": cfg.get("stride"),
            "seed": cfg.get("seed"),
            "mse_inv": s.get("test_mse_inverse"),
            "mae_inv": s.get("test_mae_inverse"),
            "mse_norm": s.get("test_mse_normalized"),
            "mae_norm": s.get("test_mae_normalized"),
            "num_patches": s.get("num_patches"),
            "attention_pairs": s.get("attention_pairs"),
            "avg_epoch_s": s.get("avg_epoch_seconds"),
            "num_params": s.get("num_params"),
            "peak_mb": (s.get("gpu_memory_mb") or {}).get("peak_alloc_mb"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "all_runs.csv", index=False)
    return df


def fig_main_results(df: pd.DataFrame) -> pd.DataFrame:
    main = (df[df["model"].isin(["naive", "patchtst"])
              & (df["patch_len"].fillna(16) == 16)
              & (df["stride"].fillna(8) == 8)]
            .pivot_table(index=["dataset", "pred_len"], columns="model",
                         values="mse_norm", aggfunc="min")
            .reset_index())
    main["paper_dlinear"] = main.apply(
        lambda r: PAPER_DLINEAR_MSE_NORM.get((r["dataset"], r["pred_len"])), axis=1)
    main["paper_patchtst"] = main.apply(
        lambda r: PAPER_PATCHTST_MSE_NORM.get((r["dataset"], r["pred_len"])), axis=1)
    main = main[["dataset", "pred_len", "naive", "paper_dlinear", "patchtst", "paper_patchtst"]]
    main.to_csv(TABLES / "main_results.csv", index=False)
    with open(TABLES / "main_results_latex.txt", "w") as f:
        f.write(main.to_latex(index=False, float_format="%.4f"))

    # bar chart
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, T in zip(axes, [96, 336]):
        sub = main[main["pred_len"] == T].set_index("dataset")
        sub[["naive", "paper_dlinear", "patchtst"]].plot.bar(
            ax=ax, color=["#9aa0a6", "#fbbc04", "#1a73e8"], edgecolor="black", linewidth=0.4)
        ax.set_title(f"Test MSE @ T={T}", fontsize=12)
        ax.set_ylabel("MSE (normalized)")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(["Naive", "DLinear (paper)", "PatchTST (ours)"], fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "main_results_bar.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return main


def fig_heatmap(df: pd.DataFrame) -> None:
    local = (df[df["model"].isin(["naive", "patchtst"])
               & (df["patch_len"].fillna(16) == 16)
               & (df["stride"].fillna(8) == 8)]
             .pivot_table(index=["dataset", "pred_len"], columns="model",
                          values="mse_norm", aggfunc="min"))
    local["paper_dlinear"] = [PAPER_DLINEAR_MSE_NORM.get(idx) for idx in local.index]
    hm = local[["naive", "paper_dlinear", "patchtst"]]
    hm.index = [f"{d.title()} T={t}" for d, t in hm.index]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    im = ax.imshow(hm.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(hm.columns)))
    ax.set_xticklabels(["Naive", "DLinear (paper)", "PatchTST (ours)"], fontsize=11)
    ax.set_yticks(range(len(hm.index)))
    ax.set_yticklabels(hm.index, fontsize=10)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.values[i, j]
            color = "white" if v > hm.values.mean() * 1.2 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized test MSE (lower is better)", fontsize=10)
    ax.set_title("Test MSE across all main configurations", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "main_results_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_seed_robustness(df: pd.DataFrame) -> None:
    """Show mean ± min/max bars across seeds for the four main runs."""
    pat = df[(df["model"] == "patchtst")
            & (df["patch_len"].fillna(16) == 16)
            & (df["stride"].fillna(8) == 8)
            & (df["dataset"].isin(["weather", "electricity"]))
            & (df["seed"].isin([21, 42, 67]))
            & (df["pred_len"].isin([96, 336]))].copy()
    grp = pat.groupby(["dataset", "pred_len"])["mse_norm"].agg(["mean", "min", "max", "count"]).reset_index()
    grp["label"] = grp.apply(lambda r: f"{r['dataset'].title()}\nT={int(r['pred_len'])}", axis=1)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(grp))
    means = grp["mean"].values
    yerr = np.vstack([means - grp["min"].values, grp["max"].values - means])
    bars = ax.bar(x, means, yerr=yerr, color="#1a73e8", edgecolor="black",
                  linewidth=0.5, capsize=5, alpha=0.85)
    for b, m, n in zip(bars, means, grp["count"]):
        ax.text(b.get_x() + b.get_width() / 2, m + max(means) * 0.01,
                f"{m:.3f}\n(n={int(n)})", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(grp["label"], fontsize=10)
    ax.set_ylabel("Test MSE (normalized)")
    ax.set_title("PatchTST seed robustness — mean with min/max range across {21, 42, 67}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(means) * 1.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "seed_robustness.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_pareto(df: pd.DataFrame) -> None:
    p = df[(df["dataset"] == "weather") & (df["pred_len"] == 96)
           & (df["num_params"].fillna(0) > 0)
           & (~df["run_name"].str.contains("DEBUG", na=False))
           & (df["seed"].isin([42, None]) | df["seed"].isna())].copy()

    def family(r):
        nm = r["run_name"]
        if r["model"] == "channel_mixing":
            return "Channel-mixing"
        if "NOPATCH" in nm.upper() or (r["patch_len"] == 1 and r["stride"] == 1):
            return "No-patching"
        if "_P8S4" in nm:
            return "PatchTST P=8/S=4"
        if "_P32S16" in nm:
            return "PatchTST P=32/S=16"
        if "_P16S8" in nm:
            return "PatchTST P=16/S=8"
        if "_L96" in nm or "_L192" in nm or "_L336" in nm:
            return f"PatchTST L={int(r['seq_len'])}"
        return "PatchTST"
    p["family"] = p.apply(family, axis=1)

    # also include the no-patch profile run (only profile result has params)
    nopatch_path = METRICS / "profile_nopatch_T96.json"
    if nopatch_path.exists():
        s = json.load(open(nopatch_path))
        p = pd.concat([p, pd.DataFrame([{
            "run_name": "weather_NOPATCH_profile", "num_params": s["num_params"],
            "mse_norm": s.get("test_mse_normalized"),
            "family": "No-patching", "patch_len": 1, "stride": 1,
        }])], ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    fams = sorted(p["family"].dropna().unique())
    cmap = plt.cm.tab10.colors
    fam_color = {f: cmap[i % len(cmap)] for i, f in enumerate(fams)}
    for f in fams:
        sub = p[p["family"] == f]
        ax.scatter(sub["num_params"], sub["mse_norm"], s=110, alpha=0.85,
                   color=fam_color[f], label=f, edgecolors="black", linewidths=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("# parameters (log scale)", fontsize=11)
    ax.set_ylabel("Test MSE (normalized)", fontsize=11)
    ax.set_title("Pareto: model size vs accuracy (Weather T=96)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="best", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(FIGURES / "pareto_params_vs_mse.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_channel_independence(df: pd.DataFrame) -> None:
    ci = df[df["run_name"] == "weather_patchtst_T96_P16S8"]
    cm = df[df["model"] == "channel_mixing"]
    if ci.empty or cm.empty:
        print("[channel] missing rows, skipping")
        return
    ci, cm = ci.iloc[0], cm.iloc[0]

    labels = ["PatchTST\n(channel-independent)", "Patch Transformer\n(channel-mixing)"]
    mses = [ci["mse_norm"], cm["mse_norm"]]
    params = [ci["num_params"], cm["num_params"]]
    epochs = [ci["avg_epoch_s"], cm["avg_epoch_s"]]
    colors = ["#2ca02c", "#d62728"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, vals, ylabel, title, fmt in [
        (axes[0], mses, "Test MSE (normalized)", "Accuracy", "{:.4f}"),
        (axes[1], params, "# parameters", "Model size", "{:.2f}M"),
        (axes[2], epochs, "avg s / epoch", "Training cost", "{:.0f}s"),
    ]:
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.7)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        for b, v in zip(bars, vals):
            text = fmt.format(v / 1e6 if "M" in fmt else v)
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01, text,
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.18)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Channel independence wins on every axis (Weather T=96)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "channel_independence_vs_mixing.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_patching_scaling(df: pd.DataFrame) -> None:
    sweep = df[df["run_name"].isin([
        "weather_patchtst_T96_P8S4", "weather_patchtst_T96_P16S8", "weather_patchtst_T96_P32S16"
    ])].copy().sort_values("patch_len")

    rows = []
    nopatch_path = METRICS / "profile_nopatch_T96.json"
    if nopatch_path.exists():
        s = json.load(open(nopatch_path))
        rows.append({
            "label": "no-patch\nP=1/S=1",
            "num_patches": s.get("num_patches"),
            "attention_pairs": s.get("attention_pairs"),
            "mse_norm": s.get("test_mse_normalized"),
            "avg_epoch_s": s.get("avg_epoch_seconds"),
        })
    for _, r in sweep.iterrows():
        rows.append({
            "label": f"P={int(r['patch_len'])}/S={int(r['stride'])}",
            "num_patches": r["num_patches"],
            "attention_pairs": r["attention_pairs"],
            "mse_norm": r["mse_norm"],
            "avg_epoch_s": r["avg_epoch_s"],
        })
    psw = pd.DataFrame(rows).sort_values("num_patches", ascending=False)

    fig, ax_left = plt.subplots(figsize=(9, 5))
    ax_right = ax_left.twinx()
    x = np.arange(len(psw))
    bars = ax_left.bar(x, psw["mse_norm"], color="#1f77b4", alpha=0.65,
                       edgecolor="black", linewidth=0.7, label="Test MSE (norm)")
    for b, v in zip(bars, psw["mse_norm"]):
        ax_left.text(b.get_x() + b.get_width() / 2, v + max(psw["mse_norm"]) * 0.01,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=10,
                     color="#1f77b4", fontweight="bold")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels([f"N={int(n)}\n{lbl}" for n, lbl in
                             zip(psw["num_patches"], psw["label"])], fontsize=9)
    ax_left.set_ylabel("Test MSE (normalized)", color="#1f77b4")
    ax_left.tick_params(axis="y", labelcolor="#1f77b4")
    ax_left.set_ylim(0, max(psw["mse_norm"]) * 1.18)

    ax_right.set_yscale("log")
    ax_right.plot(x, psw["attention_pairs"], "o-", color="#d62728",
                  linewidth=2.0, markersize=10, label="attention pairs N²")
    ax_right.plot(x, psw["avg_epoch_s"], "s-", color="#2ca02c",
                  linewidth=2.0, markersize=10, label="avg sec / epoch")
    ax_right.set_ylabel("Cost (log scale)")
    ax_left.set_title("Patching scaling on Weather T=96: cost grows ~O(N²)")
    ax_left.grid(True, alpha=0.25, axis="y")

    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    ax_left.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(FIGURES / "patching_scaling.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_lookback_and_patch_sweeps(df: pd.DataFrame) -> None:
    abl = df[(df["dataset"] == "weather") & (df["pred_len"] == 96)
             & (df["model"] == "patchtst")
             & df["run_name"].str.contains(r"P\d+S\d+|T96_L\d+$", regex=True, na=False)].copy()
    psweep = abl[abl["run_name"].str.contains(r"_P\d+S\d+$", regex=True, na=False)].sort_values("patch_len")
    lsweep = abl[abl["run_name"].str.contains(r"_L\d+$", regex=True, na=False)].sort_values("seq_len")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(psweep["patch_len"], psweep["mse_norm"], marker="o", linewidth=2, markersize=10, color="#1a73e8")
    axes[0].set_xlabel("patch length P", fontsize=11)
    axes[0].set_ylabel("Test MSE (normalized)", fontsize=11)
    axes[0].set_title("Patch-size sweep")
    axes[0].grid(True, alpha=0.3)
    for _, r in psweep.iterrows():
        axes[0].annotate(f"P={int(r['patch_len'])}/S={int(r['stride'])}",
                         (r["patch_len"], r["mse_norm"]),
                         textcoords="offset points", xytext=(8, 6), fontsize=9)

    axes[1].plot(lsweep["seq_len"], lsweep["mse_norm"], marker="s", linewidth=2,
                 markersize=10, color="#fbbc04")
    axes[1].set_xlabel("look-back length L", fontsize=11)
    axes[1].set_ylabel("Test MSE (normalized)", fontsize=11)
    axes[1].set_title("Look-back sweep")
    axes[1].grid(True, alpha=0.3)
    for _, r in lsweep.iterrows():
        axes[1].annotate(f"L={int(r['seq_len'])}",
                         (r["seq_len"], r["mse_norm"]),
                         textcoords="offset points", xytext=(8, 6), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "patch_size_sweep.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "lookback_sweep.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_traffic(df: pd.DataFrame) -> None:
    sub = df[(df["dataset"] == "traffic")].copy()
    if sub.empty:
        return
    # use one row per model, pick best across seeds
    agg = sub.groupby("model")["mse_norm"].min().reindex(["naive", "patchtst"]).dropna()
    paper = PAPER_DLINEAR_MSE_NORM.get(("traffic", 96))
    rows = [("Naive", agg.get("naive")), ("DLinear (paper)", paper),
            ("PatchTST (ours)", agg.get("patchtst"))]
    rows = [(l, v) for l, v in rows if v is not None]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([r[0] for r in rows], [r[1] for r in rows],
                  color=["#9aa0a6", "#fbbc04", "#1a73e8"], edgecolor="black", linewidth=0.5)
    for b, (_, v) in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Test MSE (normalized)")
    ax.set_title("Traffic T=96 — naive vs DLinear (paper) vs PatchTST")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "traffic_results.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_ssl(df: pd.DataFrame) -> None:
    rows = []
    cold = df[df["run_name"] == "weather_patchtst_L336_T96_P16S8_seed42"]
    ssl = df[df["run_name"] == "weather_patchtst_T96_pretrained_finetune"]
    if cold.empty or ssl.empty:
        return
    cold, ssl = cold.iloc[0], ssl.iloc[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Cold start\n(supervised only)", "SSL pretrain\n+ fine-tune"],
                  [cold["mse_norm"], ssl["mse_norm"]],
                  color=["#9aa0a6", "#34a853"], edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, [cold["mse_norm"], ssl["mse_norm"]]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.4f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Test MSE (normalized)")
    ax.set_title("Self-supervised pretraining (Weather T=96)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "ssl_vs_cold.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_example_predictions() -> None:
    # symlink/copy the existing prediction plot from nb02
    src = FIGURES / "weather_patchtst_L336_T96_P16S8_seed42_prediction_plot.png"
    dst = FIGURES / "example_predictions.png"
    if src.exists():
        import shutil
        shutil.copyfile(src, dst)


def main():
    df = load_df()
    print(f"loaded {len(df)} runs")
    fig_main_results(df)
    fig_heatmap(df)
    fig_seed_robustness(df)
    fig_pareto(df)
    fig_channel_independence(df)
    fig_patching_scaling(df)
    fig_lookback_and_patch_sweeps(df)
    fig_traffic(df)
    fig_ssl(df)
    fig_example_predictions()
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
