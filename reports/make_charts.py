"""
Lab notebook charts.

    python reports/make_charts.py ingest runs/trial_a_20261006-134953 --label trial_a
    python reports/make_charts.py charts

`ingest` smooths a run's metrics.csv and keeps one row per 500 ticks in
reports/data/<label>.csv (small enough to commit). `charts` redraws every figure
in reports/figures/ from those files. Runs that have not been ingested yet are
skipped, so figures fill in as experiments finish.
"""

from __future__ import annotations

import argparse
import os
import tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")

# Validated categorical order (light surface). Slot n goes to the n-th run in runs.toml.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


# ------------------------------------------------------------------ ingest
def ingest(src: str, label: str, every: int = 500, window: int = 50):
    path = os.path.join(src, "metrics.csv") if os.path.isdir(src) else src
    df = pd.read_csv(path)
    num = df.select_dtypes("number")
    sm = num.rolling(window, min_periods=1, center=True).mean()
    sm["tick"] = df["tick"]
    out = sm[sm["tick"] % every == 0].reset_index(drop=True)
    os.makedirs(DATA, exist_ok=True)
    dst = os.path.join(DATA, f"{label}.csv")
    out.to_csv(dst, index=False, float_format="%.6g")
    print(f"{label}: {len(df)} rows -> {len(out)} rows  ({dst})")


# ------------------------------------------------------------------ charts
def load_runs():
    with open(os.path.join(HERE, "runs.toml"), "rb") as f:
        reg = tomllib.load(f)["run"]
    runs = {}
    for i, r in enumerate(reg):
        p = os.path.join(DATA, f"{r['label']}.csv")
        runs[r["label"]] = {"name": r["name"], "color": SERIES[i % len(SERIES)],
                            "df": pd.read_csv(p) if os.path.exists(p) else None}
    return runs


def style(ax, title, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)


def panels(fname, suptitle, specs, runs, labels, x="generation_mean", xlabel="Generation", need_all=False):
    """specs: list of (column, panel title, y label, scale). One y-axis per panel.
    need_all: a comparison figure is only drawn once every run in it has finished."""
    have = [l for l in labels if runs.get(l, {}).get("df") is not None]
    if not have or (need_all and len(have) < len(labels)):
        print("skip", fname, "(waiting for runs)")
        return
    n = len(specs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), facecolor=SURFACE)
    axes = [axes] if n == 1 else axes
    for ax, (col, title, ylab, scale) in zip(axes, specs):
        for l in have:
            r = runs[l]
            if col in r["df"]:
                ax.plot(r["df"][x], r["df"][col] * scale, color=r["color"], lw=2, label=r["name"])
        style(ax, title, ylab)
        ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(have), frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, fname), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", fname)


def benchmark_chart():
    p = os.path.join(HERE, "..", "bench", "results.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    df = df[(df.status == "ok") & (df.msg == 8)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE)
    for i, (g, d) in enumerate(df.groupby("grid")):
        ax.plot(d.params_per_cell, d.ticks_per_s, color=SERIES[i], lw=2, marker="o", ms=5,
                label=f"{g} x {g} grid")
    ax.set_yscale("log")
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([1000, 2000, 4000, 8000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    style(ax, "Ticks per second vs. network size (message width 8)", "Ticks per second")
    ax.set_xlabel("Weights per cell (maximum network)", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig0_benchmark.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote fig0_benchmark.png")


def charts():
    runs = load_runs()
    benchmark_chart()
    ab = ["trial_a", "trial_b"]
    panels("fig1_sharing_collapse.png", "The sharing collapse",
           [("transfer_per_cell", "Energy given to neighbors", "per cell per tick", 1),
            ("div_weights", "Genetic diversity (weight genes)", "mean per-gene std", 1),
            ("msg_bandwidth", "Message bandwidth", "mean |message| per cell", 1)], runs, ab)
    panels("fig2_brains.png", "Brain size under scarcity",
           [("active_params_frac", "Active weights", "share of maximum network", 1),
            ("expressed_hidden_frac", "Expressed hidden units", "share switched on", 1)], runs, ab)
    panels("fig3_learning.png", "Learning signals",
           [("eta_mean", "Plasticity rate (genome)", "mean |learning rate|", 1),
            ("weight_drift", "Weight drift from genome", "mean |learned - inherited|", 1)], runs, ab)
    panels("fig4_economy.png", "Energy economy",
           [("occupancy", "Grid occupancy", "share of sites alive", 1),
            ("starved_share", "Deaths by starvation", "share of deaths", 1),
            ("overflow", "Energy lost to full storage", "per tick, whole world", 1)], runs, ab)
    panels("fig5_plasticity_test.png", "Same patchy world, learning on vs. off",
           [("transfer_per_cell", "Energy given to neighbors", "per cell per tick", 1),
            ("eta_mean", "Plasticity rate (genome)", "mean |learning rate|", 1),
            ("starved_share", "Deaths by starvation", "share of deaths", 1),
            ("occupancy", "Grid occupancy", "share of sites alive", 1)],
           runs, ["trial_b", "trial_b_noplast"], need_all=True)
    panels("fig6_plasticity_test_c.png", "Changing world (trial C), learning on vs. off",
           [("transfer_per_cell", "Energy given to neighbors", "per cell per tick", 1),
            ("eta_mean", "Plasticity rate (genome)", "mean |learning rate|", 1),
            ("starved_share", "Deaths by starvation", "share of deaths", 1),
            ("population", "Population", "living cells", 1)],
           runs, ["trial_c", "trial_c_noplast"], need_all=True)
    panels("fig7_bonds.png", "Bonds (Layer 2) in a static and a changing world",
           [("transfer_per_cell", "Open gifts to neighbors", "per cell per tick", 1),
            ("bonded_frac", "Cells with at least one bond", "share of cells", 1),
            ("in_groups_5plus", "Cells in groups of 5 or more", "share of cells", 1),
            ("bond_flow_per_cell", "Energy shared inside groups", "per cell per tick", 1)],
           runs, ["trial_b", "trial_c", "trial_b_bonds", "trial_c_bonds"], need_all=True)
    panels("fig8_birth_bonds.png", "Static world: no bonds, any-neighbor bonds, birth-only bonds",
           [("transfer_per_cell", "Open gifts to neighbors", "per cell per tick", 1),
            ("bonded_frac", "Cells with at least one bond", "share of cells", 1),
            ("starved_share", "Deaths by starvation", "share of deaths", 1),
            ("population", "Population", "living cells", 1)],
           runs, ["trial_b", "trial_b_bonds", "trial_b_birthbonds"], need_all=True)
    panels("fig9_exposure.png", "Static world: does a physical reason to stay bonded help?",
           [("transfer_per_cell", "Open gifts to neighbors", "per cell per tick", 1),
            ("bonded_frac", "Cells with at least one bond", "share of cells", 1),
            ("starved_share", "Deaths by starvation", "share of deaths", 1),
            ("population", "Population", "living cells", 1)],
           runs, ["trial_b", "trial_b_birthbonds", "trial_b_huddle"], need_all=True)
    panels("fig10_changing_bonds.png", "Changing world: any-neighbor vs. birth-only bonds",
           [("transfer_per_cell", "Open gifts to neighbors", "per cell per tick", 1),
            ("bonded_frac", "Cells with at least one bond", "share of cells", 1),
            ("starved_share", "Deaths by starvation", "share of deaths", 1),
            ("population", "Population", "living cells", 1)],
           runs, ["trial_c", "trial_c_bonds", "trial_c_birthbonds"], need_all=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ingest")
    a.add_argument("src")
    a.add_argument("--label", required=True)
    sub.add_parser("charts")
    args = ap.parse_args()
    if args.cmd == "ingest":
        ingest(args.src, args.label)
    else:
        charts()
