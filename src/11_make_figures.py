"""Paper figures (vector PDF/EPS, 174 mm wide, 8 pt text, no in-figure titles)."""
import os
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(os.environ.get("FLOOD_DATA_DIR", "data"))
ORDER = ["none", "q_drop30", "q_drop50", "q_delay3", "q_delay7", "f_delay1", "f_delay3", "q_blocks", "combined"]
LAB = ["none", "D30", "D50", "L3", "L7", "F1", "F3", "OUT", "ALL"]
W_IN = 174 / 25.4
STYLE = {   # method: (legend label, colour, marker, linestyle)
    "CQR_complete": ("Calibrated on complete data", "#d95f02", "s", "--"),
    "CQR_mondrian": ("Missingness bins", "#1b9e77", "D", "-"),
    "CQR_mondrian_region": ("Bins x region", "#7570b3", "^", ":"),
    "CQR_mondrian_basin": ("Bins x basin", "#e6ab02", "o", "-."),
    "CQR_mondrian_flow": ("Bins x forecast level", "#386cb0", "v", "-"),
}
FIG1 = ["CQR_complete", "CQR_mondrian", "CQR_mondrian_basin"]
MODELS = {"trained_complete": "Model trained on complete data", "trained_mixed": "Model trained with masked inputs"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.1, "lines.markersize": 3.5,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.eps")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)
    print("saved", name)


def letter(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom", ha="right")


def legend_below(fig, ax, ncol):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))


def fig_coverage(b, out):
    fig, axs = plt.subplots(2, 2, figsize=(W_IN, 4.8), sharey=True)
    k = 0
    for i, lead in enumerate([1, 3]):
        for j, model in enumerate(MODELS):
            ax = axs[i, j]
            for m, (lab, c, mk, ls) in {k: STYLE[k] for k in FIG1}.items():
                g = b[(b.lead == lead) & (b.model == model) & (b.method == m)].set_index("scenario").reindex(ORDER)
                if g.coverage.isna().all():
                    continue
                ax.errorbar(range(9), g.coverage, yerr=[g.coverage - g.coverage_lo, g.coverage_hi - g.coverage], color=c,
                            marker=mk, ls=ls, capsize=1.5, label=lab)
            ax.axhline(0.9, color="k", lw=0.6, alpha=0.6)
            ax.set_xticks(range(9))
            ax.set_xticklabels(LAB)
            ax.set_ylim(0.45, 1.0)
            if i == 0:
                ax.set_title(MODELS[model], fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{lead}-day lead\nEmpirical coverage")
            letter(ax, "abcd"[k])
            k += 1
    legend_below(fig, axs[0, 0], 3)
    return fig


def fig_width(b, out):
    fig, axs = plt.subplots(1, 2, figsize=(W_IN, 2.7), sharey=True)
    for i, lead in enumerate([1, 3]):
        ax = axs[i]
        for model, c, mk in (("trained_complete", "#1b9e77", "D"), ("trained_mixed", "#005a32", "o")):
            g = b[(b.lead == lead) & (b.model == model) & (b.method == "CQR_mondrian")].set_index("scenario").reindex(ORDER)
            ax.plot(range(9), np.exp(g.width), color=c, marker=mk, label=MODELS[model])
        ax.set_xticks(range(9))
        ax.set_xticklabels(LAB)
        ax.set_title(f"{lead}-day lead", fontsize=8)
        if i == 0:
            ax.set_ylabel("Upper / lower bound ratio")
        letter(ax, "ab"[i])
    legend_below(fig, axs[0], 2)
    return fig


def fig_spread(b, out):
    fig, axs = plt.subplots(1, 2, figsize=(W_IN, 3.0), sharey=True)
    for i, model in enumerate(MODELS):
        ax = axs[i]
        ms = [m for m in FIG1 if ((b.model == model) & (b.method == m) & (b.lead == 1)).any()]
        for r, m in enumerate(ms):
            lab, c, mk, ls = STYLE[m]
            g = b[(b.lead == 1) & (b.model == model) & (b.method == m)].set_index("scenario").reindex(ORDER)
            x = np.arange(9) + (r - (len(ms) - 1) / 2) * 0.16
            ax.vlines(x, g.basin_p5, g.basin_p95, color=c, lw=1.6, label=lab)
            ax.plot(x, g.basin_p50, ls="", marker=mk, color=c, ms=3)
        ax.axhline(0.9, color="k", lw=0.6, alpha=0.6)
        ax.set_xticks(range(9))
        ax.set_xticklabels(LAB)
        ax.set_ylim(0.3, 1.02)
        ax.set_title(MODELS[model], fontsize=8)
        if i == 0:
            ax.set_ylabel("Per-basin coverage (5th-95th pct.)")
        letter(ax, "ab"[i])
    legend_below(fig, axs[0], 4)
    return fig


def fig_warnings(w, lead):
    w = w[(w.source == "all") & (w.alpha_w == 0.1) & (w.model.isin(MODELS))]
    kinds = {"U_bins": ("Upper bound, missingness bins", "#1b9e77", "D", "-"),
             "U_complete": ("Upper bound, complete-data margin", "#d95f02", "s", "--"),
             "P_bins": ("Point forecast, missingness bins", "#7570b3", "^", ":")}
    fig, axs = plt.subplots(2, 2, figsize=(W_IN, 4.8), sharex=True)
    for j, model in enumerate(MODELS):
        for i, (col, yl) in enumerate((("recall", "Recall of high-flow days"), ("false_alarm", "Warnings on low-flow days"))):
            ax = axs[i, j]
            for k, (lab, c, mk, ls) in kinds.items():
                g = w[(w.lead == lead) & (w.model == model) & (w.kind == k)].groupby("scenario")[col].mean().reindex(ORDER)
                ax.plot(range(9), g, color=c, marker=mk, ls=ls, label=lab)
            if i == 0:
                ax.axhline(0.9, color="k", lw=0.6, alpha=0.6)
                ax.set_title(MODELS[model], fontsize=8)
                ax.set_ylim(0.5, 1.0)
            else:
                ax.set_ylim(0, 0.7)
            if j == 0:
                ax.set_ylabel(yl)
            ax.set_xticks(range(9))
            ax.set_xticklabels(LAB)
            letter(ax, "abcd"[2 * i + j])
    legend_below(fig, axs[0, 0], 3)
    return fig


def fig_holdout(bs, bh):
    fig, axs = plt.subplots(1, 2, figsize=(W_IN, 2.9), sharey=True)
    for j, model in enumerate(MODELS):
        ax = axs[j]
        for lab, d, meth, c, mk, ls in (("Seen basins, missingness bins", bs, "CQR_mondrian", "#1b9e77", "D", "-"),
                                        ("Unseen basins, missingness bins", bh, "CQR_mondrian", "#386cb0", "o", "-"),
                                        ("Unseen basins, complete-data calibration", bh, "CQR_complete", "#d95f02", "s", "--")):
            g = d[(d.lead == 1) & (d.model == model) & (d.method == meth)].set_index("scenario").reindex(ORDER)
            ax.errorbar(range(9), g.coverage, yerr=[g.coverage - g.coverage_lo, g.coverage_hi - g.coverage], color=c, marker=mk,
                        ls=ls, capsize=1.5, label=lab)
        ax.axhline(0.9, color="k", lw=0.6, alpha=0.6)
        ax.set_xticks(range(9))
        ax.set_xticklabels(LAB)
        ax.set_ylim(0.45, 1.0)
        ax.set_title(MODELS[model], fontsize=8)
        if j == 0:
            ax.set_ylabel("Empirical coverage")
        letter(ax, "ab"[j])
    legend_below(fig, axs[0], 3)
    return fig


def fig_predhigh(b):
    fig, axs = plt.subplots(2, 2, figsize=(W_IN, 4.8), sharey=True)
    k = 0
    for i, lead in enumerate([1, 3]):
        for j, model in enumerate(MODELS):
            ax = axs[i, j]
            for m in ("CQR_complete", "CQR_mondrian", "CQR_mondrian_flow"):
                lab, c, mk, ls = STYLE[m]
                g = b[(b.lead == lead) & (b.model == model) & (b.method == m)].set_index("scenario").reindex(ORDER)
                ax.errorbar(range(9), g.predhigh, yerr=[g.predhigh - g.predhigh_lo, g.predhigh_hi - g.predhigh], color=c, marker=mk,
                            ls=ls, capsize=1.5, label=lab)
            ax.axhline(0.9, color="k", lw=0.6, alpha=0.6)
            ax.set_xticks(range(9))
            ax.set_xticklabels(LAB)
            ax.set_ylim(0.3, 1.0)
            if i == 0:
                ax.set_title(MODELS[model], fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{lead}-day lead\nCoverage, forecast in top 5%")
            letter(ax, "abcd"[k])
            k += 1
    legend_below(fig, axs[0, 0], 3)
    return fig


def fig_transfer(base, out):
    p = base / "processed"
    files = {"Calibrated on LamaH": ("trackA_runs_lamah_to_gb.csv", "CQR_mondrian", "#7570b3", "^", ":"),
             "Calibrated on CAMELS-GB": ("trackA_runs_lamah_to_gb_cal.csv", "CQR_mondrian", "#1b9e77", "D", "-"),
             "Calibrated on LamaH, complete data": ("trackA_runs_lamah_to_gb.csv", "CQR_complete", "#d95f02", "s", "--")}
    if not all((p / f[0]).exists() for f in files.values()):
        print("transfer files missing - Fig3 skipped")
        return None
    fig, axs = plt.subplots(1, 2, figsize=(W_IN, 2.9), sharey=True)
    for j, model in enumerate(MODELS):
        ax = axs[j]
        for lab, (f, meth, c, mk, ls) in files.items():
            d = pd.read_csv(p / f)
            d = d[(d.source == "all") & (d.model == model) & (d.method == meth)]
            g = d.groupby(["seed", "scenario"]).coverage.mean().unstack("scenario").reindex(columns=ORDER)
            ax.errorbar(range(9), g.mean(), yerr=g.std(), color=c, marker=mk, ls=ls, capsize=1.5, label=lab)
        ax.axhline(0.9, color="k", lw=0.6, alpha=0.6)
        ax.set_xticks(range(9))
        ax.set_xticklabels(LAB)
        ax.set_ylim(0.45, 1.0)
        ax.set_title(MODELS[model], fontsize=8)
        if j == 0:
            ax.set_ylabel("Coverage on CAMELS-GB")
        letter(ax, "ab"[j])
    legend_below(fig, axs[0], 3)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(BASE))
    args = ap.parse_args()
    base = Path(args.base)
    out = base.parent / "figures"
    out.mkdir(exist_ok=True)
    b = pd.read_csv(base / "processed" / "trackA_bootstrap_summary.csv")
    save(fig_coverage(b, out), out, "Fig1")
    save(fig_width(b, out), out, "Fig2")
    save(fig_spread(b, out), out, "Fig3")
    f = fig_transfer(base, out)
    if f is not None:
        save(f, out, "Fig4")
    p = base / "processed"
    wf = next(iter(sorted(p.glob("trackA_warnings_*.csv"))), None)
    if wf is not None and "holdout" not in wf.name:
        w = pd.read_csv(wf)
        save(fig_warnings(w, 1), out, "Fig5")
        if (w.lead == 3).any():
            save(fig_warnings(w, 3), out, "FigS1")
    if (p / "trackA_bootstrap_summary_holdout.csv").exists():
        save(fig_holdout(b, pd.read_csv(p / "trackA_bootstrap_summary_holdout.csv")), out, "Fig6")
    if "predhigh_lo" in b.columns and (b.method == "CQR_mondrian_flow").any():
        save(fig_predhigh(b), out, "Fig7")


if __name__ == "__main__":
    main()
