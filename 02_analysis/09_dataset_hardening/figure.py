from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import data as D
from _common import style as S

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def build(fig):
    summary = D.read_json(RES / "hardening_summary.json")
    coverage = pd.read_csv(RES / "field_coverage.csv")

    gs = fig.add_gridspec(6, 1,
                          height_ratios=[1.95, 0.34, 2.05, 0.40, 2.05, 0.55])
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[2])
    sub = gs[4].subgridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.42)
    axC = fig.add_subplot(sub[0])
    axD = fig.add_subplot(sub[1])
    for spacer in (1, 3, 5):
        fig.add_subplot(gs[spacer]).axis("off")

    hs = summary["headline_sensitivity"]
    n_raw = 9944
    steps = [
        ("All CPSC\nrecalls", n_raw, S.GRAY_3),
        ("Textile\nterms match", 3251, S.GRAY_2),
        ("v1\ndataset", hs["n_v1"], S.BLUE),
        ("v2\ndataset", hs["n_v2"], S.GREEN),
        ("v2 minus\nreview queue",
         hs["n_v2"] - summary.get("n_likely_nontextile_flagged", 0), S.AMBER),
    ]
    xs = np.arange(len(steps))
    vals = [v for _, v, _ in steps]
    cols = [c for _, _, c in steps]
    axA.bar(xs, vals, width=0.62, color=cols, edgecolor=S.SURFACE, linewidth=1.2,
            zorder=2)
    for x, v in zip(xs, vals):
        axA.annotate(f"{v:,}", (x, v), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=8.5, color=S.INK, fontweight="semibold")
    axA.set_xticks(xs)
    axA.set_xticklabels([s for s, _, _ in steps], fontsize=7.8)
    axA.set_yscale("log")
    axA.set_ylim(400, n_raw * 2.2)
    axA.set_ylabel("Records (log scale)")
    S.tidy(axA)
    S.panel_title(axA, "A  From the CPSC database to the analysis dataset",
                  "Log scale; the last bar is a scope decision still open")

    fields = [c for c in coverage.columns if c not in ("period", "n")]
    periods = coverage.period.tolist()
    M = coverage[fields].to_numpy(dtype=float).T
    im = axB.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axB.set_xticks(range(len(periods)))
    axB.set_xticklabels([f"{p}\nn={int(n)}" for p, n in
                         zip(periods, coverage.n)], fontsize=7.8)
    axB.set_yticks(range(len(fields)))
    axB.set_yticklabels([f.capitalize() for f in fields], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            axB.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7.6,
                     color="white" if v > 0.55 else S.INK,
                     fontweight="semibold")
    axB.set_xticks(np.arange(-0.5, len(periods), 1), minor=True)
    axB.set_yticks(np.arange(-0.5, len(fields), 1), minor=True)
    axB.grid(which="minor", color=S.SURFACE, linewidth=1.6)
    axB.tick_params(which="both", length=0)
    axB.spines[["top", "right", "left", "bottom"]].set_visible(False)
    S.panel_title(axB, "B  Which variables can carry an analysis, and when",
                  "Share of records with the field populated")

    arch = pd.Series(summary["archetype_counts"]).sort_values(ascending=True)
    arch = arch[arch >= 4].tail(10)
    boundary = {"nightgown": S.PINK, "robe": S.PINK, "loungewear": S.PINK,
                "wearable_blanket": S.PINK, "pajama_set": S.GREEN,
                "sleepwear_generic": S.GREEN, "underwear_base": S.GREEN}
    cols = [boundary.get(k, S.GRAY_2) for k in arch.index]
    axC.barh(range(len(arch)), arch.values, color=cols, height=0.68,
             edgecolor=S.SURFACE, linewidth=0.8, zorder=2)
    axC.set_yticks(range(len(arch)))
    axC.set_yticklabels([k.replace("_", " ") for k in arch.index], fontsize=7.6)
    for i, v in enumerate(arch.values):
        axC.annotate(f"{v}", (v, i), xytext=(3, 0), textcoords="offset points",
                     va="center", fontsize=7.4, color=S.INK_2)
    axC.set_xlabel("Recalls")
    S.tidy(axC, grid="x")
    S.panel_title(axC, "C  Garment archetype",
                  "Pink cannot be tight-fitting; green can")

    bars = [
        ("v1 dataset", hs["flammability_share_2020_25_v1"], S.GRAY_2),
        ("v2 (non-textiles out)", hs["flammability_share_2020_25_v2"], S.BLUE),
        ("v2 minus review queue",
         summary.get("flammability_share_2020_25_if_queue_dropped",
                     hs["flammability_share_2020_25_v2"]), S.AMBER),
    ]
    ys = np.arange(len(bars))
    axD.barh(ys, [b[1] for b in bars], color=[b[2] for b in bars], height=0.55,
             edgecolor=S.SURFACE, linewidth=0.8, zorder=2)
    for y, (_, v, _) in zip(ys, bars):
        axD.annotate(f"{v:.1%}", (v, y), xytext=(4, 0),
                     textcoords="offset points", va="center", fontsize=8,
                     color=S.INK, fontweight="semibold")
    axD.set_yticks(ys)
    axD.set_yticklabels([b[0] for b in bars], fontsize=7.6)
    axD.set_xlim(0, 0.95)
    axD.set_xlabel("Share of recalls")
    S.tidy(axD, grid="x")
    S.panel_title(axD, "D  Sensitivity",
                  "2020-25 flammability share")

    S.freeze_free_text(fig)
    n_q = summary.get("n_likely_nontextile_flagged", 0)
    delta = abs(summary.get("flammability_share_2020_25_if_queue_dropped", 0)
                - hs["flammability_share_2020_25_v2"])
    note = (
        f"CPSC recall database (saferproducts.gov), retrieved 2026-08. "
        f"v1 n = {hs['n_v1']}; v2 n = {hs['n_v2']} after removing "
        f"{summary['n_removed_false_positive']} non-textile records.\n"
        f"A further {n_q} records are queued for a human scope decision "
        f"(results/nontextile_review_queue.csv);\n"
        f"excluding them moves the 2020-25 flammability share by "
        f"{delta:.1%} points. Description text requires the raw CPSC extract.")
    S.source_note(fig, note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.parse_args()
    out = S.save_figure(build, HERE / "figure_dataset_hardening", height_in=9.0)
    print("[09] figure ->", ", ".join(Path(p).name for p in out))


if __name__ == "__main__":
    main()
