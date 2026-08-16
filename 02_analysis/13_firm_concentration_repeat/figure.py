from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import style as S

RES = HERE / "results"

ERA_COLOR = {"pre-2010": S.BLUE, "2010+": S.PINK}
ERA_LABEL = {"pre-2010": "Pre-2010", "2010+": "2010 onwards"}


def load():
    j = json.loads((RES / "firm_summary.json").read_text())
    return {
        "summary": j,
        "lorenz": pd.read_csv(RES / "lorenz_curves.csv"),
        "firms": pd.read_csv(RES / "firm_table.csv"),
        "profiles": pd.read_csv(RES / "cluster_profiles.csv"),
        "sim": pd.read_csv(RES / "similarity_distribution.csv"),
        "pairs": pd.read_csv(RES / "entity_hopping_pairs.csv"),
    }


def cluster_colors(profiles: pd.DataFrame) -> dict:
    order = profiles.sort_values("n_recalls_total", ascending=False).cluster
    return {int(c): S.CAT[i % len(S.CAT)] for i, c in enumerate(order)}


def panel_a(ax, d):
    lor, s = d["lorenz"], d["summary"]
    ax.plot([0, 1], [0, 1], color=S.GRAY_3, linewidth=1.0, linestyle=(0, (4, 3)),
            zorder=1)
    ax.annotate("equal shares", xy=(0.62, 0.62), xytext=(4, -4),
                textcoords="offset points", ha="left", va="top", fontsize=7.2,
                color=S.INK_3, rotation=39, rotation_mode="anchor")
    handles = []
    for era in ["pre-2010", "2010+"]:
        sub = lor[lor.era == era]
        if not len(sub):
            continue
        g = s["gini_by_era"][era]
        ax.plot(sub.frac_firms, sub.frac_recalls, color=ERA_COLOR[era],
                linewidth=2.2, zorder=3)
        handles.append(Line2D(
            [], [], color=ERA_COLOR[era], linewidth=2.2,
            label=f"{ERA_LABEL[era]}   Gini {g['gini']:.2f}, {g['n_firms']} firms"))
    leg = ax.legend(handles=handles, loc="lower right", frameon=False,
                    fontsize=7.2, handlelength=1.3, borderaxespad=0.4,
                    labelspacing=0.5)
    for t in leg.get_texts():
        t.set_color(S.INK)
    dif = s["gini_era_difference"]
    ax.annotate(
        f"Difference {dif['difference']:+.3f}\n"
        f"[{dif['lo95']:.2f}, {dif['hi95']:.2f}], p = {dif['p_two_sided']:.2f}\n"
        f"no rise in concentration",
        xy=(0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
        fontsize=7.2, color=S.INK, linespacing=1.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Cumulative share of firms")
    ax.set_ylabel("Cumulative share of recalls")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    S.tidy(ax, grid="both")
    S.panel_title(ax, "A  Recall concentration", "Lorenz curves by era")


def panel_b(ax, d):
    F, prof = d["firms"], d["profiles"]
    cmap = cluster_colors(prof)
    rng = np.random.default_rng(4)
    F = F[F.median_units.notna()].copy()
    jit = rng.normal(0, 0.028, len(F))
    ax.scatter(F.median_units.clip(lower=10), F.share_online_only + jit,
               s=6 + 9 * F.n_recalls, c=[cmap[int(c)] for c in F.cluster],
               alpha=0.45, linewidth=0, zorder=2)

    LABEL_AT = {"Marketplace importer, online only": (0, 13, "center", "bottom"),
                "National retailer / mass brand": (0, 24, "center", "bottom"),
                "Premium / specialist brand": (0, -13, "center", "top")}
    for _, r in prof.iterrows():
        c = int(r.cluster)
        x = max(float(r.median_units), 10)
        ax.scatter([x], [r.share_online_only], s=70, facecolor=cmap[c],
                   edgecolor=S.SURFACE, linewidth=1.4, zorder=4)
        if r.label in LABEL_AT:
            dx, dy, ha, va = LABEL_AT[r.label]
            ax.annotate(f"{r.label}\n{int(r.n_firms)} firms, "
                        f"{int(r.n_recalls_total)} recalls",
                        xy=(x, r.share_online_only), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va=va,
                        fontsize=7.2, color=S.INK, linespacing=1.3,
                        arrowprops=dict(arrowstyle="-", color=cmap[c],
                                        linewidth=1.0, shrinkA=1, shrinkB=4))
    ax.set_xscale("log")
    ax.set_xlim(F.median_units.clip(lower=10).min() * 0.5,
                F.median_units.max() * 2.5)
    ax.set_ylim(-0.72, 1.62)
    ax.set_yticks([0, 0.5, 1.0])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Median units per recall")
    ax.set_ylabel("Online-only share of recalls")
    S.tidy(ax)
    ax.spines["left"].set_bounds(0, 1)
    S.panel_title(ax, "B  Firm typology", "GMM, k = 7; area = recalls")


def panel_c(ax, d):
    F, prof = d["firms"], d["profiles"]
    cmap = cluster_colors(prof)
    top = F.nlargest(16, "n_recalls").sort_values("n_recalls")
    y = np.arange(len(top))
    xmax = float(top.n_recalls.max())
    left = -xmax * 0.62
    for yi, (_, r) in zip(y, top.iterrows()):
        c = cmap[int(r.cluster)]
        ax.plot([0, r.n_recalls], [yi, yi], color=c, linewidth=1.6, alpha=0.55,
                solid_capstyle="butt", zorder=2)
        ax.plot([r.n_recalls], [yi], marker="o", markersize=6.5, color=c,
                zorder=3)
        ax.annotate(f"{int(r.n_recalls)}", xy=(r.n_recalls, yi), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=7.2, color=S.INK_2)
        ax.annotate(f"{r.firm_name}  {int(r.first_year)}-{int(r.last_year)}",
                    xy=(-0.25, yi), ha="right", va="center", fontsize=7.4,
                    color=S.INK, annotation_clip=False)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, xmax + 1, 5))
    ax.set_xlim(left, xmax * 1.10)
    ax.set_ylim(-0.8, len(top) - 0.2)
    ax.set_xlabel("Recalls, 1974-2026")
    S.tidy(ax, grid="x", spines=("top", "right", "left"))
    ax.spines["bottom"].set_bounds(0, xmax)
    for gl in ax.get_xgridlines():
        gl.set_zorder(0)
    S.panel_title(ax, "C  The repeat offenders",
                  "Top 16 firms after fuzzy-merging near-duplicate names")


def panel_d(ax, d):
    sim, pairs, s = d["sim"], d["pairs"], d["summary"]
    hop = s["entity_hopping"]
    mid = (sim.bin_left + sim.bin_right) / 2
    ax.bar(mid, sim.n_pairs.clip(lower=0.6), width=0.018, color=S.GRAY_3,
           zorder=2)
    ax.set_yscale("log")
    ax.set_xlim(-0.02, 1.03)
    ax.set_ylim(0.6, sim.n_pairs.max() * 400)
    ax.axvline(hop["similarity_threshold"], color=S.INK_3, linewidth=0.9,
               linestyle=(0, (4, 3)), zorder=3)
    ax.annotate(f"candidate threshold {hop['similarity_threshold']:.2f}",
                xy=(hop["similarity_threshold"], sim.n_pairs.max() * 250),
                xytext=(-6, 0), textcoords="offset points", ha="right",
                va="center", fontsize=7.2, color=S.INK_3)

    flagged = pairs[pairs.flagged] if "flagged" in pairs else pairs.iloc[:0]
    tier2 = (pairs[pairs.tier == "category_match_online"]
             if "tier" in pairs else pairs.iloc[:0])
    rng = np.random.default_rng(9)
    y_tier2 = sim.n_pairs.max() * 4.0
    y_flag = sim.n_pairs.max() * 60.0
    if len(tier2):
        ax.scatter(tier2.similarity,
                   10 ** (np.log10(y_tier2) + rng.normal(0, 0.035, len(tier2))),
                   s=20, facecolor="none", edgecolor=S.AMBER, linewidth=1.1,
                   zorder=4)
        ax.annotate(f"{len(tier2)} pairs: same category, same origin, "
                    f"both online only",
                    xy=(tier2.similarity.min(), y_tier2), xytext=(-10, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=7.2, color=S.AMBER)
    if len(flagged):
        ax.scatter(flagged.similarity,
                   10 ** (np.log10(y_flag) + rng.normal(0, 0.035, len(flagged))),
                   s=24, color=S.PINK, zorder=5)
        ax.annotate(
            f"{len(flagged)} pairs share a brand token: \u201c"
            f"{flagged.phrase_a.iloc[0]}\u201d under "
            f"{hop['n_firms_involved']} importer names",
            xy=(flagged.similarity.min(), y_flag), xytext=(-10, 0),
            textcoords="offset points", ha="right", va="center",
            fontsize=7.2, color=S.PINK)
    ax.set_xlabel("TF-IDF cosine similarity between two product descriptions")
    ax.set_ylabel("Pairs (log scale)")
    S.tidy(ax)
    S.panel_title(ax, "D  Entity hopping",
                  f"All {hop['n_pairs_compared']:,} product-description pairs")


def build(fig):
    d = load()
    fig.get_layout_engine().set(hspace=0.06, wspace=0.05)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.05, 1.02, 0.90, 0.72])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    ax_d = fig.add_subplot(gs[2, :])
    ax_leg = fig.add_subplot(gs[3, :])
    ax_leg.axis("off")
    panel_a(ax_a, d)
    panel_b(ax_b, d)
    panel_c(ax_c, d)
    panel_d(ax_d, d)

    prof = d["profiles"]
    cmap = cluster_colors(prof)
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=6,
                      color=cmap[int(r.cluster)], label=r.label)
               for _, r in prof.sort_values("n_recalls_total",
                                            ascending=False).iterrows()]
    leg = ax_leg.legend(handles=handles, loc="upper center", ncol=3,
                        frameon=False, fontsize=7.4, handlelength=1.0,
                        columnspacing=1.5, labelspacing=0.5,
                        bbox_to_anchor=(0.5, 1.02))
    for t in leg.get_texts():
        t.set_color(S.INK)

    s = d["summary"]
    ex = s["extraction_qa"]
    mq = s["merge_qa"]
    note = (
        f"CPSC recalls of apparel and home textiles, 1974-2026. Firm names are "
        f"re-extracted from the recall title ({ex['parse_rate_rebuilt']:.1%} of "
        f"{ex['n_records']} records parsed, against "
        f"{ex['parse_rate_baseline']:.1%} for the shared extractor, which "
        f"returns \"CPSC, <firm> Announce\" on {ex['n_baseline_keys_containing_cpsc']} "
        f"records) and near-duplicate spellings are fuzzy-merged "
        f"({mq['n_merge_pairs']} merges, {mq['n_keys_before_merge']} keys to "
        f"{mq['n_firms_after_merge']} firms). Panel B jitters the online-only "
        f"share vertically so overlapping firms are visible. 2026 is partial, "
        f"to 16 July.")
    S.source_note(fig, "\n".join(textwrap.wrap(note, 128)), y=0.002)


def main(quick: bool = False):
    out = S.save_figure(build, str(HERE / "figure_firm_concentration"),
                        height_in=8.6)
    print("[13] wrote " + ", ".join(Path(p).name for p in out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
