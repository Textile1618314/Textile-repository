from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import style as S

RES = HERE / "results"

INELIGIBLE_ORDER = ["robe", "loungewear", "nightgown", "wearable_blanket"]
ELIGIBLE_ORDER = ["pajama_set", "sleepwear_generic", "underwear_base"]
ARCHE_COLOR = {
    "robe": S.PINK,
    "loungewear": S.AMBER,
    "nightgown": S.PURPLE,
    "wearable_blanket": S.RED,
    "pajama_set": S.BLUE,
    "sleepwear_generic": S.TEAL,
    "underwear_base": S.SEQ_BLUE[2],
}
ARCHE_LABEL = {
    "robe": "Robes", "loungewear": "Loungewear", "nightgown": "Nightgowns",
    "wearable_blanket": "Wearable blankets",
    "pajama_set": "Pyjama sets", "sleepwear_generic": "Sleepwear (generic)",
    "underwear_base": "Underwear / thermals", "outerwear": "Outerwear",
    "daywear": "Daywear", "generic_apparel": "Unnamed garments",
    "footwear": "Footwear", "accessory": "Accessories",
    "home_textile": "Home textiles", "costume": "Costumes",
    "swimwear": "Swimwear", "toy_juvenile": "Toys / juvenile products",
    "non_textile_other": "Non-textile", "unclassified": "Unclassified",
}
BOUNDARY_COLOR = {
    "exemption_ineligible": S.PINK,
    "exemption_eligible": S.BLUE,
    "not_sleepwear": S.GRAY_2,
}
BOUNDARY_LABEL = {
    "exemption_ineligible": "Cannot qualify for the tight-fitting exemption",
    "exemption_eligible": "Could qualify (pyjamas, thermals)",
    "not_sleepwear": "Not sleepwear",
}


def load():
    j = json.loads((RES / "boundary_summary.json").read_text())
    return {
        "summary": j,
        "arche_bin": pd.read_csv(RES / "sleepwear_archetype_by_bin.csv"),
        "by_year": pd.read_csv(RES / "arbitrage_by_year.csv"),
        "by_bin": pd.read_csv(RES / "arbitrage_by_bin.csv"),
        "fit": pd.read_csv(RES / "arbitrage_trend_fit.csv"),
        "pu": pd.read_csv(RES / "price_units_by_archetype.csv"),
    }


def panel_a(ax, d):
    tab = d["arche_bin"].set_index("bin5")
    bins = list(tab.index)
    x = np.arange(len(bins))
    order = [c for c in INELIGIBLE_ORDER if c in tab.columns] + \
            [c for c in ELIGIBLE_ORDER if c in tab.columns]
    other = [c for c in tab.columns if c not in order]
    bottom = np.zeros(len(bins))
    n_inel = np.zeros(len(bins))

    series = order + (["_other"] if other else [])
    last_mid = {}
    for col in series:
        vals = (tab[other].sum(axis=1).to_numpy(float) if col == "_other"
                else tab[col].to_numpy(float))
        color = S.GRAY_3 if col == "_other" else ARCHE_COLOR[col]
        ax.bar(x, vals, bottom=bottom, width=0.62, color=color,
               edgecolor=S.SURFACE, linewidth=1.1, zorder=2)
        for xi, v, b in zip(x, vals, bottom):
            if v >= 4:
                ax.text(xi, b + v / 2, f"{int(v)}", ha="center", va="center",
                        fontsize=7.5,
                        color=S.INK if col == "_other" else "white",
                        fontweight="semibold", zorder=3)
        if vals[-1] > 0:
            last_mid[col] = bottom[-1] + vals[-1] / 2
        if col in INELIGIBLE_ORDER:
            n_inel += vals
        bottom = bottom + vals

    for xi, tot, inel in zip(x, bottom, n_inel):
        if tot:
            ax.annotate(f"{inel / tot:.0%}", xy=(xi, tot), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=S.PINK, fontweight="semibold")
    spread = S.spread_labels(last_mid, gap=bottom.max() * 0.088)
    for col, ypos in spread.items():
        color = S.GRAY_3 if col == "_other" else ARCHE_COLOR[col]
        label = ("Other, not named sleepwear" if col == "_other"
                 else ARCHE_LABEL[col])
        ax.annotate(label, xy=(x[-1] + 0.33, last_mid[col]),
                    xytext=(x[-1] + 0.62, ypos), ha="left", va="center",
                    fontsize=7.6, color=S.INK,
                    arrowprops=dict(arrowstyle="-", color=color, linewidth=1.2,
                                    shrinkA=0, shrinkB=1))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\nn = {int(t)}" for b, t in zip(bins, bottom)])
    ax.set_xlim(-0.62, len(bins) + 1.35)
    ax.set_ylabel("Recalls citing the sleepwear standard")
    ax.set_ylim(0, bottom.max() * 1.12)
    S.tidy(ax)
    S.panel_title(
        ax, "A  Garments recalled under the sleepwear standard",
        "Pink % = share that could never take the tight-fitting exemption")


def panel_b(ax, d):
    fit, by_year, by_bin = d["fit"], d["by_year"], d["by_bin"]
    s = d["summary"]
    tr = s["arbitrage_trend"]["broad"]["logistic"]
    ca = s["arbitrage_trend"]["broad"]["cochran_armitage"]
    pois = s["poisson_count_trend"]["arbitrage_count"]

    ax.fill_between(fit.year, fit.lo, fit.hi, color=S.SEQ_BLUE[1],
                    alpha=0.75, linewidth=0, zorder=1)
    ax.plot(fit.year, fit.fit, color=S.BLUE, linewidth=2.2, zorder=3)

    ax.scatter(by_year.year, by_year.share, s=8 + 5.5 * by_year.n_sleepwear_std,
               facecolor="none", edgecolor=S.GRAY_2, linewidth=1.0, zorder=4)
    mid = [float(np.clip(np.mean([int(p) for p in
                                  (b.split("-")[0],
                                   b.split("-")[0][:2] + b.split("-")[1])]),
                         fit.year.min(), fit.year.max()))
           for b in by_bin.bin5]
    ax.errorbar(mid, by_bin.share,
                yerr=[by_bin.share - by_bin.lo95, by_bin.hi95 - by_bin.share],
                fmt="o", color=S.PINK, ecolor=S.PINK, elinewidth=1.4,
                capsize=0, markersize=6.5, zorder=5)
    for m, sh, hi, b, k, n in zip(mid, by_bin.share, by_bin.hi95, by_bin.bin5,
                                  by_bin.n_arbitrage, by_bin.n_sleepwear_std):
        ax.annotate(f"{b}\n{int(k)} of {int(n)}", xy=(m, hi), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.2, color=S.INK, linespacing=1.3)

    ax.set_ylim(-0.04, 1.42)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(fit.year.min() - 1.4, fit.year.max() + 2.6)
    ax.set_ylabel("Share of sleepwear-standard recalls")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    S.tidy(ax)
    ax.spines["left"].set_bounds(0, 1.0)
    S.panel_title(
        ax, "B  Category arbitrage over time",
        "Share of sleepwear-standard recalls whose product is not named "
        "sleepwear")

    ax.annotate("logistic trend\n95% band",
                xy=(fit.year.iloc[-1], fit.fit.iloc[-1]), xytext=(6, -2),
                textcoords="offset points", ha="left", va="center",
                fontsize=7.4, color=S.BLUE, linespacing=1.3)
    key_x = fit.year.min() + 6.5
    ax.scatter([key_x], [1.06], s=46, facecolor="none", edgecolor=S.GRAY_2,
               linewidth=1.0, zorder=4, clip_on=False)
    ax.annotate("one year (circle area = recalls that year)",
                xy=(key_x, 1.06), xytext=(9, 0), textcoords="offset points",
                ha="left", va="center", fontsize=7.2, color=S.INK_3)
    ax.annotate("five-year bins, 95% Wilson CI",
                xy=(mid[-2], by_bin.hi95.iloc[-2]), xytext=(0, 34),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=7.2, color=S.PINK,
                arrowprops=dict(arrowstyle="-", color=S.PINK, linewidth=0.9,
                                shrinkA=2, shrinkB=3))

    ax.annotate(
        f"Share: logistic OR {tr['or_per_decade']:.2f} per decade "
        f"[{tr['or_ci95'][0]:.2f}, {tr['or_ci95'][1]:.2f}], p = {tr['p'][1]:.2f}; "
        f"Cochran-Armitage z = {ca['z']:.2f}, p = {ca['p_two_sided']:.2f} - flat.\n"
        f"Count: Poisson trend {pois['rate_ratio_per_decade']:.1f}x per decade "
        f"[{pois['rr_ci95'][0]:.1f}, {pois['rr_ci95'][1]:.1f}], "
        f"p = {pois['p_quasi']:.3f} - the caseload is what grows.",
        xy=(0.005, 1.0), xycoords="axes fraction", ha="left", va="top",
        fontsize=7.5, color=S.INK, linespacing=1.5)


def panel_c(axes, d):
    ax_u, ax_p = axes
    pu = d["pu"]
    keep = pu[(pu.n_recalls >= 3) &
              (~pu.archetype.isin(["non_textile_other", "unclassified",
                                   "toy_juvenile"]))].copy()
    keep = keep.sort_values("units_median")
    y = np.arange(len(keep))

    for ax, med, q1, q3, nlab, title, sub, xlab in [
        (ax_u, "units_median", "units_q1", "units_q3", "units_n",
         "C  Units recalled per action", "Median, IQR bar; count of recalls in brackets",
         "Units (log scale)"),
        (ax_p, "price_median", "price_q1", "price_q3", "price_n",
         "D  Unit price", "Median and IQR; row order as in C",
         "US$ (log scale)"),
    ]:
        for yi, (_, r) in zip(y, keep.iterrows()):
            c = BOUNDARY_COLOR[r.boundary_class]
            ax.plot([max(r[q1], 1e-9), max(r[q3], 1e-9)], [yi, yi], color=c,
                    linewidth=2.0, alpha=0.35, solid_capstyle="butt", zorder=2)
            ax.plot([r[med]], [yi], marker="o", markersize=6.5, color=c,
                    zorder=3)
        ax.set_yticks(y)
        ax.set_xscale("log")
        S.tidy(ax, grid="x")
        ax.set_xlabel(xlab)
        S.panel_title(ax, title, sub)

    ax_u.set_yticklabels([f"{ARCHE_LABEL.get(a, a)}  ({int(n)})"
                          for a, n in zip(keep.archetype, keep.n_recalls)])
    ax_p.set_yticklabels([])
    ax_u.tick_params(axis="y", labelsize=7.8)
    for ax in (ax_u, ax_p):
        ax.set_ylim(-0.8, len(keep) - 0.2)

    for yi, (_, r) in zip(y, keep.iterrows()):
        ax_u.annotate(f"{r.units_median:,.0f}", xy=(r.units_q3, yi),
                      xytext=(4, 0), textcoords="offset points", ha="left",
                      va="center", fontsize=6.8, color=S.INK_2)
        ax_p.annotate(f"${r.price_median:,.0f}", xy=(r.price_q3, yi),
                      xytext=(4, 0), textcoords="offset points", ha="left",
                      va="center", fontsize=6.8, color=S.INK_2)
    ax_u.set_xlim(ax_u.get_xlim()[0] * 0.75, ax_u.get_xlim()[1] * 2.4)
    ax_p.set_xlim(ax_p.get_xlim()[0] * 0.85, ax_p.get_xlim()[1] * 2.0)


def build(fig):
    d = load()
    fig.get_layout_engine().set(hspace=0.055, wspace=0.03)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.00, 0.92, 1.02, 0.42])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :])
    ax_c1 = fig.add_subplot(gs[2, 0])
    ax_c2 = fig.add_subplot(gs[2, 1])
    ax_leg = fig.add_subplot(gs[3, :])
    ax_leg.axis("off")
    panel_a(ax_a, d)
    panel_b(ax_b, d)
    panel_c((ax_c1, ax_c2), d)

    handles = [Patch(facecolor=BOUNDARY_COLOR[k], label=BOUNDARY_LABEL[k])
               for k in ["exemption_ineligible", "exemption_eligible",
                         "not_sleepwear"]]
    leg = ax_leg.legend(handles=handles, loc="upper center", ncol=3,
                        frameon=False, fontsize=7.8, handlelength=1.0,
                        handleheight=1.0, columnspacing=1.8,
                        bbox_to_anchor=(0.5, 1.05))
    for t in leg.get_texts():
        t.set_color(S.INK)

    s = d["summary"]
    n_sw = s["sleepwear_standard_flags"]["sleep_std_broad"]
    note = (
        f"CPSC recalls of apparel and home textiles, 1974-2026 (n = "
        f"{s['n_records']}); {n_sw} of them cite the children's sleepwear "
        f"standard, 16 CFR 1615/1616. Archetypes are parsed from the product "
        f"phrase of each recall title rather than the whole title "
        f"({s['parse_quality']['unclassified_refined']} unclassified, against "
        f"{s['parse_quality']['unclassified_v2_baseline']} for the first-match "
        f"regex it replaces). Panels A and B start in 2005 because CPSC titles "
        f"only name the standard from the 2000s on. 2026 is partial, to 16 July.")
    S.source_note(fig, "\n".join(textwrap.wrap(note, 128)), y=0.002)


def main(quick: bool = False):
    out = S.save_figure(build, str(HERE / "figure_boundary_archetypes"),
                        height_in=8.6)
    print("[12] wrote " + ", ".join(Path(p).name for p in out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
