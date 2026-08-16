from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import style as S

RES = HERE / "results"

N_LABEL = 8
A_COLOR = S.PINK
B_COLOR = S.BLUE

CHANNEL_ORDER = ["online_only", "mixed", "store_only", "unknown"]
CHANNEL_LABEL = {"online_only": "Online\nonly", "mixed": "Online +\nstores",
                 "store_only": "Stores\nonly", "unknown": "Channel\nunknown"}
HAZARD_ORDER = ["flammability_burn", "choking_small_parts",
                "drawstring_strangulation", "chemical", "fall_slip",
                "laceration_puncture"]
HAZARD_LABEL = {"flammability_burn": "Flamm-\nability",
                "choking_small_parts": "Choking",
                "drawstring_strangulation": "Draw-\nstring",
                "chemical": "Chemical", "fall_slip": "Fall /\nslip",
                "laceration_puncture": "Lacer-\nation"}


def load():
    j = json.loads((RES / "summary.json").read_text())
    return {
        "s": j,
        "period": pd.read_csv(RES / "topic_by_period.csv"),
        "group": pd.read_csv(RES / "topic_by_group.csv"),
        "sel": pd.read_csv(RES / "topic_selection.csv"),
        "trend": pd.read_csv(RES / "topic_trend_posterior.csv"),
        "params": pd.read_csv(RES / "topic_trend_params.csv"),
        "lo_ch": pd.read_csv(RES / "logodds_online_vs_store.csv"),
        "lo_era": pd.read_csv(RES / "logodds_recent_vs_early.csv"),
    }


def topic_order_colors(d):
    tot = (d["period"].groupby(["topic", "topic_name"], as_index=False)
           .n_dominant.sum().sort_values("n_dominant", ascending=False))
    order = [int(t) for t in tot.topic]
    names = dict(zip(tot.topic.astype(int), tot.topic_name))
    extras = [S.GRAY_1, S.GRAY_2, S.GRAY_3, S.GRAY_4]
    colors = {}
    for i, t in enumerate(order):
        colors[t] = S.CAT[i] if i < len(S.CAT) else extras[(i - len(S.CAT))
                                                           % len(extras)]
    return order, names, colors


def nice(name: str) -> str:
    return name[:1].upper() + name[1:]


def panel_a(ax, d, order, names, colors):
    p = d["period"]
    periods = [q for q in ["1974-89", "1990-99", "2000-09", "2010-19",
                           "2020-26"] if q in set(p.period)]
    x = np.arange(len(periods))
    tab = (p.pivot_table(index="period", columns="topic",
                         values="share_dominant", aggfunc="first")
           .reindex(periods))
    n_by_period = (p.groupby("period").n_period.first().reindex(periods))
    series = {t: tab[t].fillna(0).to_numpy() for t in order}
    S.stacked_bars(ax, x, series, colors, width=0.74, gap=0.006,
                   min_label=0.10, label_fmt="{:.0%}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q[2:4]}-{q[-2:]}\n{int(n)}"
                        for q, n in zip(periods, n_by_period)], fontsize=6.8)
    ax.annotate("period / n", xy=(0, 0), xycoords="axes fraction",
                xytext=(-4, -22), textcoords="offset points", ha="right",
                va="center", fontsize=6.4, color=S.INK_3)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("Share of recalls")
    S.tidy(ax)
    S.panel_title(ax, "A  What the recalls are about",
                  "Dominant topic, by period")


def panel_b(ax, d, colors, names):
    tr, s = d["trend"], d["s"]
    slopes = {r["topic_name"]: r for r in s["trend_model"]["slopes"]}
    modelled = s["trend_model"]["topics"]
    name_to_topic = {v: k for k, v in names.items()}
    ends = {}
    for i, nm in enumerate(modelled):
        sub = tr[tr.topic == nm].sort_values("year")
        c = colors.get(name_to_topic.get(nm), S.GRAY_2)
        ax.fill_between(sub.year, sub.lo94, sub.hi94, color=c, alpha=0.16,
                        linewidth=0, zorder=1)
        ax.plot(sub.year, sub["mean"], color=c, linewidth=2.0, zorder=3)
        ends[nm] = float(sub["mean"].iloc[-1])
    top = min(1.0, max(0.62, tr.hi94.max() * 1.05))
    gap = 0.155 * top
    spread = S.spread_labels(ends, gap)
    hi = max(spread.values())
    if hi > top - 0.06 * top:
        shift = hi - (top - 0.06 * top)
        spread = {k2: v - shift for k2, v in spread.items()}
    for nm, ypos in spread.items():
        c = colors.get(name_to_topic.get(nm), S.GRAY_2)
        sl = slopes.get(nm)
        lab = nice(nm) if sl is None else (
            f"{nice(nm)}\n{sl['beta_per_decade']:+.2f} "
            f"[{sl['hdi_3%']:+.2f}, {sl['hdi_97%']:+.2f}]")
        ax.annotate(lab, xy=(tr.year.max(), ends[nm]),
                    xytext=(tr.year.max() + 1.4, ypos), ha="left",
                    va="center", fontsize=6.2, color=S.INK, linespacing=1.3,
                    arrowprops=dict(arrowstyle="-", color=c, linewidth=1.0,
                                    shrinkA=0, shrinkB=1))
    ax.set_xlim(tr.year.min() - 0.5, tr.year.max() + 13.0)
    ax.set_ylim(0, top)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("Posterior share of recalls")
    S.tidy(ax)
    diag = s["trend_model"]["diagnostics"]
    ax.annotate(f"Dirichlet-multinomial, logistic random walk\n"
                f"{diag['n_chains']} chains x {diag['n_draws']:,} draws, "
                f"max R-hat {diag['max_rhat_trend']:.2f}",
                xy=(0.0, 1.0), xycoords="axes fraction", xytext=(2, -2),
                textcoords="offset points", ha="left", va="top", fontsize=6.6,
                color=S.INK_3, linespacing=1.4)
    S.panel_title(ax, "B  Topic trajectories, modelled",
                  "Posterior mean and 94% HDI")


def panel_logodds(ax, t, label_a, label_b, title, sub, letter,
                  n_label=N_LABEL,
                  side_note=("", "")):
    freq = t.count_total.to_numpy(float)
    z = t.z.to_numpy(float)
    sig = np.abs(z) >= 1.96
    ax.axhline(0, color=S.AXIS, linewidth=0.9, zorder=1)
    for lev in (1.96, -1.96):
        ax.axhline(lev, color=S.RULE, linewidth=0.9, linestyle=(0, (4, 3)),
                   zorder=1)
    ax.scatter(freq[~sig], z[~sig], s=5, color=S.GRAY_3, alpha=0.55,
               linewidth=0, zorder=2)
    ax.scatter(freq[sig & (z > 0)], z[sig & (z > 0)], s=13, color=A_COLOR,
               alpha=0.85, linewidth=0, zorder=3)
    ax.scatter(freq[sig & (z < 0)], z[sig & (z < 0)], s=13, color=B_COLOR,
               alpha=0.85, linewidth=0, zorder=3)

    ax.set_xscale("log")
    lo_z, hi_z = z.min(), z.max()
    pad = 0.16 * (hi_z - lo_z)
    ax.set_ylim(lo_z - pad, hi_z + pad)

    y0, y1 = lo_z - pad, hi_z + pad
    span = y1 - y0
    gap = span * 0.045
    lo_x, hi_x = freq.min() * 0.55, freq.max() * 14.0
    x_lab = 10 ** (np.log10(hi_x) - 0.012 * (np.log10(hi_x) - np.log10(lo_x)))
    ax.set_xlim(lo_x, hi_x)
    for head, color, top_side in [(t.head(n_label), A_COLOR, True),
                                  (t.tail(n_label).iloc[::-1], B_COLOR,
                                   False)]:
        n = len(head)
        anchor = (y1 - 0.10 * span) if top_side else (y0 + 0.10 * span)
        ladder = [anchor - i * gap if top_side else anchor + i * gap
                  for i in range(n)]
        for (r, yl) in zip(head.itertuples(), ladder):
            ax.annotate(r.term, xy=(r.count_total, r.z), xytext=(x_lab, yl),
                        ha="right", va="center", fontsize=6.3, color=color,
                        arrowprops=dict(arrowstyle="-", color=color,
                                        linewidth=0.6, alpha=0.38,
                                        shrinkA=1, shrinkB=2))
    ax.annotate(side_note[0], xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(2, -2), textcoords="offset points", ha="left",
                va="top", fontsize=6.8, color=A_COLOR, fontweight="semibold")
    ax.annotate(side_note[1], xy=(0.0, 0.0), xycoords="axes fraction",
                xytext=(2, 3), textcoords="offset points", ha="left",
                va="bottom", fontsize=6.8, color=B_COLOR,
                fontweight="semibold")
    ax.set_xlabel("Term frequency in the corpus (log scale)")
    ax.set_ylabel("Log-odds z-score")
    S.tidy(ax, grid="y")
    S.panel_title(ax, f"{letter}  {title}", sub)


def panel_heat(ax, d, order, names, colors, letter):
    g = d["group"]
    ch = g[g.variable == "sales_channel"]
    hz = g[g.variable == "hazard_category"]
    cols = ([("sales_channel", c) for c in CHANNEL_ORDER
             if c in set(ch.group)]
            + [("hazard_category", h) for h in HAZARD_ORDER
               if h in set(hz.group)])
    M = np.zeros((len(order), len(cols)))
    for i, t in enumerate(order):
        for j, (var, grp) in enumerate(cols):
            r = g[(g.variable == var) & (g.topic == t) & (g.group == grp)]
            M[i, j] = float(r.share_of_group.iloc[0]) if len(r) else np.nan
    ramp = S.SEQ_BLUE
    vmax = np.nanmax(M)
    for i in range(len(order)):
        for j in range(len(cols)):
            v = M[i, j]
            k = int(np.clip(v / max(vmax, 1e-9) * (len(ramp) - 1), 0,
                            len(ramp) - 1))
            ax.add_patch(plt_rect(j, i, ramp[k]))
            if v >= 0.04:
                ax.text(j + 0.5, i + 0.5, f"{v:.0%}", ha="center",
                        va="center", fontsize=6.2,
                        color="white" if k >= 4 else S.INK,
                        fontweight="semibold" if k >= 5 else "normal")
    n_by_group = {}
    for var, grp in cols:
        n_by_group[(var, grp)] = int(g[(g.variable == var) &
                                       (g.group == grp)].n.sum())
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(
        [f"{CHANNEL_LABEL.get(c, HAZARD_LABEL.get(c, c))}\n"
         f"n={n_by_group[(v, c)]}" for v, c in cols], fontsize=6.3)
    ax.set_yticks(np.arange(len(order)) + 0.5)
    ax.set_yticklabels([textwrap.shorten(nice(names[t]), 22, placeholder="…")
                        for t in order], fontsize=7.0)
    for i, t in enumerate(order):
        ax.add_patch(plt_rect(-0.30, i, colors[t], w=0.22))
    ax.set_xlim(-0.34, len(cols))
    ax.set_ylim(len(order), 0)
    n_ch = sum(1 for v, _ in cols if v == "sales_channel")
    ax.axvline(n_ch, color=S.SURFACE, linewidth=3.2, zorder=4)
    ax.annotate("Sales channel", xy=(n_ch / 2, 0), xytext=(0, 4),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=7.2, color=S.INK_2)
    ax.annotate("Hazard category", xy=((len(cols) + n_ch) / 2, 0),
                xytext=(0, 4), textcoords="offset points", ha="center",
                va="bottom", fontsize=7.2, color=S.INK_2)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0, colors=S.INK_3)
    ax.grid(False)
    tests = d["s"]["association_tests"]
    S.panel_title(
        ax, f"{letter}  Which topics sit in which channel and hazard",
        f"Column shares; V = {tests['sales_channel']['cramers_v']:.2f} "
        f"(channel), {tests['hazard_category']['cramers_v']:.2f} (hazard)",
        pad=22)


def plt_rect(x, y, color, w=0.94, h=0.94):
    from matplotlib.patches import Rectangle
    return Rectangle((x + (1 - w) / 2, y + (1 - h) / 2), w, h, facecolor=color,
                     edgecolor="none", zorder=2)


def build(fig):
    d = load()
    s = d["s"]
    order, names, colors = topic_order_colors(d)
    fig.get_layout_engine().set(hspace=0.075, wspace=0.06)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.02, 1.40, 1.16, 0.82])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])
    ax_leg = fig.add_subplot(gs[3, :])
    ax_leg.axis("off")

    panel_a(ax_a, d, order, names, colors)
    panel_b(ax_b, d, colors, names)
    lset = s["logodds_settings"]
    panel_logodds(ax_c, d["lo_ch"], "online_only", "store_only",
                  "Online-only vs store-only words",
                  f"{lset['n_online_only']} vs {lset['n_store_only']} recalls",
                  "C", side_note=("More likely online-only",
                                  "More likely stores only"))
    panel_logodds(ax_d, d["lo_era"], "2015-2026", "pre-2015",
                  "2015-2026 vs pre-2015",
                  f"{lset['n_2015plus']} vs {lset['n_pre2015']} recalls",
                  "D", side_note=("More likely 2015-2026",
                                  "More likely pre-2015"))
    panel_heat(ax_e, d, order, names, colors, "E")

    handles = [Patch(facecolor=colors[t],
                     label=textwrap.shorten(nice(names[t]), 26,
                                            placeholder="…"))
               for t in order]
    leg = ax_leg.legend(handles=handles, loc="upper center",
                        ncol=(len(handles) if len(handles) <= 5 else 4),
                        frameon=False, fontsize=7.2,
                        handlelength=1.0, handleheight=1.0, columnspacing=1.5,
                        bbox_to_anchor=(0.5, 1.0), title=None)
    for t in leg.get_texts():
        t.set_color(S.INK)

    c = s["corpus"]
    sel = s["selection"]
    note = (
        f"CPSC apparel and home-textile recalls, {c['n_documents']} of "
        f"{c['n_records']} with usable text. Topics are fitted to the PRODUCT "
        f"PHRASE of each notice ({c['total_tokens_product']:,} tokens, "
        f"vocabulary {c['vocabulary_product']}), which is the seller's own "
        f"description; the log-odds panels use the whole notice text "
        f"({c['total_tokens']:,} tokens, vocabulary {c['vocabulary_full']}). "
        f"Raw CPSC JSON present: {str(c['raw_json_present']).lower()} - when it "
        f"is, product names, hazard text and descriptions are appended and both "
        f"corpora grow. k = {sel['k']} chosen by {sel['rule']} over "
        f"k = {min(sel['k_range'])}-{max(sel['k_range'])} "
        f"({sel['primary_model'].upper()}, NPMI {sel['nmf_best_coherence']:.3f}). "
        f"Panels C and D: log-odds with an informative Dirichlet prior "
        f"(a0 = {lset['prior_a0']:.0f}), dashed lines at z = ±1.96; the "
        f"{N_LABEL} most distinctive terms per side are labelled. "
        f"Run mode: {s['mode']}.")
    S.source_note(fig, "\n".join(textwrap.wrap(note, 133)), y=0.001)


def main(quick: bool = False):
    out = S.save_figure(build, str(HERE / "figure_text_topics"), height_in=8.8)
    print("[17] wrote " + ", ".join(Path(p).name for p in out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
