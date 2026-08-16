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

CV_COLOR = S.BLUE
TEMPORAL_COLOR = S.PINK
REFUND_COLOR = S.PURPLE
UNITS_COLOR = S.GREEN

MODEL_LABEL = {
    "elastic_net": "Elastic net",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Boosted trees",
    "dummy_stratified": "Dummy strat.",
    "dummy_prior": "Dummy prior",
}
MODEL_ORDER = ["hist_gradient_boosting", "random_forest", "elastic_net",
               "dummy_stratified", "dummy_prior"]

SHORT = {
    "flammability_burn": "Flam.", "choking_small_parts": "Choke",
    "drawstring_strangulation": "Drawstr.", "chemical": "Chem.",
    "fall_slip": "Fall", "laceration_puncture": "Lacer.",
    "entrapment_entanglement": "Entrap.", "protective_failure": "Protect.",
    "other": "Other",
    "online_only": "Online", "mixed": "Mixed", "store_only": "Store",
    "unknown": "Unkn.",
    "United States": "USA", "Hong Kong": "HK", "missing": "Missing",
    "Bangladesh": "Bangl.", "Indonesia": "Indon.", "Vietnam": "Viet.",
    "Pakistan": "Pak.",
    "unclassified": "Unclass.", "outerwear": "Outer", "footwear": "Foot",
    "pajama_set": "Pyjama", "daywear": "Daywear", "accessory": "Access.",
    "sleepwear_generic": "Sleepw.", "home_textile": "Home", "robe": "Robe",
    "loungewear": "Lounge", "nightgown": "Nightg.", "costume": "Costume",
    "underwear_base": "Underw.", "swimwear": "Swim",
    "not_sleepwear": "Not sw.", "exemption_eligible": "Eligible",
    "exemption_ineligible": "Ineligible",
    "firm_led": "Firm-led", "passive_announced": "Passive",
    "other_action": "Other", "unilateral_warning": "Warning",
    "apparel": "Apparel", "home_textile_seg": "Home text.",
}


def load():
    j = json.loads((RES / "summary.json").read_text())
    return {
        "s": j,
        "comp": pd.read_csv(RES / "model_comparison.csv"),
        "temp": pd.read_csv(RES / "temporal_split.csv"),
        "imp_r": pd.read_csv(RES / "permutation_importance_refund.csv"),
        "imp_u": pd.read_csv(RES / "permutation_importance_units.csv"),
        "pdp": pd.read_csv(RES / "pdp_refund.csv"),
        "ice": pd.read_csv(RES / "ice_refund.csv"),
        "cal": pd.read_csv(RES / "calibration_refund.csv"),
        "resid": pd.read_csv(RES / "residuals_units.csv"),
    }


SHORT_FEATURE = {
    "log10_price": "Price (log10 $)",
    "log10_units": "Units (log10)",
    "is_childrens": "Children's",
    "is_violation": "Violation cited",
    "injuries_reported": "Injury reported",
    "archetype": "Archetype",
    "hazard_category": "Hazard",
    "sales_channel": "Channel",
    "primary_country": "Origin",
    "year": "Recall year",
    "firm_prior_recalls": "Prior recalls",
    "title_words": "Title words",
    "n_countries": "No. countries",
    "boundary_class": "Boundary class",
    "enforcement_mode": "Enforcement",
    "sleepwear_standard": "Sleepwear std.",
    "category_arbitrage": "Arbitrage",
    "segment": "Segment",
    "is_electric_textile": "Heated textile",
}


def short(v) -> str:
    return SHORT.get(str(v), str(v))


def panel_ab(ax, d, task, metric, title, sub, xlabel, chance, letter):
    comp = d["comp"][d["comp"].task == task].set_index("model")
    temp = d["temp"][d["temp"].task == task].set_index("model")
    models = [m for m in MODEL_ORDER if m in comp.index]
    y = np.arange(len(models))[::-1]

    ax.axvline(chance, color=S.GRAY_3, linewidth=1.0, linestyle=(0, (4, 3)),
               zorder=1)
    for yi, m in zip(y, models):
        mu = comp.at[m, f"{metric}_mean"]
        sd = comp.at[m, f"{metric}_sd"]
        ax.plot([mu - sd, mu + sd], [yi + 0.21, yi + 0.21], color=CV_COLOR,
                linewidth=1.6, alpha=0.45, solid_capstyle="butt", zorder=2)
        ax.plot([mu], [yi + 0.21], marker="o", markersize=6, color=CV_COLOR,
                zorder=3)
        tv = temp.at[m, metric]
        ax.plot([tv], [yi - 0.21], marker="D", markersize=5.2,
                color=TEMPORAL_COLOR, zorder=3)
        ax.annotate(f"{mu:.2f}", (mu + sd, yi + 0.21), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=6.6, color=CV_COLOR)
        ax.annotate(f"{tv:.2f}", (tv, yi - 0.21), xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=6.6, color=TEMPORAL_COLOR)

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m] for m in models], fontsize=7.6)
    ax.set_ylim(-0.75, len(models) - 0.25)
    lo = min(comp[f"{metric}_mean"].min() - comp[f"{metric}_sd"].max(),
             temp[metric].min(), chance)
    hi = max(comp[f"{metric}_mean"].max() + comp[f"{metric}_sd"].max(),
             temp[metric].max(), chance)
    ax.set_xlim(lo - 0.06 * (hi - lo), hi + 0.30 * (hi - lo))
    ax.set_xlabel(xlabel)
    S.tidy(ax, grid="x")
    S.panel_title(ax, f"{letter}  {title}", sub)


def panel_imp(ax, imp, color, title, sub, xlabel, letter, top=9):
    t = imp.head(top).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(t))
    for yi, r in zip(y, t.itertuples()):
        solid = r.lo95 > 0
        ax.barh(yi, r.importance, height=0.62,
                color=color if solid else S.GRAY_4, zorder=2)
        ax.plot([max(r.lo95, 0 if r.importance > 0 else r.lo95), r.hi95],
                [yi, yi], color=S.INK_2 if solid else S.GRAY_2,
                linewidth=1.1, zorder=3, solid_capstyle="butt")
        ax.annotate(f"{r.importance:.3f}", (max(r.hi95, r.importance), yi),
                    xytext=(4, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=6.8, color=S.INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT_FEATURE.get(f, textwrap.shorten(l, 16,
                                                             placeholder="…"))
                        for f, l in zip(t.feature, t.label)], fontsize=7.2)
    ax.set_ylim(-0.7, len(t) - 0.3)
    ax.set_xlim(min(0, t.lo95.min() * 1.1), max(t.hi95.max(), t.importance.max()) * 1.32)
    ax.axvline(0, color=S.AXIS, linewidth=0.8, zorder=1)
    ax.set_xlabel(xlabel)
    S.tidy(ax, grid="x")
    S.panel_title(ax, f"{letter}  {title}", sub)


def panel_pdp(axes, d, letter, ax_head=None):
    pdp, ice = d["pdp"], d["ice"]
    feats = d["s"]["task_refund"]["pdp_features"]
    lo = min(pdp.pd_mean.min(), 0.55)
    hi = max(pdp.pd_mean.max(), 0.95)
    pad = 0.06 * (hi - lo)
    for k, (ax, f) in enumerate(zip(axes, feats)):
        p = pdp[pdp.feature == f].sort_values("grid_index")
        i = ice[ice.feature == f]
        kind = p.kind.iloc[0]
        x = p.grid_index.to_numpy(float)
        for _, g in i.groupby("line"):
            ax.plot(g.sort_values("grid_index").grid_index,
                    g.sort_values("grid_index").pred, color=S.GRAY_3,
                    linewidth=0.4, alpha=0.28, zorder=1)
        ax.plot(x, p.pd_mean, color=REFUND_COLOR, linewidth=2.1, zorder=3)
        if kind == "categorical":
            ax.plot(x, p.pd_mean, marker="o", markersize=3.4, linestyle="none",
                    color=REFUND_COLOR, zorder=4)
            ax.set_xticks(x)
            ax.set_xticklabels([short(v) for v in p.grid_value], rotation=90,
                               ha="center", fontsize=6.2)
        else:
            v = p.grid_value.to_numpy(float)
            ticks = [0, len(v) // 2, len(v) - 1]
            ax.set_xticks([x[t] for t in ticks])
            ax.set_xticklabels([f"{v[t]:.3g}" for t in ticks], fontsize=6.6)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(x.min() - 0.35, x.max() + 0.35)
        S.tidy(ax, grid="y")
        if k == 0:
            ax.set_ylabel("P(refund)", fontsize=7.4)
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}")
        else:
            ax.set_yticklabels([])
        lab = SHORT_FEATURE.get(f, textwrap.shorten(p.label.iloc[0], 15,
                                                    placeholder="…"))
        ax.annotate(lab, xy=(0, 1), xycoords="axes fraction", xytext=(0, 3),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=7.0, color=S.INK, fontweight="semibold")
    if ax_head is not None:
        ax_head.text(0, 1.02, f"{letter}  Partial dependence of P(refund), "
                     f"top five features",
                     ha="left", va="top", fontsize=10, color=S.INK,
                     fontweight="semibold", transform=ax_head.transAxes)
        ax_head.text(0, 0.34, f"Thick line: partial dependence; thin lines: "
                     f"{int(d['ice'].line.nunique())} recall-level ICE curves",
                     ha="left", va="top", fontsize=8, color=S.INK_3,
                     transform=ax_head.transAxes)


def panel_cal(ax, d, letter):
    cal = d["cal"]
    s = d["s"]["task_refund"]
    ax.plot([0, 1], [0, 1], color=S.GRAY_3, linewidth=1.0,
            linestyle=(0, (4, 3)), zorder=1)
    ax.annotate("perfect calibration", xy=(0.60, 0.60), xytext=(3, -4),
                textcoords="offset points", ha="left", va="top", fontsize=6.9,
                color=S.INK_3, rotation=36, rotation_mode="anchor")
    ax.errorbar(cal.pred_mean, cal.obs_rate,
                yerr=[cal.obs_rate - cal.lo95, cal.hi95 - cal.obs_rate],
                fmt="o", color=REFUND_COLOR, ecolor=REFUND_COLOR,
                elinewidth=1.2, markersize=5.5, capsize=0, zorder=3)
    ax.plot(cal.pred_mean, cal.obs_rate, color=REFUND_COLOR, linewidth=1.4,
            alpha=0.55, zorder=2)
    cm = s["confusion_matrix"]
    ax.annotate(
        f"AUC {s['pooled_auc']:.3f}  Brier {s['pooled_brier']:.3f}\n"
        f"ECE {s['ece']:.3f}  base rate {s['outcome_value']:.3f}",
        xy=(0.02, 0.99), xycoords="axes fraction", ha="left", va="top",
        fontsize=7.0, color=S.INK, linespacing=1.45)
    ax.annotate(
        f"At p = 0.5 the model finds\n"
        f"{cm[1][1]} of {cm[1][0] + cm[1][1]} refunds and\n"
        f"{cm[0][0]} of {cm[0][0] + cm[0][1]} non-refunds",
        xy=(0.98, 0.02), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=7.0, color=S.INK_2, linespacing=1.45)
    lo = min(cal.pred_mean.min(), cal.lo95.min()) - 0.06
    ax.set_xlim(lo, 1.02)
    ax.set_ylim(lo, 1.02)
    ax.set_xlabel("Mean predicted P(refund) in bin")
    ax.set_ylabel("Observed refund share")
    S.tidy(ax, grid="both")
    S.panel_title(ax, f"{letter}  Calibration of the classifier",
                  f"Pooled out-of-fold, {len(cal)} equal-count bins")


def panel_resid(ax, d, letter):
    r = d["resid"]
    s = d["s"]["task_units"]
    lo = min(r.y.min(), r.pred.min()) - 0.2
    hi = max(r.y.max(), r.pred.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], color=S.GRAY_3, linewidth=1.0,
            linestyle=(0, (4, 3)), zorder=1)
    for split, color, size, alpha in [("oof_random_cv", CV_COLOR, 9, 0.40),
                                      ("temporal_2020plus", TEMPORAL_COLOR,
                                       16, 0.75)]:
        sub = r[r.split == split]
        ax.scatter(sub.pred, sub.y, s=size, color=color, alpha=alpha,
                   linewidth=0, zorder=2 if split == "oof_random_cv" else 3)
    tg = [g for g in s["cv_vs_temporal"] if g["model"] == s["best_model"]][0]
    ax.annotate(
        f"Random CV  R² {s['oof_r2']:.2f}\n"
        f"2020+ test  R² {tg['temporal']:.2f}",
        xy=(0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
        fontsize=7.0, color=S.INK, linespacing=1.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted log10 units")
    ax.set_ylabel("Observed log10 units")
    S.tidy(ax, grid="both")
    S.panel_title(ax, f"{letter}  Predicted vs observed",
                  "log10 units; dashed = perfect fit")


def build(fig):
    d = load()
    s = d["s"]
    fig.get_layout_engine().set(hspace=0.075, wspace=0.06)
    gs = fig.add_gridspec(6, 2,
                          height_ratios=[0.95, 1.12, 0.26, 0.86, 1.22, 0.72])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_dd = fig.add_subplot(gs[1, 1])
    ax_head = fig.add_subplot(gs[2, :])
    ax_head.axis("off")
    sub = gs[3, :].subgridspec(1, 5, wspace=0.12)
    ax_e = [fig.add_subplot(sub[0, k]) for k in range(5)]
    ax_f = fig.add_subplot(gs[4, 0])
    ax_g = fig.add_subplot(gs[4, 1])
    ax_leg = fig.add_subplot(gs[5, :])
    ax_leg.axis("off")

    a, b = s["task_refund"], s["task_units"]
    panel_ab(ax_a, d, "refund", "auc",
             "Refund classifier",
             f"n = {a['n']}, 2010-2026, {a['outcome_value']:.0%} refunds",
             "Area under the ROC curve", 0.5, "A")
    panel_ab(ax_b, d, "units", "r2", "log10(units) regression",
             f"n = {b['n']}, {b['year_range'][0]}-{b['year_range'][1]}",
             "R² on held-out recalls", 0.0, "B")

    panel_imp(ax_c, d["imp_r"], REFUND_COLOR, "What predicts a refund",
              f"{s['settings']['perm_folds']} held-out folds, 95% CI",
              "Drop in AUC when permuted", "C")
    panel_imp(ax_dd, d["imp_u"], UNITS_COLOR, "What predicts recall size",
              f"{s['settings']['perm_folds']} held-out folds, 95% CI",
              "Drop in R² when permuted", "D")

    panel_pdp(ax_e, d, "E", ax_head=ax_head)
    panel_cal(ax_f, d, "F")
    panel_resid(ax_g, d, "G")

    handles = [
        Line2D([], [], color=CV_COLOR, marker="o", linestyle="none",
               markersize=6, label="Random CV, mean ± sd"),
        Line2D([], [], color=TEMPORAL_COLOR, marker="D", linestyle="none",
               markersize=5.2, label="Train \u2264 2019, test 2020+"),
        Line2D([], [], color=S.GRAY_4, marker="s", linestyle="none",
               markersize=7, label="CI includes zero"),
    ]
    leg = ax_leg.legend(handles=handles, loc="upper center", ncol=3,
                        frameon=False, fontsize=7.6, handlelength=1.0,
                        handletextpad=0.6, columnspacing=1.8,
                        bbox_to_anchor=(0.5, 1.0))
    for t in leg.get_texts():
        t.set_color(S.INK)

    cfg = s["settings"]
    note = (
        f"CPSC apparel and home-textile recalls, 1974-2026 (n = "
        f"{s['data']['n_records']}). Task A: refund offered, {a['n']} recalls "
        f"2010-2026 with a populated remedy field ({a['n_dropped_missing_remedy_field']} "
        f"2010-15 records with an empty field are dropped, not scored as "
        f"no-refund). Task B: log10 units, {b['n']} recalls with a units count. "
        f"Every number comes from outer {cfg['outer_folds']}-fold x "
        f"{cfg['outer_repeats']}-repeat nested cross-validation with an inner "
        f"{cfg['inner_folds']}-fold randomised search "
        f"({cfg['n_iter']['hist_gradient_boosting']} candidates for the boosted "
        f"model); no outer test fold contributes to imputation, encoding or "
        f"hyperparameter choice. Panels C-G use the best model per task "
        f"({MODEL_LABEL[a['best_model']]} and {MODEL_LABEL[b['best_model']]}). "
        f"Run mode: {s['mode']}.")
    S.source_note(fig, "\n".join(textwrap.wrap(note, 132)), y=0.001)


def main(quick: bool = False):
    out = S.save_figure(build, str(HERE / "figure_ml_remedy_and_scale"),
                        height_in=8.7)
    print("[16] wrote " + ", ".join(Path(p).name for p in out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
