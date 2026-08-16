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
RES = HERE / 'results'
SERIES_COLOR = {'hazard': S.BLUE, 'channel': S.PINK, 'violation': S.PURPLE}
SERIES_LABEL = {'hazard': 'Hazard composition', 'channel': 'Online-only share', 'violation': 'Violation-detected share'}
SHORT = {'flammability_burn': 'Flammability', 'choking_small_parts': 'Choking', 'drawstring_strangulation': 'Drawstring', 'chemical': 'Chemical', 'fall_slip': 'Fall / slip', 'laceration_puncture': 'Laceration', 'entrapment_entanglement': 'Entrapment', 'protective_failure': 'Protective', 'other': 'Other'}
REGIME_COLOR = [S.SEQ_BLUE[6], S.SEQ_BLUE[4], S.SEQ_BLUE[2]]

def load():
    j = json.loads((RES / 'changepoint_summary.json').read_text())
    return {'summary': j, 'haz': pd.read_csv(RES / 'hazard_year_matrix.csv'), 'cp': pd.read_csv(RES / 'changepoint_posterior.csv'), 'comp': pd.read_csv(RES / 'regime_composition.csv'), 'delta': pd.read_csv(RES / 'composition_change.csv'), 'cmp': pd.read_csv(RES / 'model_comparison.csv'), 'sec': pd.read_csv(RES / 'secondary_series.csv')}

def free_text(ax):
    for t in ax.texts:
        t.set_in_layout(False)

def best_cp(d, series):
    K = d['summary']['best_K_by_loo'][series]
    cp = d['cp']
    return (K, cp[(cp.series == series) & (cp.K == K)])

def hazard_K(d):
    K = d['summary']['best_K_by_loo']['hazard']
    return (K, True) if K >= 1 else (1, False)

def panel_a(ax, d):
    h = d['haz']
    cats = [c for c in S.HAZARD_ORDER if c in h.columns]
    years = h.year.to_numpy(int)
    shares = h[cats].to_numpy(float)
    shares = shares / shares.sum(axis=1, keepdims=True)
    bottom = np.zeros(len(years))
    for k, c in enumerate(cats):
        ax.bar(years, shares[:, k], bottom=bottom, width=0.86, color=S.HAZARD_COLORS[c], edgecolor=S.SURFACE, linewidth=0.35, zorder=2)
        bottom = bottom + shares[:, k]
    K, preferred = hazard_K(d)
    cp = d['cp'][(d['cp'].series == 'hazard') & (d['cp'].K == K)]
    strip_lo, strip_hi = (1.035, 1.2)
    ax.axhline(1.0, color=S.AXIS, linewidth=0.6, zorder=3)
    blk = d['summary']['changepoints']['hazard'][f'K{K}']
    for which in range(1, K + 1):
        w = cp[cp.cp == which]
        p = w.prob.to_numpy(float)
        if p.max() <= 0:
            continue
        yy = strip_lo + (strip_hi - strip_lo) * p / p.max()
        ax.fill_between(w.year.to_numpy(int), strip_lo, yy, color=S.BLUE, alpha=0.55, linewidth=0, zorder=4)
        ax.plot(w.year.to_numpy(int), yy, color=S.BLUE, linewidth=1.0, zorder=5)
        mode = int(blk[f'cp{which}_mode_year'])
        hdi = blk[f'cp{which}_hdi94']
        ax.plot([mode, mode], [0, strip_hi], color=S.INK, linewidth=0.9, linestyle=(0, (3, 2.5)), zorder=6)
        side = -1 if which == 1 else 1
        ax.annotate(f'{mode}  [{hdi[0]}-{hdi[1]}]', xy=(mode, strip_hi), xytext=(4 * side, 2), textcoords='offset points', ha='right' if side < 0 else 'left', va='bottom', fontsize=7.2, color=S.INK, fontweight='semibold')
    ax.set_ylim(0, 1.44)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
    ax.set_xlim(years.min() - 1.1, years.max() + 1.1)
    ax.set_ylabel('Share of recalls that year')
    S.tidy(ax, grid='y')
    ax.spines['left'].set_bounds(0, 1.0)
    ax.annotate('posterior of the\nchange-point year' if preferred else 'K = 1 shown; LOO prefers\nno change point', xy=(years.min() - 0.4, strip_lo + 0.02), xytext=(0, 0), textcoords='offset points', ha='left', va='bottom', fontsize=7.0, color=S.BLUE, linespacing=1.25)
    S.panel_title(ax, 'A  Hazard composition of apparel recalls, by year', f"Bars: share of that year's recalls.  n = {int(d['haz'].n.sum())} recalls, {len(years)} years")
    free_text(ax)

def panel_b(ax, d):
    cp, s = (d['cp'], d['summary'])
    lo, hi = (3000, 0)
    curves = []
    for key in ['hazard', 'channel', 'violation']:
        K, w = best_cp(d, key)
        if K == 0 or w.empty:
            continue
        for which in range(1, K + 1):
            ww = w[w.cp == which]
            p = ww.prob.to_numpy(float)
            if p.sum() <= 0:
                continue
            yrs = ww.year.to_numpy(int)
            sel = yrs[p > 0.004]
            if len(sel):
                lo, hi = (min(lo, sel.min()), max(hi, sel.max()))
            curves.append((key, which, yrs, p))
    lo, hi = (max(lo - 2, 1974), hi + 2)
    peak = {}
    for key, which, yrs, p in curves:
        col = SERIES_COLOR[key]
        ax.fill_between(yrs, 0, p, color=col, alpha=0.3, linewidth=0, zorder=2)
        ax.plot(yrs, p, color=col, linewidth=1.8, zorder=3, linestyle='-' if which == 1 else (0, (4, 2)))
        i = int(np.argmax(p))
        peak[key, which] = (int(yrs[i]), float(p[i]))
    ymax = max((v[1] for v in peak.values())) if peak else 1.0
    nudge = {'hazard': (0, 'center'), 'channel': (-7, 'right'), 'violation': (7, 'left')}
    for (key, which), (yr, pk) in peak.items():
        dx, ha = nudge[key]
        ax.annotate(f'{yr}', xy=(yr, pk), xytext=(dx, 3), textcoords='offset points', ha=ha, va='bottom', fontsize=7.0, color=SERIES_COLOR[key], fontweight='semibold')
    handles = [Line2D([], [], color=SERIES_COLOR[k], linewidth=1.8, label=SERIES_LABEL[k]) for k in SERIES_COLOR if any((c[0] == k for c in curves))]
    lg = ax.legend(handles=handles, loc='upper right', frameon=False, fontsize=7.2, handlelength=1.4, borderaxespad=0.2, labelspacing=0.35)
    for t in lg.get_texts():
        t.set_color(S.INK)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, ymax * 1.48)
    ax.set_ylabel('Posterior probability')
    ax.set_xlabel('Year the new regime starts')
    S.tidy(ax, grid='y')
    co = s.get('changepoint_coincidence', {})
    bits = []
    for k, v in co.items():
        a, b = k.split('_vs_')
        c = v['closest']
        short = {'hazard': 'hazard', 'channel': 'online', 'violation': 'violation'}
        bits.append(f"{short[a]} vs {short[b]}  {c['mean_years']:+.1f} yr (P {c['p_within_2_years']:.2f})")
    if bits:
        ax.annotate('Closest pair of breaks; P = P(within 2 years)\n' + '\n'.join(bits), xy=(0.008, 0.97), xycoords='axes fraction', ha='left', va='top', fontsize=6.8, color=S.INK_2, linespacing=1.45)
    S.panel_title(ax, 'B  Do the three breaks coincide?', 'Posterior change-point year; dashed = second break')
    free_text(ax)

def panel_c(ax, d):
    comp = d['comp']
    K = hazard_K(d)[0]
    c = comp[(comp.series == 'hazard') & (comp.K == K) & (comp.category != '_concentration')].copy()
    order = c[c.regime == K + 1].sort_values('mean').category.tolist()
    y0 = {cat: i for i, cat in enumerate(order)}
    blk = d['summary']['changepoints']['hazard'][f'K{K}']
    years = d['haz'].year.to_numpy(int)
    edges = [years.min()] + [int(blk[f'cp{w}_mode_year']) for w in range(1, K + 1)]
    spans = []
    for r in range(K + 1):
        a = edges[r]
        b = edges[r + 1] - 1 if r < K else years.max()
        spans.append(f'{a}-{b}')
    h = 0.19

    def ypos(cat, regime):
        return y0[cat] + (1 - (int(regime) - 1)) * h
    for cat in order:
        rows = c[c.category == cat].sort_values('regime')
        ax.plot(rows['mean'], [ypos(cat, r) for r in rows.regime], color=S.GRAY_4, linewidth=0.9, zorder=1)
    for _, row in c.iterrows():
        r = int(row.regime) - 1
        yy = ypos(row.category, row.regime)
        ax.plot([row.hdi_lo, row.hdi_hi], [yy, yy], color=REGIME_COLOR[r], linewidth=1.5, alpha=0.5, zorder=3, solid_capstyle='butt')
        ax.plot([row['mean']], [yy], marker='o', markersize=4.2, color=REGIME_COLOR[r], zorder=4)
    for _, row in c[c.regime == K + 1].iterrows():
        if row['mean'] >= 0.02:
            ax.annotate(f"{row['mean']:.0%}", xy=(row.hdi_hi, ypos(row.category, row.regime)), xytext=(4, 0), textcoords='offset points', ha='left', va='center', fontsize=6.8, color=S.INK_2, zorder=6, bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.75))
    ax.set_yticks(list(y0.values()))
    ax.set_yticklabels([S.HAZARD_LABELS[c_] for c_ in order], fontsize=7.4)
    ax.set_ylim(-0.65, len(order) - 0.35)
    ax.set_xlim(0, min(1.0, c.hdi_hi.max() * 1.16))
    ax.xaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
    ax.set_xlabel('Share of recalls in the regime')
    S.tidy(ax, grid='x')
    handles = [Line2D([], [], color=REGIME_COLOR[r], marker='o', linestyle='none', markersize=4.2, label=f'Regime {r + 1}  {spans[r]}') for r in range(K + 1)]
    leg = ax.legend(handles=handles, loc='lower right', frameon=False, fontsize=7.2, handlelength=1.0, borderaxespad=0.4, labelspacing=0.35)
    for t in leg.get_texts():
        t.set_color(S.INK)
    S.panel_title(ax, 'C  What each regime is made of', 'Posterior mean and 94% HDI; dots joined 1 -> 3')
    free_text(ax)

def build(fig):
    d = load()
    fig.get_layout_engine().set(hspace=0.085, wspace=0.04)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.1, 0.26, 0.86, 1.3])
    panel_a(fig.add_subplot(gs[0]), d)
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis('off')
    panel_b(fig.add_subplot(gs[2]), d)
    panel_c(fig.add_subplot(gs[3]), d)
    cats = [c for c in S.HAZARD_ORDER if c in d['haz'].columns]
    handles = [Patch(facecolor=S.HAZARD_COLORS[c], label=SHORT[c]) for c in cats]
    leg = ax_leg.legend(handles=handles, loc='upper center', ncol=5, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, 1.05), handlelength=1.0, handleheight=1.0, columnspacing=1.6, handletextpad=0.5)
    leg.set_in_layout(False)
    for t in leg.get_texts():
        t.set_color(S.INK)
    s = d['summary']
    K = hazard_K(d)[0]
    alt = d['cmp'][(d['cmp'].series == 'hazard') & (d['cmp'].K == K - 1)]
    dtxt = '' if alt.empty else f' (d elpd {float(alt.d_elpd.iloc[0]):.0f} +/- {float(alt.d_elpd_se.iloc[0]):.0f} against K = {K - 1})'
    note = f"CPSC recalls of apparel and home textiles, {s['series']['hazard']['year_range'][0]}-{s['series']['hazard']['year_range'][1]} (n = {s['series']['hazard']['n_records']}). Each year is a Dirichlet-multinomial draw from its regime's composition; change points are discrete year parameters, marginalised exactly (minimum regime 3 years). PSIS-LOO prefers K = {K}{dtxt}. The channel and violation series begin in {s['series']['channel']['year_range'][0]}. Recall counts are enforcement outputs: these date a change in what CPSC detected and published, not necessarily in the products."
    S.source_note(fig, '\n'.join(textwrap.wrap(note, 130)), y=0.004)

def main(quick: bool=False):
    out = S.save_figure(build, str(HERE / 'figure_bayes_changepoint'), height_in=8.1)
    print('[15] wrote ' + ', '.join((Path(p).name for p in out)))
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    main(**vars(ap.parse_args()))
