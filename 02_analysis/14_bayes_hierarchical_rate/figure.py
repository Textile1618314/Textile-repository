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
RES = HERE / 'results'
C_HI, C_LO = (S.AMBER, S.BLUE)
MIN_RECALLS_A, MIN_BN_A = (2, 2.0)
RIDGE = ['India', 'Pakistan', 'Hong Kong', 'Philippines', 'Thailand', 'Indonesia', 'Vietnam', 'Bangladesh', 'Cambodia', 'Mexico']

def load():
    return {'summary': json.loads((RES / 'bayes_rate_summary.json').read_text()), 'rates': pd.read_csv(RES / 'country_rates.csv'), 'cmp': pd.read_csv(RES / 'model_comparison.csv'), 'rr': pd.read_csv(RES / 'rate_ratio_draws.csv'), 'ppc': pd.read_csv(RES / 'ppc_stats.csv'), 'rep': pd.read_csv(RES / 'ppc_replicates.csv')}

def free_text(ax):
    for t in ax.texts:
        t.set_in_layout(False)

def hue(row):
    return C_HI if row.rr_vs_reference_mean > 1 else C_LO

def panel_a(ax, d):
    r = d['rates']
    keep = r[(r.recalls >= MIN_RECALLS_A) | (r.bn_sme >= MIN_BN_A)].copy()
    keep = keep.sort_values('post_rate_mean')
    y = np.arange(len(keep))
    grand = d['summary']['shrinkage']['grand_mean_rate_per_bn_sme']
    ax.axvline(grand, color=S.GRAY_2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    xmin = max(min(keep.hdi_lo.min(), keep.raw_rate[keep.raw_rate > 0].min()) * 0.55, 0.001)
    xmax = max(keep.hdi_hi.max(), keep.raw_rate.max()) * 1.35
    for yi, (_, row) in zip(y, keep.iterrows()):
        c = hue(row)
        ax.plot([row.hdi_lo, row.hdi_hi], [yi, yi], color=c, linewidth=1.9, alpha=0.42, solid_capstyle='butt', zorder=2)
        if row.raw_rate > 0:
            ax.plot([row.raw_rate], [yi], marker='o', markersize=5.2, markerfacecolor='none', markeredgecolor=c, markeredgewidth=1.1, zorder=3)
        else:
            ax.plot([xmin * 1.25], [yi], marker='4', markersize=6, color=S.GRAY_2, zorder=3)
        ax.plot([row.post_rate_mean], [yi], marker='o', markersize=5.6, color=c, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{c}  {int(n)}/{v:,.1f}' if v < 10 else f'{c}  {int(n)}/{v:,.0f}' for c, n, v in zip(keep.country, keep.recalls, keep.bn_sme)], fontsize=6.8)
    for t, (_, row) in zip(ax.get_yticklabels(), keep.iterrows()):
        t.set_color(S.INK)
        if False:
            t.set_fontweight('semibold')
    ax.set_xscale('log')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.9, len(keep) + 0.55)
    ax.set_xlabel('Recalls per bn SME (log)')
    S.tidy(ax, grid='x')
    ax.annotate(f'grand mean {grand:.2f}', xy=(grand, len(keep) - 0.15), xytext=(3, 0), textcoords='offset points', ha='left', va='bottom', fontsize=7.0, color=S.GRAY_1)
    S.panel_title(ax, 'A  Recall rate by origin', 'Row label: recalls / bn SME imported')
    free_text(ax)

def panel_b(ax, d):
    r = d['rates']
    grand = d['summary']['shrinkage']['grand_mean_rate_per_bn_sme']
    pos = r[r.raw_rate > 0].copy()
    lim = [min(pos.raw_rate.min(), pos.post_rate_mean.min()) * 0.5, max(pos.raw_rate.max(), pos.post_rate_mean.max()) * 1.75]
    ax.plot(lim, lim, color=S.GRAY_2, linewidth=1.0, zorder=1)
    ax.axhline(grand, color=S.GRAY_3, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    size = 6 + 46 * np.sqrt(pos.bn_sme / pos.bn_sme.max())
    ax.scatter(pos.raw_rate, pos.post_rate_mean, s=size, color=[hue(row) for _, row in pos.iterrows()], alpha=0.85, linewidth=0, zorder=3)
    callout = {'Peru': (3.3, 21.0), 'China': (0.42, 6.6), 'Vietnam': (1.75, 0.115), 'Bangladesh': (2.3, 0.3)}
    for _, row in pos[pos.country.isin(callout)].iterrows():
        tx, ty = callout[row.country]
        ax.annotate(row.country, xy=(row.raw_rate, row.post_rate_mean), xytext=(tx, ty), textcoords='data', ha='left', va='center', fontsize=7.0, color=S.INK, zorder=5, arrowprops=dict(arrowstyle='-', color=S.GRAY_1, linewidth=0.6, shrinkA=2, shrinkB=6.0))
    sw = pos[pos.country == 'Sweden']
    if len(sw):
        r0 = sw.iloc[0]
        ax.annotate('Sweden', xy=(r0.raw_rate, r0.post_rate_mean), xytext=(-34, 2), textcoords='offset points', ha='right', va='center', fontsize=7.0, color=S.INK, zorder=5, arrowprops=dict(arrowstyle='-', color=S.GRAY_1, linewidth=0.6, shrinkA=1, shrinkB=2.5))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel('Raw rate per bn SME')
    ax.set_ylabel('Posterior rate')
    S.tidy(ax, grid='both')
    ax.annotate('grand mean', xy=(lim[1] * 0.92, grand), xytext=(0, 3), textcoords='offset points', ha='right', va='bottom', fontsize=6.8, color=S.GRAY_1)
    ax.scatter([lim[1] * 0.14], [lim[0] * 2.1], s=52, color=S.GRAY_3, linewidth=0, zorder=3)
    ax.annotate('marker area = import volume', xy=(lim[1] * 0.14, lim[0] * 2.1), xytext=(-8, 0), textcoords='offset points', ha='right', va='center', fontsize=6.8, color=S.INK_3)
    S.panel_title(ax, 'B  Shrinkage', 'Diagonal = no shrinkage')
    free_text(ax)

def panel_c(ax, d):
    rr, rates = (d['rr'], d['rates'].set_index('country'))
    order = [c for c in RIDGE if c in rr.columns]
    order = sorted(order, key=lambda c: rates.loc[c, 'rr_vs_reference_mean'])
    lo, hi = (-2.2, 1.4)
    grid = np.linspace(lo, hi, 220)
    step = 1.0
    for i, c in enumerate(order):
        x = np.log10(np.clip(rr[c].to_numpy(float), 0.0001, None))
        dens, edges = np.histogram(x, bins=48, range=(lo, hi), density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        v = np.interp(grid, centres, dens)
        v = np.convolve(v, np.ones(9) / 9, mode='same')
        v = v / v.max() * 0.78
        base = i * step
        col = C_LO if rates.loc[c, 'rr_vs_reference_mean'] < 1 else C_HI
        ax.fill_between(grid, base, base + v, color=col, alpha=0.55, linewidth=0, zorder=2 + i)
        ax.plot(grid, base + v, color=col, linewidth=1.0, zorder=2 + i)
        ax.annotate(c, xy=(lo, base + 0.12), xytext=(2, 0), textcoords='offset points', ha='left', va='bottom', fontsize=7.2, color=S.INK, zorder=40)
        ax.annotate(f"{rates.loc[c, 'p_rate_below_reference']:.2f}", xy=(hi, base + 0.12), xytext=(-1, 0), textcoords='offset points', ha='right', va='bottom', fontsize=7.2, color=col, zorder=40, fontweight='semibold')
    ax.axvline(0.0, color=S.GRAY_1, linewidth=1.4, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.12, len(order) * step + 0.5)
    ticks = [-2, -1, 0, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(['0.01x', '0.1x', 'mean', '10x'])
    ax.set_yticks([])
    S.tidy(ax, grid='x')
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Rate ratio, posterior')
    ax.annotate('P(rate < mean)', xy=(hi, len(order) * step + 0.12), xytext=(-1, 0), textcoords='offset points', ha='right', va='bottom', fontsize=7.0, color=S.INK_3)
    S.panel_title(ax, 'C  Rate ratio vs the grand mean', 'Posterior, log scale')
    free_text(ax)

def panel_d(ax, d):
    ax.axis('off')
    c = d['cmp'].copy()
    best = d['summary']['best_model']
    labs = {'M1': 'pooled Poisson', 'M2': 'hier. Poisson', 'M3': 'hier. NB', 'M4': 'hier. NB + year RW', 'M5': 'hier. NB + period'}
    rows = [('', 'model', 'elpd', 'd elpd', 'k-hat')]
    for _, r in c.iterrows():
        d_txt = 'best' if r.model == best else f'{r.d_elpd:,.0f} ({r.d_elpd_se:.0f})'
        rows.append((r.model, labs.get(r.model, ''), f'{r.elpd_loo:,.0f}', d_txt, f'{r.khat_max:.2f}'))
    xs = [0.0, 0.1, 0.64, 0.88, 1.0]
    xs_head = [0.0, 0.1, 0.64, 0.855, 1.0]
    has = ['left', 'left', 'right', 'right', 'right']
    for i, row in enumerate(rows):
        yy = 0.88 - i * 0.158
        head = i == 0
        for x, txt, ha in zip(xs_head if head else xs, row, has):
            ax.text(x, yy, txt, ha=ha, va='center', fontsize=6.9, color=S.INK_3 if head else S.INK, fontweight='semibold' if head or row[0] == best else 'normal', transform=ax.transAxes)
        if head:
            ax.plot([0, 1.0], [yy - 0.075] * 2, color=S.RULE, linewidth=0.9, transform=ax.transAxes, clip_on=False)
    S.panel_title(ax, 'D  Model comparison', 'PSIS-LOO, higher is better')
    free_text(ax)

def panel_e(ax, d):
    rep, ppc = (d['rep'], d['ppc'].set_index('statistic'))
    stats = [('share_zero_cells', 'Empty cells', '{:.3f}'), ('var_mean_ratio', 'Variance / mean', '{:.1f}'), ('max_cell_count', 'Largest cell', '{:.0f}')]
    for i, (key, label, fmt) in enumerate(stats):
        v = rep[key].to_numpy(float)
        obs = float(ppc.loc[key, 'observed'])
        p = float(ppc.loc[key, 'bayes_p'])
        lo, hi = (min(v.min(), obs), max(v.max(), obs))
        pad = 0.12 * (hi - lo + 1e-09)
        lo, hi = (lo - pad, hi + pad)
        dens, edges = np.histogram(v, bins=26, range=(lo, hi), density=True)
        xg = 0.5 * (edges[:-1] + edges[1:])
        xn = (xg - lo) / (hi - lo)
        dn = dens / dens.max() * 0.58
        base = 2 - i
        ax.fill_between(xn, base, base + dn, color=S.SEQ_BLUE[2], alpha=0.9, linewidth=0)
        ax.plot([(obs - lo) / (hi - lo)] * 2, [base, base + 0.66], color=S.PINK, linewidth=1.8, solid_capstyle='butt', zorder=4)
        ax.annotate(f'{label}: observed {fmt.format(obs)},  p = {p:.2f}', xy=(0.0, base + 0.68), xytext=(0, 0), textcoords='offset points', ha='left', va='bottom', fontsize=6.9, color=S.INK)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.1, 2.95)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ('top', 'right', 'left', 'bottom'):
        ax.spines[sp].set_visible(False)
    S.panel_title(ax, 'E  Predictive checks', 'Blue replicated, pink observed')
    free_text(ax)

def build(fig):
    d = load()
    fig.get_layout_engine().set(hspace=0.075, wspace=0.03)
    gs = fig.add_gridspec(5, 2, width_ratios=[1.0, 1.0], height_ratios=[1.45, 1.22, 0.72, 0.85, 0.55])
    panel_a(fig.add_subplot(gs[:4, 0]), d)
    panel_b(fig.add_subplot(gs[0, 1]), d)
    panel_c(fig.add_subplot(gs[1, 1]), d)
    panel_d(fig.add_subplot(gs[2, 1]), d)
    panel_e(fig.add_subplot(gs[3, 1]), d)
    ax_leg = fig.add_subplot(gs[4, 0])
    ax_leg.axis('off')
    s = d['summary']
    p, sens = (s['panel'], s['sensitivity_no_suspect_year'])
    hp = s['hyperparameters']
    handles = [Line2D([], [], color=C_HI, marker='o', linestyle='none', markersize=5.5, label='Rate above the grand mean'), Line2D([], [], color=C_LO, marker='o', linestyle='none', markersize=5.5, label='Rate below the grand mean'), Line2D([], [], color=S.GRAY_1, marker='o', linestyle='none', markersize=5.5, markerfacecolor='none', label='Raw count / volume')]
    leg = ax_leg.legend(handles=handles, loc='upper left', ncol=2, frameon=False, fontsize=7.2, bbox_to_anchor=(0.0, 1.0), handlelength=1.0, columnspacing=1.2, handletextpad=0.5, labelspacing=0.5)
    leg.set_in_layout(False)
    for t in leg.get_texts():
        t.set_color(S.INK)
    note = f"CPSC recalls of apparel and home textiles by country of manufacture, {p['years'][0]}-{p['years'][1]}, against OTEXA category 1 apparel imports: {p['n_recalls_modelled']} recalls over {p['n_panel_rows']:,} country-year cells ({p['n_countries_modelled']} countries, {p['share_zero_cells']:.0%} empty). A recall counts once per country named; US-origin recalls have no import denominator and are excluded. Rates are from {s['best_model']} (hierarchical NB with a year random walk): country log rates theta ~ N(mu, tau), tau = {hp['tau_country_sd_of_log_rate']['mean']:.2f}, 4 chains of blocked Metropolis, max Rhat {max((m['max_rhat'] for m in s['model_diagnostics'])):.3f}. Panel A shows countries with >= {MIN_RECALLS_A} recalls or >= {MIN_BN_A:.0f} bn SME; all {p['n_countries_modelled']} are in country_rates.csv. Dropping the duplicated OTEXA 2024 column moves every country rate by at most {sens['max_abs_pct_change_in_country_rate']:.0f}% (Spearman {sens['spearman_with_main']:.3f})."
    S.source_note(fig, '\n'.join(textwrap.wrap(note, 124)), y=0.004)

def main(quick: bool=False):
    out = S.save_figure(build, str(HERE / 'figure_bayes_hierarchical_rate'), height_in=8.6)
    print('[14] wrote ' + ', '.join((Path(p).name for p in out)))
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    main(**vars(ap.parse_args()))
