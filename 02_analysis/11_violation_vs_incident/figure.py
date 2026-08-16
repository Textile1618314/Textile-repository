from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import data as D
from _common import style as S
HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
HEIGHT_IN = 7.0
C_VIOLATION = S.PURPLE
C_INJURY = S.AMBER
C_BOTH = S.RED
C_NEITHER = S.GRAY_3
CLASS_ORDER = ['violation_only', 'both', 'injury_only', 'neither']
CLASS_COLORS = {'violation_only': C_VIOLATION, 'both': C_BOTH, 'injury_only': C_INJURY, 'neither': C_NEITHER}
CLASS_LABELS = {'violation_only': 'Violation, no injury', 'both': 'Violation and injury', 'injury_only': 'Injury, no violation', 'neither': 'Neither recorded'}
PERIODS = ['1974-89', '1990-99', '2000-09', '2010-19', '2020-26']
HAZ_SHORT = {'flammability_burn': 'Flammability', 'choking_small_parts': 'Choking', 'drawstring_strangulation': 'Drawstring', 'chemical': 'Chemical', 'fall_slip': 'Fall / slip', 'laceration_puncture': 'Laceration', 'entrapment_entanglement': 'Entrapment', 'protective_failure': 'Protective', 'other': 'Other'}
X_LIM = 2026.4

def title_block(ax, title, subtitle, pad=12, gap=15):
    S.panel_title(ax, title, subtitle, pad=pad)

def freeze_free_text(fig):
    for ax in fig.axes:
        for t in ax.texts:
            t.set_in_layout(False)

def load():
    by_year = pd.read_csv(RES / 'posterior_by_year.csv')
    per = pd.read_csv(RES / 'period_classification.csv')
    haz = pd.read_csv(RES / 'violation_by_hazard.csv')
    summ = D.read_json(RES / 'summary.json')
    return (by_year, per, haz, summ)

def panel_a(ax, by_year, summ):
    spec = [('violation', C_VIOLATION, 'P(violation | recall)', 'Share of recalls that\ncite a standards violation'), ('injury', C_INJURY, 'P(injury reported | recall)', 'Share of recalls that\nreport any injury')]
    ends = {}
    for key, colour, _, _ in spec:
        s = by_year[by_year.series == key].sort_values('year')
        ax.fill_between(s.year, s.hdi_lo, s.hdi_hi, color=colour, alpha=0.17, linewidth=0, zorder=1)
        ax.plot(s.year, s.observed, 'o', color=colour, markersize=3.0, alpha=0.42, markeredgewidth=0, zorder=2)
        ax.plot(s.year, s.posterior_mean, color=colour, linewidth=2.4, zorder=3, solid_capstyle='round')
        ax.plot(s.year.iloc[-1], s.posterior_mean.iloc[-1], 'o', color=colour, markersize=5.5, markeredgecolor=S.SURFACE, markeredgewidth=1.6, zorder=4)
        ends[key] = (float(s.year.iloc[-1]), float(s.posterior_mean.iloc[-1]))
    ax.set_xlim(2000, X_LIM)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '25%', '50%', '75%', '100%'])
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    ax.set_xticklabels(['2000', '2005', '2010', '2015', '2020', '2025'])
    S.tidy(ax, grid='y')
    handles = [Line2D([], [], color=c, linewidth=2.4, label=short) for _, c, short, _ in spec]
    handles.append(Line2D([], [], color=S.GRAY_2, marker='o', linestyle='none', markersize=3.6, label='observed annual share'))
    leg = ax.legend(handles=handles, loc='upper left', ncol=1, frameon=False, fontsize=7.2, handlelength=1.4, borderaxespad=0.0, labelspacing=0.34, bbox_to_anchor=(0.005, 1.0))
    for t in leg.get_texts():
        t.set_color(S.INK_2)
    title_block(ax, 'A   Recalls stopped following injuries and started following tests', 'Hierarchical logistic random walk; ribbons are 94% HDIs, dots the observed annual shares', pad=15)

def panel_b(ax, per):
    per = per.set_index('period').reindex(PERIODS)
    x = np.arange(len(PERIODS), dtype=float)
    shares = np.vstack([per[f'share_{c}'].to_numpy(float) for c in CLASS_ORDER])
    bottom = np.zeros(len(x))
    mids = {}
    for row, c in zip(shares, CLASS_ORDER):
        ax.fill_between(x, bottom, bottom + row, color=CLASS_COLORS[c], linewidth=1.2, edgecolor=S.SURFACE, zorder=2)
        mids[c] = bottom[-1] + row[-1] / 2
        bottom = bottom + row
    for c in CLASS_ORDER:
        y = mids[c]
        share = shares[CLASS_ORDER.index(c)][-1]
        if share >= 0.1:
            ax.text(len(x) - 1 - 0.04, y, f'{share:.0%}', ha='right', va='center', fontsize=7.8, color=S.INK, fontweight='semibold', zorder=6, bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.8))
    ax.set_xlim(0, len(x) - 1)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '25%', '50%', '75%', '100%'])
    ax.set_xticks(x)
    ax.set_xticklabels([f'{p}\nn = {int(per.n.iloc[j])}' for j, p in enumerate(PERIODS)], fontsize=7.6, color=S.INK_2, linespacing=1.4)
    S.tidy(ax, grid='y')
    handles = [Patch(facecolor=CLASS_COLORS[c], label=CLASS_LABELS[c]) for c in CLASS_ORDER]
    leg = ax.legend(handles=handles, ncol=2, loc='upper left', frameon=True, fontsize=7.2, handlelength=1.0, handleheight=1.0, labelspacing=0.42, columnspacing=1.2, borderaxespad=0.0, bbox_to_anchor=(0.012, 0.985))
    leg.get_frame().set_facecolor(S.SURFACE)
    leg.get_frame().set_edgecolor('none')
    leg.get_frame().set_alpha(0.88)
    for t in leg.get_texts():
        t.set_color(S.INK)
    title_block(ax, 'B   What a recall is evidence of, by period', 'Every apparel and home-textile recall 1974-2026, classified on the two flags jointly', pad=18)

def panel_c(ax, haz):
    v = haz[haz.series == 'violation'].set_index('hazard')
    i = haz[haz.series == 'injury'].set_index('hazard')
    order = [h for h in S.HAZARD_ORDER if h in v.index]
    order = sorted(order, key=lambda h: v.loc[h, 'share'])
    y = np.arange(len(order), dtype=float)
    for j, h in enumerate(order):
        colour = S.HAZARD_COLORS[h]
        ax.plot([v.loc[h, 'ci_lo'], v.loc[h, 'ci_hi']], [j, j], color=colour, linewidth=2.0, alpha=0.42, solid_capstyle='round', zorder=2)
        ax.plot(i.loc[h, 'share'], j, 'o', color=S.SURFACE, markersize=7.0, markeredgecolor=S.GRAY_2, markeredgewidth=1.5, zorder=3)
        ax.plot(v.loc[h, 'share'], j, 'o', color=colour, markersize=8.0, markeredgecolor=S.SURFACE, markeredgewidth=1.4, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([HAZ_SHORT[h] for h in order], fontsize=7.6, color=S.INK_2)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlim(-0.02, 1.3)
    for j, h in enumerate(order):
        ax.text(1.29, j, f"n = {int(v.loc[h, 'n'])}", fontsize=7.0, color=S.INK_3, ha='right', va='center')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0', '25%', '50%', '75%', '100%'])
    S.tidy(ax, grid='x')
    ax.grid(axis='y', visible=False)
    top = order[-1]
    ax.annotate('94% credible interval', xy=(v.loc[top, 'ci_hi'] + 0.03, len(order) - 1 + 0.42), fontsize=7.0, color=S.INK_3, ha='left', va='center')
    handles = [Line2D([], [], color=S.GRAY_1, marker='o', linestyle='none', markersize=7.0, label='share citing a standards violation'), Line2D([], [], color=S.SURFACE, marker='o', linestyle='none', markersize=6.5, markeredgecolor=S.GRAY_2, markeredgewidth=1.5, label='share reporting an injury')]
    leg = ax.legend(handles=handles, loc='lower left', ncol=1, frameon=False, fontsize=7.0, handlelength=1.0, borderaxespad=0.0, labelspacing=0.45, bbox_to_anchor=(0.36, 0.005))
    for t in leg.get_texts():
        t.set_color(S.INK_2)
    title_block(ax, 'C   Violation and injury do not travel together', 'By hazard category, 2000-2026; dot colour is the fixed hazard hue used throughout the paper', pad=22)

def note(ax, summ):
    ax.axis('off')

def build(fig, by_year, per, haz, summ):
    gs = fig.add_gridspec(4, 1, height_ratios=[2.46, 1.7, 1.94, 0.42])
    panel_a(fig.add_subplot(gs[0]), by_year, summ)
    panel_b(fig.add_subplot(gs[1]), per)
    panel_c(fig.add_subplot(gs[2]), haz)
    fig.add_subplot(gs[3]).axis('off')
    S.source_note(fig, f"CPSC apparel and home-textile recalls, n = {summ['n_total']} (apparel_recalls_v2); panels A and C use the {summ['n_window']} records from 2000 onward.\nis_violation and injuries_reported are coded from CPSC text, so both are lower bounds; pre-2000 records carry little incident text.")
    freeze_free_text(fig)

def main(quick=False):
    by_year, per, haz, summ = load()
    out = S.save_figure(lambda f: build(f, by_year, per, haz, summ), str(HERE / 'figure_violation_vs_incident'), height_in=HEIGHT_IN)
    print('[11] wrote ' + ', '.join((Path(p).name for p in out)))
    return out
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    main(**vars(ap.parse_args()))
