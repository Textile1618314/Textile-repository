from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import data as D
from _common import style as S
HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
PERIODS = ['2000-09', '2010-19', '2020-26']
GROUPS = [('online_only', 'Online only'), ('store_or_mixed', 'Stores/mixed')]
HEIGHT_IN = 7.6
DIVMAP = LinearSegmentedColormap.from_list('resid', S.DIV[::-1])

def title_block(ax, title, subtitle, pad=12, gap=15):
    S.panel_title(ax, title, subtitle, pad=pad)

def freeze_free_text(fig):
    for ax in fig.axes:
        for t in ax.texts:
            t.set_in_layout(False)
        for t in getattr(ax, '_children', []):
            if t.__class__.__name__ == 'Annotation':
                t.set_in_layout(False)

def load():
    comp = pd.read_csv(RES / 'composition_panelA.csv')
    roll = pd.read_csv(RES / 'rolling_shares.csv')
    roll['plotted'] = roll['plotted'].astype(str).str.lower().eq('true')
    resid = pd.read_csv(RES / 'residuals_2020_26.csv')
    summ = D.read_json(RES / 'summary.json')
    return (comp, roll, resid, summ)

def panel_a(ax, comp, summ):
    hazards = [h for h in S.HAZARD_ORDER if h in set(comp.hazard)]
    lookup = comp.set_index(['period', 'channel_group', 'hazard']).share.to_dict()
    ns = comp.set_index(['period', 'channel_group']).group_n.to_dict()
    slots, ticks, ticklabels, spans = ([], [], [], [])
    x = 0.0
    for p in PERIODS:
        pair = []
        for g, gl in GROUPS:
            slots.append((p, g, x))
            ticks.append(x)
            ticklabels.append(f'{gl}\nn = {int(ns[p, g])}')
            pair.append(x)
            x += 1.04
        spans.append((p, pair[0], pair[-1]))
        x += 0.62
    positions = [s[2] for s in slots]
    series = {h: np.array([lookup.get((per, grp, h), 0.0) for per, grp, _ in slots]) for h in hazards}
    S.stacked_bars(ax, positions, series, S.HAZARD_COLORS, width=0.84, labels=S.HAZARD_LABELS, min_label=0.085)
    ax.set_ylim(0, 1.2)
    ax.set_xlim(-0.66, positions[-1] + 0.66)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '25%', '50%', '75%', '100%'])
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=7.4, color=S.INK_2, linespacing=1.4)
    S.tidy(ax, grid='y')
    ax.set_axisbelow(True)
    for per, x0, x1 in spans:
        ax.plot([x0 - 0.46, x1 + 0.46], [1.055, 1.055], color=S.AXIS, linewidth=0.9, solid_capstyle='butt')
        ax.text((x0 + x1) / 2, 1.085, per, ha='center', va='bottom', fontsize=9.5, color=S.INK, fontweight='semibold')
    title_block(ax, 'A   The flammability regime is a channel regime', 'Hazard mix of recalls, by period and channel; flammability anchored at the baseline of every bar', pad=12)

def panel_a_legend(ax, comp):
    ax.axis('off')
    hazards = [h for h in S.HAZARD_ORDER if h in set(comp.hazard)]
    handles = [Patch(facecolor=S.HAZARD_COLORS[h], label=S.HAZARD_LABELS[h]) for h in hazards]
    leg = ax.legend(handles=handles, ncol=3, loc='center', frameon=False, fontsize=7.8, handlelength=1.0, handleheight=1.0, columnspacing=1.8, labelspacing=0.55, borderaxespad=0.0)
    for t in leg.get_texts():
        t.set_color(S.INK)
SERIES_B = [('flam_given_online_only', S.PINK, 'P(flammability | online-only)', 'Flammability share of\nonline-only recalls'), ('online_only_given_flam', S.GRAY_1, 'P(online-only | flammability)', 'Online-only share of\nflammability recalls'), ('flam_given_store_only', S.BLUE, 'P(flammability | stores-only)', 'Flammability share of\nstores-only recalls')]
X_LIM = 2027.2

def panel_b(ax, roll):
    ends = {}
    for key, colour, _, _ in SERIES_B:
        s = roll[(roll.quantity == key) & roll['plotted']].sort_values('year')
        ax.fill_between(s.year, s.ci_lo, s.ci_hi, color=colour, alpha=0.16, linewidth=0, zorder=1)
        ax.plot(s.year, s.estimate, color=colour, linewidth=2.2, zorder=3, solid_capstyle='round')
        ax.plot(s.year.iloc[-1], s.estimate.iloc[-1], 'o', color=colour, markersize=5.5, markeredgecolor=S.SURFACE, markeredgewidth=1.6, zorder=4)
        ends[key] = (float(s.year.iloc[-1]), float(s.estimate.iloc[-1]))
    ax.set_xlim(2000, X_LIM)
    ax.set_ylim(-0.03, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '25%', '50%', '75%', '100%'])
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    ax.set_xticklabels(['2000', '2005', '2010', '2015', '2020', '2025'])
    S.tidy(ax, grid='y')
    handles = [Line2D([], [], color=c, linewidth=2.2, label=short) for _, c, short, _ in SERIES_B]
    leg = ax.legend(handles=handles, loc='upper left', ncol=1, frameon=False, fontsize=7.2, handlelength=1.4, borderaxespad=0.0, labelspacing=0.34, bbox_to_anchor=(0.005, 1.0))
    for t in leg.get_texts():
        t.set_color(S.INK_2)
    title_block(ax, 'B   The two conditional shares climb together', '7-year centred window, plotted where the window holds 8+ recalls; bands are 94% bootstrap intervals', pad=15)

def panel_c(ax, resid):
    channels = ['online_only', 'mixed', 'store_only']
    hazards = [h for h in S.HAZARD_ORDER if h in set(resid.hazard)]
    M = np.array([[float(resid[(resid.channel == c) & (resid.hazard == h)].std_residual.iloc[0]) for h in hazards] for c in channels])
    vmax = float(np.ceil(np.abs(M).max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    ax.pcolormesh(np.arange(len(hazards) + 1), np.arange(len(channels) + 1), M, cmap=DIVMAP, norm=norm, edgecolors=S.SURFACE, linewidth=2.0)
    for i in range(len(channels)):
        for j in range(len(hazards)):
            v = M[i, j]
            ax.text(j + 0.5, i + 0.5, f'{v:+.1f}', ha='center', va='center', fontsize=7.6, fontweight='semibold', color='white' if abs(v) > 0.52 * vmax else S.INK)
    ax.set_xticks(np.arange(len(hazards)) + 0.5)
    ax.set_xticklabels([S.HAZARD_LABELS[h].replace(' / ', '/\n') for h in hazards], fontsize=7.1, color=S.INK_2, linespacing=1.25)
    ax.set_yticks(np.arange(len(channels)) + 0.5)
    ax.set_yticklabels([S.CHANNEL_LABELS[c] for c in channels], fontsize=7.6, color=S.INK_2)
    ax.set_xlim(0, len(hazards))
    ax.set_ylim(len(channels), 0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.grid(False)
    title_block(ax, 'C   Where the association sits, 2020-26', 'Adjusted residuals of the hazard x channel table (z scores under independence)', pad=24)
    return (vmax, norm)

def panel_c_key(ax_cb, ax_note, vmax, norm, summ):
    w = 0.34
    grad = np.linspace(-vmax, vmax, 256).reshape(1, -1)
    ax_cb.pcolormesh(np.linspace(0.0, w, 257), np.array([0.0, 1.0]), grad, cmap=DIVMAP, norm=norm)
    ax_cb.set_xlim(0, 1)
    ax_cb.set_ylim(-2.6, 2.6)
    ax_cb.set_xticks([])
    ax_cb.set_yticks([])
    for sp in ax_cb.spines.values():
        sp.set_visible(False)
    ax_cb.grid(False)
    for xv, lab in [(0.0, f'{-vmax:.0f}'), (w / 2, '0'), (w, f'+{vmax:.0f}')]:
        ax_cb.text(xv, -0.35, lab, fontsize=6.8, color=S.INK_3, ha='center', va='top')
    ax_cb.text(0.0, -1.95, 'fewer recalls than expected', fontsize=6.8, color=S.INK_3, ha='left', va='top')
    ax_cb.text(w, -1.95, 'more', fontsize=6.8, color=S.INK_3, ha='right', va='top')
    ax_note.axis('off')

def build(fig, comp, roll, resid, summ):
    gs = fig.add_gridspec(7, 1, height_ratios=[2.48, 0.42, 1.8, 1.0, 0.46, 0.1, 0.2])
    panel_a(fig.add_subplot(gs[0]), comp, summ)
    panel_a_legend(fig.add_subplot(gs[1]), comp)
    panel_b(fig.add_subplot(gs[2]), roll)
    vmax, norm = panel_c(fig.add_subplot(gs[3]), resid)
    panel_c_key(fig.add_subplot(gs[4]), fig.add_subplot(gs[5]), vmax, norm, summ)
    fig.add_subplot(gs[6]).axis('off')
    freeze_free_text(fig)
    S.source_note(fig, f"CPSC apparel and home-textile recalls, n = {summ['n_records_2000_2026']} for 2000-2026 (apparel_recalls_v2). Sales channel is unknown for\n{summ['n_channel_unknown_excluded']} records: they are kept in the panel B denominators and excluded from panels A and C.")

def main(quick=False):
    comp, roll, resid, summ = load()
    out = S.save_figure(lambda f: build(f, comp, roll, resid, summ), str(HERE / 'figure_hazard_channel_regime'), height_in=HEIGHT_IN)
    print('[10] wrote ' + ', '.join((Path(p).name for p in out)))
    return out
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    main(**vars(ap.parse_args()))
