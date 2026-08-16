from __future__ import annotations
import io
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
PAGE_W, PAGE_H = (8.5, 11.0)
MARGIN = 1.0
CONTENT_W = PAGE_W - 2 * MARGIN
MAX_CONTENT_H = PAGE_H - 2 * MARGIN
SURFACE = '#FFFFFF'
CAT = ['#2464C7', '#B8155A', '#C07400', '#6838BE', '#12854B', '#992525', '#0C93A3']
BLUE, PINK, AMBER, PURPLE, GREEN, RED, TEAL = CAT
INK = '#000000'
INK_2 = '#000000'
INK_3 = '#000000'
GRAY_1 = '#4A4845'
GRAY_2 = '#8A8782'
GRAY_3 = '#B4B1AB'
GRAY_4 = '#D6D4CE'
RULE = '#E4E2DD'
AXIS = '#C6C4BF'
GRAYS = [GRAY_1, GRAY_2, GRAY_3, GRAY_4]
HAZARD_COLORS = {'flammability_burn': GRAY_1, 'choking_small_parts': BLUE, 'drawstring_strangulation': AMBER, 'chemical': PURPLE, 'fall_slip': GREEN, 'laceration_puncture': RED, 'entrapment_entanglement': TEAL, 'protective_failure': GRAY_2, 'other': PINK}
HAZARD_LABELS = {'flammability_burn': 'Flammability / burn', 'choking_small_parts': 'Choking / small parts', 'drawstring_strangulation': 'Drawstring / strangulation', 'chemical': 'Chemical', 'fall_slip': 'Fall / slip', 'laceration_puncture': 'Laceration / puncture', 'entrapment_entanglement': 'Entrapment', 'protective_failure': 'Protective failure', 'other': 'Other'}
HAZARD_ORDER = ['flammability_burn', 'choking_small_parts', 'drawstring_strangulation', 'chemical', 'fall_slip', 'laceration_puncture', 'entrapment_entanglement', 'protective_failure', 'other']
CHANNEL_COLORS = {'online_only': PINK, 'mixed': AMBER, 'store_only': BLUE, 'unknown': GRAY_3}
CHANNEL_LABELS = {'online_only': 'Online only', 'mixed': 'Online + stores', 'store_only': 'Stores only', 'unknown': 'Channel unknown'}
COUNTRY_COLORS = {'China': PINK, 'Vietnam': TEAL, 'India': AMBER, 'Bangladesh': BLUE, 'Indonesia': PURPLE, 'Pakistan': GREEN, 'United States': RED, 'Other': GRAY_2}
REMEDY_COLORS = {'refund': BLUE, 'replace': AMBER, 'repair': GREEN, 'none': RED, 'other': GRAY_2}
SEQ_BLUE = ['#EAF1FB', '#C6D9F4', '#9DBDEA', '#6E9BDD', '#4479CE', '#2464C7', '#1B4C97', '#133468']
DIV = ['#8E0F44', '#B8155A', '#DE7CA4', '#E4E2DD', '#6FBE92', '#12854B', '#0B5F36']

def _pick_font() -> list[str]:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    prefer = ['Poppins', 'Inter', 'Source Sans 3', 'Source Sans Pro', 'IBM Plex Sans', 'Lato', 'Liberation Sans', 'DejaVu Sans']
    return [f for f in prefer if f in installed] or ['DejaVu Sans']
FONTS = _pick_font()
BASE_RC = {'figure.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'font.family': 'sans-serif', 'font.sans-serif': FONTS, 'font.size': 8.5, 'text.color': INK, 'axes.labelcolor': INK, 'axes.labelsize': 8.5, 'axes.titlesize': 10, 'axes.titleweight': 'semibold', 'axes.titlecolor': INK, 'axes.titlelocation': 'left', 'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'xtick.color': INK_3, 'ytick.color': INK_3, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False, 'legend.fontsize': 8, 'legend.labelcolor': INK, 'grid.color': RULE, 'grid.linewidth': 0.7, 'lines.linewidth': 2.0, 'lines.solid_capstyle': 'round', 'patch.linewidth': 0, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42}
plt.rcParams.update(BASE_RC)

def tidy(ax, *, grid='y', spines=('top', 'right'), zero_line=False):
    ax.spines[list(spines)].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=RULE, linewidth=0.7)
        ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=INK_3)
    if zero_line:
        ax.axhline(0, color=AXIS, linewidth=0.8)
    return ax

def panel_title(ax, title, subtitle=None, pad=16):
    m = re.match('\\s*([A-Za-z])\\b', str(title))
    ax._panel_letter = f"({(m.group(1).lower() if m else 'a')})"
    ax.set_title(' ', loc='left', pad=10, fontsize=9)

def _place_panel_letters(fig):
    axes = [ax for ax in fig.axes if getattr(ax, '_panel_letter', None)]
    if not axes:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    entries = []
    for ax in axes:
        tb = ax.get_tightbbox(renderer)
        x0, y1 = inv.transform((tb.x0, tb.y1))
        col = round(ax.get_position().x0, 1)
        entries.append([ax, x0, y1, col])
    col_x = {}
    for _, x0, _, col in entries:
        col_x[col] = min(col_x.get(col, 1.0), x0)
    for ax, _, y1, col in entries:
        t = fig.text(min(max(col_x[col], 0.0), 0.97), min(y1, 0.998), ax._panel_letter, ha='left', va='top', fontsize=10, fontweight='normal', color=INK)
        t.set_in_layout(False)

def spread_labels(values: dict, gap: float) -> dict:
    out, prev = ({}, float('-inf'))
    for k, v in sorted(values.items(), key=lambda kv: kv[1]):
        v = max(float(v), prev + gap)
        out[k], prev = (v, v)
    return out

def stacked_bars(ax, x, series, colors, *, width=0.72, gap=0.004, labels=None, min_label=0.06, label_fmt='{:.0%}'):
    import numpy as np
    bottom = np.zeros(len(x), dtype=float)
    for key, vals in series.items():
        vals = np.asarray(vals, dtype=float)
        ax.bar(x, vals - gap, bottom=bottom + gap / 2, width=width, color=colors[key], edgecolor=SURFACE, linewidth=0.8, label=(labels or {}).get(key, key), zorder=2)
        for xi, v, b in zip(x, vals, bottom):
            if v >= min_label:
                ax.text(xi, b + v / 2, label_fmt.format(v), ha='center', va='center', fontsize=7.5, color='white', fontweight='semibold', zorder=3)
        bottom = bottom + vals
    return bottom

def value_labels(ax, xs, ys, fmt='{:,.0f}', dy=4, color=None, fontsize=7.5, every=1):
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i % every:
            continue
        ax.annotate(fmt.format(y), (x, y), xytext=(0, dy), textcoords='offset points', ha='center', va='bottom', fontsize=fontsize, color=color or INK_2)

def legend_row(fig_or_ax, handles_labels, *, ncol=4, y=0.0, fontsize=8):
    handles, labels = handles_labels
    leg = fig_or_ax.legend(handles, labels, loc='lower center', ncol=ncol, frameon=False, fontsize=fontsize, bbox_to_anchor=(0.5, y), handlelength=1.1, handleheight=1.1, columnspacing=1.4, borderaxespad=0)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg

def freeze_free_text(fig):
    for ax in fig.axes:
        for t in list(ax.texts):
            t.set_in_layout(False)
        for child in list(getattr(ax, '_children', [])):
            if child.__class__.__name__ == 'Annotation':
                child.set_in_layout(False)
        if ax.get_legend() is not None:
            ax.get_legend().set_in_layout(False)
    for t in list(fig.texts):
        t.set_in_layout(False)

def source_note(fig, text, y=0.002):
    return None

def _letter_pdf_from(fig, out_pdf: Path, height_in: float):
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', facecolor=SURFACE)
    buf.seek(0)
    try:
        from pypdf import PdfReader, PdfWriter, Transformation
        from pypdf.generic import RectangleObject
    except ImportError:
        try:
            from PyPDF2 import PdfReader, PdfWriter, Transformation
            from PyPDF2.generic import RectangleObject
        except ImportError:
            fig.savefig(out_pdf, facecolor=SURFACE)
            return
    src = PdfReader(buf).pages[0]
    writer = PdfWriter()
    page = writer.add_blank_page(width=PAGE_W * 72, height=PAGE_H * 72)
    ty = (PAGE_H - MARGIN - height_in) * 72
    page.merge_transformed_page(src, Transformation().translate(MARGIN * 72, ty))
    page.mediabox = RectangleObject((0, 0, PAGE_W * 72, PAGE_H * 72))
    with open(out_pdf, 'wb') as fh:
        writer.write(fh)

def _force_black_text(fig):
    import matplotlib.colors as mcolors
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        try:
            r, g, b, _ = mcolors.to_rgba(t.get_color())
        except (ValueError, TypeError):
            continue
        if r > 0.92 and g > 0.92 and (b > 0.92):
            continue
        t.set_color(INK)

def save_figure(build, stem, height_in=8.6, *, width_in=CONTENT_W, dpi=300):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    if height_in > MAX_CONTENT_H:
        raise ValueError(f'height_in {height_in} exceeds printable {MAX_CONTENT_H}')
    fig = plt.figure(figsize=(width_in, height_in), facecolor=SURFACE)
    fig.set_layout_engine('constrained', w_pad=0.0, h_pad=0.0, wspace=0.0, hspace=0.0)
    build(fig)
    _force_black_text(fig)
    _place_panel_letters(fig)
    fig.savefig(f'{stem}.png', dpi=dpi, facecolor=SURFACE)
    fig.savefig(f'{stem}.svg', facecolor=SURFACE)
    _letter_pdf_from(fig, Path(f'{stem}.pdf'), height_in)
    plt.close(fig)
    return [f'{stem}.png', f'{stem}.svg', f'{stem}.pdf']
