from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RECALLS = HERE.parent / "01_dataset_construction" / "apparel_recalls_clean.csv"
OTEXA_SME = HERE.parent.parent / "01_data" / "otexa_m2_total_apparel_imports_1989_2025_sme.csv"
OTEXA_USD = HERE.parent.parent / "01_data" / "otexa_m2_total_apparel_imports_1989_2025_dollars.csv"

YEAR_MIN, YEAR_MAX = 1990, 2025
PERIOD_BINS = [1989, 1999, 2009, 2019, 2025]
PERIOD_LABELS = ["1990-99", "2000-09", "2010-19", "2020-25"]

COUNTRY_FIX = {
    "China (Mini Lunn, Beau Kid)": "China",
    "India (Fabric Flavours)": "India",
    "Korea": "South Korea",
}
COUNTRY_DROP = {"United States", "European Union (My Little Pie, Joha)"}

MIN_BN_SME = 0.5
FOCUS = ["China", "India", "Vietnam", "Bangladesh", "Indonesia", "Pakistan"]

COLORS = {
    "China": "#C05A38",
    "India": "#E0A11C",
    "Vietnam": "#0A9396",
    "Bangladesh": "#3E5296",
    "Indonesia": "#7A6A9B",
    "Pakistan": "#4C7A4A",
    "Other": "#8F8B82",
}
INK, MUTED, RULE = "#1A1A19", "#6E6A63", "#E7E5DE"

plt.rcParams.update({
    "font.family": "Poppins",
    "axes.edgecolor": "#C3C2B7",
    "axes.labelcolor": "#52514E",
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.8,
})


def load_recalls() -> pd.DataFrame:
    df = pd.read_csv(RECALLS)
    df = df[df.year.between(YEAR_MIN, YEAR_MAX) & df.countries.notna()].copy()
    df["country"] = df.countries.str.split(";")
    df = df.explode("country")
    df["country"] = df.country.str.strip().replace(COUNTRY_FIX)
    df = df[~df.country.isin(COUNTRY_DROP)]
    return df[["recall_id", "year", "country", "primary_country"]]


def load_imports() -> pd.DataFrame:
    sme = pd.read_csv(OTEXA_SME).rename(columns={"DATA_VALUE": "sme"})
    usd = pd.read_csv(OTEXA_USD).rename(columns={"DATA_VALUE": "usd"})
    keys = ["Country", "Year"]
    imp = sme[keys + ["sme"]].merge(usd[keys + ["usd"]], on=keys, how="outer")
    imp = imp.rename(columns={"Country": "country", "Year": "year"})
    imp = imp[imp.country.notna() & imp.year.between(YEAR_MIN, YEAR_MAX)]
    imp = imp[~imp.country.str.startswith("_")]
    return imp.groupby(["country", "year"], as_index=False)[["sme", "usd"]].sum()


def check_duplicate_years(imp: pd.DataFrame) -> list[int]:
    w = imp[imp.country == "World"].set_index("year").sme.sort_index()
    suspect = []
    for y in w.index[1:]:
        prev = w.get(y - 1)
        if prev and abs(w[y] - prev) / prev < 1e-3:
            suspect.append(int(y))
    return suspect


def panel(recalls: pd.DataFrame, imp: pd.DataFrame,
          drop_years: tuple[int, ...] = ()) -> pd.DataFrame:
    r = recalls[~recalls.year.isin(drop_years)]
    i = imp[(imp.country != "World") & (~imp.year.isin(drop_years))]
    counts = (r.groupby(["country", "year"], as_index=False)
                .recall_id.nunique().rename(columns={"recall_id": "recalls"}))
    p = i.merge(counts, on=["country", "year"], how="left")
    p["recalls"] = p.recalls.fillna(0).astype(int)
    p = p[p.sme > 0].copy()
    p["period"] = pd.cut(p.year, bins=PERIOD_BINS, labels=PERIOD_LABELS)
    return p.dropna(subset=["period"])


def quasipoisson(y: np.ndarray, X: np.ndarray, offset: np.ndarray,
                 tol: float = 1e-10, maxiter: int = 100):
    beta = np.zeros(X.shape[1])
    for _ in range(maxiter):
        eta = X @ beta + offset
        mu = np.exp(np.clip(eta, -30, 30))
        W = mu
        z = eta - offset + (y - mu) / mu
        XtW = X.T * W
        beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    mu = np.exp(np.clip(X @ beta + offset, -30, 30))
    df_resid = len(y) - X.shape[1]
    pearson = float(np.sum((y - mu) ** 2 / mu) / df_resid)
    cov = np.linalg.inv((X.T * mu) @ X) * pearson
    return beta, np.sqrt(np.diag(cov)), pearson, df_resid


def fit_irr(p: pd.DataFrame, focus: list[str], ref: str = "China",
            min_bn_sme: float = MIN_BN_SME):
    vol = p.groupby("country").sme.sum() / 1e9
    keep = set(vol[vol >= min_bn_sme].index) | set(focus)
    d = p[p.country.isin(keep)].copy()
    d["ctry"] = d.country.where(d.country.isin(focus), "Other")
    levels = [ref] + [c for c in focus if c != ref] + ["Other"]
    cols, names = [np.ones(len(d))], ["intercept"]
    for c in levels[1:]:
        cols.append((d.ctry == c).astype(float).values)
        names.append(f"country: {c}")
    present = [q for q in PERIOD_LABELS if (d.period == q).any()]
    for per in present[1:]:
        cols.append((d.period == per).astype(float).values)
        names.append(f"period: {per}")
    X = np.column_stack(cols)
    beta, se, disp, dfr = quasipoisson(d.recalls.values.astype(float), X,
                                       np.log(d.sme.values.astype(float)))
    t = beta / se
    pv = 2 * stats.t.sf(np.abs(t), dfr)
    out = pd.DataFrame({
        "term": names, "irr": np.exp(beta),
        "lo": np.exp(beta - 1.96 * se), "hi": np.exp(beta + 1.96 * se), "p": pv,
    })
    return out, disp, dfr, len(d)


def save_figure(draw, stem, width=6.5, height=4.2):
    fig, ax = plt.subplots(figsize=(width, height))
    draw(ax)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300)
    fig.savefig(f"{stem}.svg")
    plt.close(fig)
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([1.7 / 8.5, (11 - 1 - height) / 11,
                       (6.5 - 1.0) / 8.5, (height - 0.7) / 11])
    draw(ax)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)


def spread(values: dict, gap: float) -> dict:
    items = sorted(values.items(), key=lambda kv: kv[1])
    out = {}
    prev = -np.inf
    for k, v in items:
        v = max(v, prev + gap)
        out[k] = v
        prev = v
    return out


def figure_rate_by_period(rate_wide: pd.DataFrame):
    ends = spread({c: float(rate_wide[c].values[-1]) for c in rate_wide.columns},
                  gap=float(rate_wide.max().max()) * 0.055)

    def draw(ax):
        for c in rate_wide.columns:
            ax.plot(range(len(rate_wide)), rate_wide[c].values, marker="o",
                    markersize=4.5, linewidth=2.6 if c == "China" else 1.8,
                    color=COLORS.get(c, "#8F8B82"), label=c,
                    solid_capstyle="round", markerfacecolor="white",
                    markeredgecolor=COLORS.get(c, "#8F8B82"), markeredgewidth=1.6)
            ax.annotate(c, (len(rate_wide) - 1, ends[c]),
                        xytext=(7, 0), textcoords="offset points",
                        fontsize=8.5, color=COLORS.get(c, "#8F8B82"),
                        va="center", fontweight="medium")
        ax.set_xticks(range(len(rate_wide)), rate_wide.index, fontsize=8)
        ax.set_ylabel("Recalls per billion SME imported")
        ax.set_xlabel("Period")
        ax.set_xlim(-0.3, len(rate_wide) - 0.05)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", color=RULE, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    save_figure(draw, HERE / "figure_recall_rate")


def figure_share_vs_share(sh: pd.DataFrame):
    d = sh[(sh.rec_share > 0) & (sh.imp_share > 0)]

    def draw(ax):
        lo = min(d.imp_share.min(), d.rec_share.min()) / 2.2
        hi = max(d.imp_share.max(), d.rec_share.max()) * 2.2
        ax.plot([lo, hi], [lo, hi], color="#C3C2B7", linewidth=0.9,
                linestyle=(0, (4, 3)), zorder=1)
        ax.annotate("parity: recall share = import share", (hi * 0.28, hi * 0.28),
                    xytext=(3, -12), textcoords="offset points", fontsize=7.5,
                    color=MUTED, rotation=45, rotation_mode="anchor", ha="center")
        pts = np.column_stack([np.log10(d.imp_share.values),
                               np.log10(d.rec_share.values)])
        dirs = [(1, .4), (-1, -.4), (1, -.4), (-1, .4), (0, 1), (0, -1)]
        taken: list[np.ndarray] = []
        for i, (c, row) in enumerate(d.iterrows()):
            col = COLORS.get(c, "#8F8B82")
            ax.scatter(row.imp_share, row.rec_share, s=36, color=col, zorder=3,
                       edgecolor="white", linewidth=1.1)
            others = np.delete(pts, i, axis=0)
            best, best_score = dirs[0], -np.inf
            for dx, dy in dirs:
                probe = pts[i] + np.array([dx, dy]) * 0.13
                near = np.min(np.linalg.norm(others - probe, axis=1))
                if taken:
                    near = min(near, np.min(np.linalg.norm(
                        np.array(taken) - probe, axis=1)) * 1.2)
                if near > best_score:
                    best, best_score = (dx, dy), near
            taken.append(pts[i] + np.array(best) * 0.13)
            dx, dy = best
            ax.annotate(c, (row.imp_share, row.rec_share),
                        xytext=(8 * dx if dx else 0, 9 * dy),
                        textcoords="offset points", fontsize=8, color=col,
                        ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                        va="bottom" if dy > 0 else "top", fontweight="medium")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        for a in (ax.xaxis, ax.yaxis):
            a.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{v:g}"))
        ax.set_xlabel("Share of US apparel imports, SME (%, log scale)")
        ax.set_ylabel("Share of textile recalls (%, log scale)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color=RULE, linewidth=0.7, which="major")
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    save_figure(draw, HERE / "figure_share_vs_share", height=4.6)


def md_table(df: pd.DataFrame, fmts: dict) -> list[str]:
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(fmts.get(c, str)(r[c]) for c in cols) + " |")
    return rows


def main():
    recalls, imp = load_recalls(), load_imports()
    suspect = check_duplicate_years(imp)

    p = panel(recalls, imp)
    BILLION = 1e9

    def rate_table(pp: pd.DataFrame) -> pd.DataFrame:
        g = pp.groupby("country").agg(recalls=("recalls", "sum"),
                                      sme=("sme", "sum")).reset_index()
        g["bn_sme"] = g.sme / BILLION
        g["rate"] = g.recalls / g.bn_sme
        lo = stats.chi2.ppf(0.025, 2 * g.recalls) / 2
        hi = stats.chi2.ppf(0.975, 2 * (g.recalls + 1)) / 2
        g["rate_lo"], g["rate_hi"] = lo / g.bn_sme, hi / g.bn_sme
        return g.sort_values("recalls", ascending=False)

    all_years = rate_table(p)
    recent = rate_table(p[p.year >= 2010])
    top = all_years[all_years.recalls >= 5].head(15)

    pf = p.copy()
    pf["ctry"] = pf.country.where(pf.country.isin(FOCUS), "Other")
    per = pf.groupby(["period", "ctry"], observed=True).agg(
        recalls=("recalls", "sum"), sme=("sme", "sum")).reset_index()
    per["rate"] = per.recalls / (per.sme / BILLION)
    rate_wide = per.pivot(index="period", columns="ctry", values="rate")
    rate_wide = rate_wide[[c for c in FOCUS + ["Other"] if c in rate_wide.columns]]
    cnt_wide = per.pivot(index="period", columns="ctry", values="recalls")[rate_wide.columns]

    irr, disp, dfr, nobs = fit_irr(p, FOCUS)
    irr_recent, disp_r, _, nobs_r = fit_irr(p[p.year >= 2010], FOCUS)

    sens = None
    if suspect:
        ps = panel(recalls, imp, drop_years=tuple(suspect))
        sens = rate_table(ps[ps.year >= 2010])

    rr = p[p.year >= 2010]
    sh = rr.groupby("country").agg(recalls=("recalls", "sum"),
                                   sme=("sme", "sum"))
    sh["rec_share"] = sh.recalls / sh.recalls.sum() * 100
    sh["imp_share"] = sh.sme / sh.sme.sum() * 100
    sh = sh[(sh.rec_share >= 1.5) | (sh.imp_share >= 3)]
    figure_share_vs_share(sh.sort_values("imp_share", ascending=False))
    figure_rate_by_period(rate_wide)

    f2, f1, f0 = (lambda v: f"{v:,.2f}"), (lambda v: f"{v:,.1f}"), (lambda v: f"{v:,.0f}")
    L = ["# Recall rate per unit of import volume (OTEXA-normalized)", "",
         "Numerator: CPSC apparel/home-textile recalls, 1990-2025, counted once per",
         "country of manufacture. Denominator: OTEXA category 1 (M2 total apparel",
         "imports) square metre equivalents (SME) for the same country and year.",
         "US-origin recalls are excluded (no import denominator).", ""]

    if suspect:
        L += [f"> **Data quality flag**: OTEXA year(s) {suspect} have world totals",
              "> within 0.1% of the previous year in both SME and dollars, which points",
              "> to a duplicated column in the export rather than real trade. Re-pull",
              "> those years before submission. A sensitivity analysis excluding them",
              "> is reported at the end of this file.", ""]

    L += ["## 1. Recall rate by country of manufacture, 1990-2025", "",
          "Rate = recalls per billion SME imported. CI is an exact Poisson interval",
          "on the recall count.", ""]
    t1 = top[["country", "recalls", "bn_sme", "rate", "rate_lo", "rate_hi"]].copy()
    t1.columns = ["Country", "Recalls", "Imports (bn SME)", "Recalls per bn SME",
                  "95% CI low", "95% CI high"]
    L += md_table(t1, {"Recalls": f0, "Imports (bn SME)": f1,
                       "Recalls per bn SME": f2, "95% CI low": f2, "95% CI high": f2})

    L += ["", "## 2. Recall rate, 2010-2025 only", ""]
    t2 = recent[recent.recalls >= 4].head(15)[
        ["country", "recalls", "bn_sme", "rate"]].copy()
    t2.columns = ["Country", "Recalls", "Imports (bn SME)", "Recalls per bn SME"]
    L += md_table(t2, {"Recalls": f0, "Imports (bn SME)": f1,
                       "Recalls per bn SME": f2})

    L += ["", "## 3. Recall rate by period (recalls per billion SME)", ""]
    rw = rate_wide.reset_index().rename(columns={"period": "Period"})
    L += md_table(rw, {c: f2 for c in rate_wide.columns})
    L += ["", "Underlying recall counts:", ""]
    cw = cnt_wide.reset_index().rename(columns={"period": "Period"})
    L += md_table(cw, {c: f0 for c in cnt_wide.columns})

    def irr_block(tab, dispersion, n, title):
        b = ["", f"## {title}", "",
             "Quasi-Poisson log-linear model, recalls ~ country + period, offset =",
             f"log(SME imported). Reference country China; reference period {PERIOD_LABELS[0]}.",
             f"n = {n} country-year cells; Pearson dispersion = {dispersion:.2f}.", ""]
        d = tab[tab.term != "intercept"].copy()
        d["ci"] = d.apply(lambda r: f"{r.lo:.2f}-{r.hi:.2f}", axis=1)
        d["pp"] = d.p.apply(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")
        d = d[["term", "irr", "ci", "pp"]]
        d.columns = ["Term", "IRR", "95% CI", "p"]
        return b + md_table(d, {"IRR": f2})

    L += irr_block(irr, disp, nobs, "4. Incidence rate ratios, 1990-2025")
    L += irr_block(irr_recent, disp_r, nobs_r, "5. Incidence rate ratios, 2010-2025")

    L += ["", "## 6. Recall share vs import share, 2010-2025", "",
          "Countries above parity are over-represented in recalls relative to the",
          "volume they ship.", ""]
    t6 = sh.reset_index()[["country", "rec_share", "imp_share"]].copy()
    t6["ratio"] = t6.rec_share / t6.imp_share
    t6.columns = ["Country", "Share of recalls (%)", "Share of imports (%)",
                  "Ratio (recall share / import share)"]
    L += md_table(t6.sort_values("Share of recalls (%)", ascending=False),
                  {c: f2 for c in t6.columns if c != "Country"})

    if sens is not None:
        L += ["", f"## 7. Sensitivity: 2010-2025 excluding suspect year(s) {suspect}", ""]
        t7 = sens[sens.recalls >= 4].head(10)[["country", "recalls", "bn_sme", "rate"]].copy()
        t7.columns = ["Country", "Recalls", "Imports (bn SME)", "Recalls per bn SME"]
        L += md_table(t7, {"Recalls": f0, "Imports (bn SME)": f1,
                           "Recalls per bn SME": f2})

    (HERE / "table_recall_rate.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
