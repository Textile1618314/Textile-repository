from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import data as D
from _common import mcmc as M

RES = D.results_dir(__file__)

YEAR_MIN, YEAR_MAX = 2000, 2026
PERIOD_BINS = [1973, 1989, 1999, 2009, 2019, 2026]
PERIOD_LABELS = ["1974-89", "1990-99", "2000-09", "2010-19", "2020-26"]

CLASSES = ["violation_only", "both", "injury_only", "neither"]
CLASS_LABELS = {"violation_only": "Violation, no injury reported",
                "both": "Violation and injury",
                "injury_only": "Injury reported, no violation",
                "neither": "Neither recorded"}

FULL = dict(n_draws=6000, n_tune=6000, n_chains=4)
QUICK = dict(n_draws=500, n_tune=500, n_chains=4)
N_SIGMA_GRID_FULL, N_SIGMA_GRID_QUICK = 401, 121
HDI_PROB = 0.94


def build_frame() -> pd.DataFrame:
    df = D.load_recalls()
    d = df.copy()
    d["period"] = pd.cut(d.year, bins=PERIOD_BINS, labels=PERIOD_LABELS)
    d["is_violation"] = d.is_violation.astype(bool)
    d["injuries_reported"] = d.injuries_reported.astype(bool)
    d["klass"] = np.select(
        [d.is_violation & d.injuries_reported,
         d.is_violation & ~d.injuries_reported,
         ~d.is_violation & d.injuries_reported],
        ["both", "violation_only", "injury_only"], default="neither")
    return d


def classification_tables(d: pd.DataFrame):
    ann = (d[d.year.between(YEAR_MIN, YEAR_MAX)]
           .groupby(["year", "klass"], observed=True).size()
           .unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0))
    ann = ann.reindex(range(YEAR_MIN, YEAR_MAX + 1), fill_value=0)
    ann["n"] = ann[CLASSES].sum(axis=1)
    for c in CLASSES:
        ann[f"share_{c}"] = np.where(ann.n > 0, ann[c] / ann.n.replace(0, 1), np.nan)
    ann["share_violation"] = np.where(ann.n > 0,
                                      (ann.violation_only + ann.both) / ann.n.replace(0, 1),
                                      np.nan)
    ann["share_injury"] = np.where(ann.n > 0,
                                   (ann.injury_only + ann.both) / ann.n.replace(0, 1),
                                   np.nan)
    ann = ann.reset_index(names="year")

    per = (d.groupby(["period", "klass"], observed=True).size()
           .unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0))
    per["n"] = per[CLASSES].sum(axis=1)
    for c in CLASSES:
        per[f"share_{c}"] = per[c] / per.n
    per["share_violation"] = (per.violation_only + per.both) / per.n
    per["share_injury"] = (per.injury_only + per.both) / per.n
    return ann, per.reset_index()


LOG_SIGMA_LO, LOG_SIGMA_HI = np.log(0.02), np.log(4.0)
A0_PRIOR_SD = 1.5
SIGMA_PRIOR_SD = 0.5


def _conditional_mode(k, n, sigma, DtD, iters=100, tol=1e-11):
    T = len(k)
    a = np.log((k + 0.5) / (n - k + 0.5))
    P = DtD / sigma ** 2
    P[0, 0] += 1.0 / A0_PRIOR_SD ** 2
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(a, -30, 30)))
        W = n * p * (1 - p)
        H = np.diag(W) + P
        step = np.linalg.solve(H, (k - n * p) - P @ a)
        a = a + step
        if np.max(np.abs(step)) < tol:
            break
    p = 1.0 / (1.0 + np.exp(-np.clip(a, -30, 30)))
    H = np.diag(n * p * (1 - p)) + P
    return a, H, P


def _loglik(a, k, n):
    eta = np.clip(a, -30, 30)
    return np.sum(k * eta - n * np.logaddexp(0.0, eta), axis=-1)


def _log_prior_states(a, sigma):
    T = a.shape[-1]
    da = np.diff(a, axis=-1)
    return (-0.5 * (a[..., 0] / A0_PRIOR_SD) ** 2 - np.log(A0_PRIOR_SD)
            - (T - 1) * np.log(sigma)
            - np.sum(da ** 2, axis=-1) / (2 * sigma ** 2)
            - 0.5 * T * np.log(2 * np.pi))


def build_sigma_grid(k, n, n_grid: int):
    T = len(k)
    Dm = np.diff(np.eye(T), axis=0)
    DtD = Dm.T @ Dm
    grid = np.exp(np.linspace(LOG_SIGMA_LO, LOG_SIGMA_HI, n_grid))
    logml = np.empty(n_grid)
    a_hat = np.empty((n_grid, T))
    cholH = np.empty((n_grid, T, T))
    logdetL = np.empty(n_grid)
    for g, s in enumerate(grid):
        a, H, _ = _conditional_mode(k, n, s, DtD.copy())
        L = np.linalg.cholesky(H)
        a_hat[g] = a
        cholH[g] = L
        logdetL[g] = float(np.sum(np.log(np.diag(L))))
        logml[g] = float(_loglik(a, k, n) + _log_prior_states(a, s)
                         + M.log_halfnormal(s, SIGMA_PRIOR_SD)
                         - logdetL[g] + 0.5 * T * np.log(2 * np.pi))
    return {"grid": grid, "log_grid": np.log(grid), "logml": logml,
            "a_hat": a_hat, "cholH": cholH, "logdetL": logdetL, "T": T}


def _interp_logml(G, xi):
    return float(np.interp(xi, G["log_grid"], G["logml"],
                           left=-np.inf, right=-np.inf))


def draw_states(G, xi_draws, k, n, rng):
    T = G["T"]
    S = len(xi_draws)
    idx = np.clip(np.searchsorted(G["log_grid"], xi_draws), 1, len(G["grid"]) - 1)
    lo = G["log_grid"][idx - 1]
    hi = G["log_grid"][idx]
    idx = np.where(xi_draws - lo < hi - xi_draws, idx - 1, idx)
    A = np.empty((S, T))
    logq = np.empty(S)
    for g in np.unique(idx):
        sel = np.flatnonzero(idx == g)
        z = rng.standard_normal((len(sel), T))
        A[sel] = G["a_hat"][g] + np.linalg.solve(G["cholH"][g].T, z.T).T
        logq[sel] = (-0.5 * np.sum(z ** 2, axis=1) + G["logdetL"][g]
                     - 0.5 * T * np.log(2 * np.pi))
    sigma = np.exp(xi_draws)
    log_target = (_loglik(A, k, n) + _log_prior_states(A, sigma)
                  + np.array([M.log_halfnormal(s, SIGMA_PRIOR_SD) for s in sigma]))
    log_prop = np.array([_interp_logml(G, x) for x in xi_draws]) + logq
    return A, log_target - log_prop


def weighted_hdi(x, w, prob=HDI_PROB):
    x = np.asarray(x, float)
    o = np.argsort(x)
    xs, ws = x[o], np.asarray(w, float)[o]
    c = np.cumsum(ws) / ws.sum()
    ok = c >= prob
    if not ok.any():
        return float(xs[0]), float(xs[-1])
    lo_i = np.clip(np.searchsorted(c, c - prob, side="left"), 0, len(xs) - 1)
    widths = np.where(ok, xs - xs[lo_i], np.inf)
    j = int(np.argmin(widths))
    return float(xs[lo_i[j]]), float(xs[j])


def weighted_mean(x, w):
    return float(np.sum(w * x) / np.sum(w))


def weighted_sd(x, w):
    m = weighted_mean(x, w)
    return float(np.sqrt(np.sum(w * (x - m) ** 2) / np.sum(w)))


def fit_bayes(ann: pd.DataFrame, cfg: dict, n_grid=401, seed=11) -> dict:
    years = ann.year.to_numpy()
    T = len(years)
    n = ann.n.to_numpy(float)
    kv = (ann.violation_only + ann.both).to_numpy(float)
    ki = (ann.injury_only + ann.both).to_numpy(float)

    Gv = build_sigma_grid(kv, n, n_grid)
    Gi = build_sigma_grid(ki, n, n_grid)

    def log_post(xi):
        if not np.all(np.isfinite(xi)):
            return -np.inf
        if not (LOG_SIGMA_LO < xi[0] < LOG_SIGMA_HI
                and LOG_SIGMA_LO < xi[1] < LOG_SIGMA_HI):
            return -np.inf
        return (_interp_logml(Gv, xi[0]) + xi[0]
                + _interp_logml(Gi, xi[1]) + xi[1])

    init = np.array([Gv["log_grid"][int(np.argmax(Gv["logml"]))],
                     Gi["log_grid"][int(np.argmax(Gi["logml"]))]])
    post = M.sample(log_post, init, names=["log_sigma_violation", "log_sigma_injury"],
                    seed=seed, verbose=True, jitter=0.15, **cfg)

    rng = np.random.default_rng(seed + 5)
    xi = post.flat()
    Av, lwv = draw_states(Gv, xi[:, 0], kv, n, rng)
    Ai, lwi = draw_states(Gi, xi[:, 1], ki, n, rng)
    lw = lwv + lwi
    lw -= lw.max()
    w = np.exp(lw)
    w_ess = float(w.sum() ** 2 / np.sum(w ** 2))

    pv = 1.0 / (1.0 + np.exp(-np.clip(Av, -30, 30)))
    pi = 1.0 / (1.0 + np.exp(-np.clip(Ai, -30, 30)))

    def grid_post(G):
        lp = G["logml"] + G["log_grid"]
        pr = np.exp(lp - lp.max())
        pr /= np.trapezoid(pr, G["log_grid"])
        mean = float(np.trapezoid(pr * G["grid"], G["log_grid"]))
        cdf = np.concatenate([[0.0], np.cumsum(
            0.5 * (pr[1:] + pr[:-1]) * np.diff(G["log_grid"]))])
        cdf /= cdf[-1]
        q = lambda a: float(np.exp(np.interp(a, cdf, G["log_grid"])))
        return {"mean": mean, "q03": q(0.03), "q50": q(0.5), "q97": q(0.97)}

    return {"post": post, "years": years, "p_violation": pv, "p_injury": pi,
            "weights": w, "weight_ess": w_ess,
            "k_violation": kv, "k_injury": ki, "n": n,
            "sigma_violation_draws": np.exp(xi[:, 0]),
            "sigma_injury_draws": np.exp(xi[:, 1]),
            "grid_check": {"violation": grid_post(Gv), "injury": grid_post(Gi)},
            "n_grid": n_grid}


def hdi(x: np.ndarray, prob=HDI_PROB) -> tuple[float, float]:
    x = np.sort(np.asarray(x))
    nn = len(x)
    k = max(int(np.floor(prob * nn)), 1)
    if k >= nn:
        return float(x[0]), float(x[-1])
    w = x[k:] - x[:nn - k]
    i = int(np.argmin(w))
    return float(x[i]), float(x[i + k])


def summarise_bayes(fit: dict) -> tuple[pd.DataFrame, dict]:
    years, pv, pi = fit["years"], fit["p_violation"], fit["p_injury"]
    w = fit["weights"]
    rows = []
    for j, y in enumerate(years):
        for name, arr, k in (("violation", pv, fit["k_violation"]),
                             ("injury", pi, fit["k_injury"])):
            lo, hi = weighted_hdi(arr[:, j], w)
            rows.append({"year": int(y), "series": name,
                         "posterior_mean": weighted_mean(arr[:, j], w),
                         "posterior_sd": weighted_sd(arr[:, j], w),
                         "hdi_lo": lo, "hdi_hi": hi,
                         "observed": float(k[j] / fit["n"][j]) if fit["n"][j] else np.nan,
                         "n": int(fit["n"][j])})
    by_year = pd.DataFrame(rows)

    early = np.isin(years, np.arange(2000, 2010))
    late = np.isin(years, np.arange(2020, 2026))
    late26 = np.isin(years, np.arange(2020, 2027))

    def block(arr, mask):
        return arr[:, mask].mean(axis=1)

    dv = block(pv, late) - block(pv, early)
    dv26 = block(pv, late26) - block(pv, early)
    di = block(pi, late) - block(pi, early)

    corr = np.array([np.corrcoef(pv[s], pi[s])[0, 1] for s in range(pv.shape[0])])
    xc = (years - years.mean()) / years.std()
    def slope(arr):
        lg = np.log(np.clip(arr, 1e-6, 1 - 1e-6) / (1 - np.clip(arr, 1e-6, 1 - 1e-6)))
        return (lg * xc).sum(axis=1) / (xc ** 2).sum()
    sv, si = slope(pv), slope(pi)
    gap_early = block(pv, early) - block(pi, early)
    gap_late = block(pv, late) - block(pi, late)
    d_gap = gap_late - gap_early

    mv = np.array([weighted_mean(pv[:, j], w) for j in range(len(years))])
    mi = np.array([weighted_mean(pi[:, j], w) for j in range(len(years))])
    up = np.flatnonzero(np.diff(np.sign(mv - mi)) > 0)
    crossing_year = int(years[up[0] + 1]) if len(up) else None
    p_violation_exceeds_injury_by_year = {
        int(y): float(np.sum(w * (pv[:, j] > pi[:, j])) / w.sum())
        for j, y in enumerate(years)}

    post = fit["post"]
    rhat, ess = post.rhat(), post.ess()

    def summ(x, label):
        lo, hi = weighted_hdi(x, w)
        return {"mean": weighted_mean(x, w), "sd": weighted_sd(x, w),
                "hdi_lo": lo, "hdi_hi": hi,
                "p_gt_0": float(np.sum(w * (x > 0)) / w.sum()),
                "label": label}

    out = {
        "hdi_prob": HDI_PROB,
        "crossing_year": crossing_year,
        "posterior_prob_violation_exceeds_injury_by_year":
            p_violation_exceeds_injury_by_year,
        "n_chains": int(post.n_chains), "n_draws_per_chain": int(post.n_draws),
        "seconds": float(post.seconds),
        "accept_rate": [float(a) for a in post.accept_rate],
        "sampled_parameters": list(post.names),
        "rhat": {nm: float(r) for nm, r in zip(post.names, rhat)},
        "ess": {nm: float(e) for nm, e in zip(post.names, ess)},
        "rhat_max": float(np.nanmax(rhat)), "ess_min": float(np.nanmin(ess)),
        "converged_rhat_lt_1.01_and_ess_gt_400": bool(post.converged()),
        "importance_weight_ess": float(fit["weight_ess"]),
        "importance_weight_ess_fraction":
            float(fit["weight_ess"] / len(fit["weights"])),
        "sigma_posterior": {
            "violation": {"mean": weighted_mean(fit["sigma_violation_draws"], w),
                          "hdi": list(weighted_hdi(fit["sigma_violation_draws"], w))},
            "injury": {"mean": weighted_mean(fit["sigma_injury_draws"], w),
                       "hdi": list(weighted_hdi(fit["sigma_injury_draws"], w))}},
        "grid_integration_crosscheck": fit["grid_check"],
        "n_sigma_grid": int(fit["n_grid"]),
        "violation_share_2020_25_minus_2000_09":
            summ(dv, "posterior difference in mean P(violation)"),
        "violation_share_2020_26_minus_2000_09": summ(dv26, "same, 2026 included"),
        "injury_share_2020_25_minus_2000_09":
            summ(di, "posterior difference in mean P(injury)"),
        "posterior_prob_violation_2020_25_exceeds_2000_09":
            float(np.sum(w * (dv > 0)) / w.sum()),
        "decoupling": {
            "correlation_of_latent_series": summ(corr, "Pearson r across years"),
            "logit_trend_violation_per_sd_year": summ(sv, "logit slope"),
            "logit_trend_injury_per_sd_year": summ(si, "logit slope"),
            "trend_difference": summ(sv - si, "violation slope minus injury slope"),
            "gap_2000_09": summ(gap_early, "P(violation) - P(injury), 2000-09"),
            "gap_2020_25": summ(gap_late, "P(violation) - P(injury), 2020-25"),
            "gap_change": summ(d_gap, "change in the gap"),
        },
        "period_means": {
            "violation_2000_09": summ(block(pv, early), "posterior mean"),
            "violation_2020_25": summ(block(pv, late), "posterior mean"),
            "injury_2000_09": summ(block(pi, early), "posterior mean"),
            "injury_2020_25": summ(block(pi, late), "posterior mean"),
        },
    }
    return by_year, out


def g_and_perm(tab: np.ndarray, rng, n_perm=5000) -> dict:
    tab = np.asarray(tab, float)
    n = tab.sum()
    if n == 0:
        return {"G": None, "df": None, "p_G": None}
    exp = np.outer(tab.sum(1), tab.sum(0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        g = float(2 * np.where(tab > 0, tab * np.log(np.where(exp > 0, tab / exp, 1)), 0).sum())
        x2 = float(np.where(exp > 0, (tab - exp) ** 2 / exp, 0).sum())
    df = (int((tab.sum(1) > 0).sum()) - 1) * (int((tab.sum(0) > 0).sum()) - 1)
    out = {"G": g, "chi2": x2, "df": df,
           "p_G": float(stats.chi2.sf(g, df)) if df > 0 else None,
           "p_chi2": float(stats.chi2.sf(x2, df)) if df > 0 else None,
           "n": int(n), "cells_expected_lt5": int((exp < 5).sum())}
    if tab.shape == (2, 2):
        odds, p = stats.fisher_exact(tab.astype(int))
        out["fisher_odds_ratio"] = float(odds)
        out["fisher_p"] = float(p)
        a, b, c, e = tab.ravel()
        se = np.sqrt(1 / (a + .5) + 1 / (b + .5) + 1 / (c + .5) + 1 / (e + .5))
        orh = (a + .5) * (e + .5) / ((b + .5) * (c + .5))
        out["odds_ratio_haldane"] = float(orh)
        out["or_ci_lo"] = float(np.exp(np.log(orh) - 1.959964 * se))
        out["or_ci_hi"] = float(np.exp(np.log(orh) + 1.959964 * se))
    return out


def crosstabs(d: pd.DataFrame, rng) -> dict:
    w = d[d.year.between(YEAR_MIN, YEAR_MAX)]
    out = {"window": f"{YEAR_MIN}-{YEAR_MAX}", "n": int(len(w))}

    vi = pd.crosstab(w.is_violation, w.injuries_reported).reindex(
        index=[False, True], columns=[False, True], fill_value=0).to_numpy()
    out["violation_x_injury_overall"] = {
        "table_rows_violation_cols_injury": vi.tolist(), **g_and_perm(vi, rng)}

    for key, col in (("by_hazard", "hazard_category"), ("by_channel", "sales_channel")):
        strata = {}
        for lvl, s in w.groupby(col, observed=True):
            t = pd.crosstab(s.is_violation, s.injuries_reported).reindex(
                index=[False, True], columns=[False, True], fill_value=0).to_numpy()
            strata[str(lvl)] = {
                "n": int(len(s)),
                "violation_share": float(s.is_violation.mean()),
                "injury_share": float(s.injuries_reported.mean()),
                "table": t.tolist(),
                **g_and_perm(t, rng)}
        tab = pd.crosstab(w[col], w.is_violation).reindex(
            columns=[False, True], fill_value=0).to_numpy()
        tab_i = pd.crosstab(w[col], w.injuries_reported).reindex(
            columns=[False, True], fill_value=0).to_numpy()
        out[key] = {"strata": strata,
                    "violation_share_homogeneity": g_and_perm(tab, rng),
                    "injury_share_homogeneity": g_and_perm(tab_i, rng)}
    return out


def violation_by_hazard(d: pd.DataFrame) -> pd.DataFrame:
    w = d[d.year.between(YEAR_MIN, YEAR_MAX)]
    rows = []
    for lvl, s in w.groupby("hazard_category", observed=True):
        n = len(s)
        for series, flag in (("violation", s.is_violation), ("injury", s.injuries_reported)):
            k = int(flag.sum())
            a, b = k + 0.5, n - k + 0.5
            lo, hi = stats.beta.ppf([(1 - HDI_PROB) / 2, 1 - (1 - HDI_PROB) / 2], a, b)
            rows.append({"hazard": str(lvl), "series": series, "n": n, "k": k,
                         "share": k / n, "posterior_mean": float(a / (a + b)),
                         "ci_lo": float(lo), "ci_hi": float(hi)})
    return pd.DataFrame(rows).sort_values(["series", "share"], ascending=[True, False])


def units_weighted(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for per, s in d.groupby("period", observed=True):
        u = pd.to_numeric(s.units, errors="coerce")
        k = u.notna()
        tot = float(u[k].sum())
        rows.append({
            "period": str(per), "n_recalls": int(len(s)),
            "n_with_units": int(k.sum()),
            "units_total": tot,
            "units_median": float(u[k].median()) if k.any() else np.nan,
            "share_violation_unweighted": float(s.is_violation.mean()),
            "share_violation_units_weighted":
                float(u[k & s.is_violation].sum() / tot) if tot else np.nan,
            "share_injury_unweighted": float(s.injuries_reported.mean()),
            "share_injury_units_weighted":
                float(u[k & s.injuries_reported].sum() / tot) if tot else np.nan,
            "median_units_violation":
                float(u[k & s.is_violation].median()) if (k & s.is_violation).any() else np.nan,
            "median_units_non_violation":
                float(u[k & ~s.is_violation].median()) if (k & ~s.is_violation).any() else np.nan,
        })
    return pd.DataFrame(rows)


def main(quick=False):
    cfg = QUICK if quick else FULL
    rng = np.random.default_rng(4)
    d = build_frame()
    ann, per = classification_tables(d)
    ann.to_csv(RES / "annual_classification.csv", index=False)
    per.to_csv(RES / "period_classification.csv", index=False)

    n_grid = N_SIGMA_GRID_QUICK if quick else N_SIGMA_GRID_FULL
    print(f"[11] nested Laplace: {len(ann)} latent states per series integrated "
          f"out on a {n_grid}-point scale grid; M.sample runs "
          f"{cfg['n_chains']} chains x {cfg['n_draws']} draws (tune "
          f"{cfg['n_tune']}) over the 2 walk-scale hyperparameters")
    fit = fit_bayes(ann, cfg, n_grid=n_grid)
    by_year, bayes = summarise_bayes(fit)
    by_year.to_csv(RES / "posterior_by_year.csv", index=False)
    D.write_json(bayes, RES / "posterior_summary.json")

    xt = crosstabs(d, rng)
    D.write_json(xt, RES / "crosstabs.json")

    vbh = violation_by_hazard(d)
    vbh.to_csv(RES / "violation_by_hazard.csv", index=False)

    uw = units_weighted(d)
    uw.to_csv(RES / "units_weighted.csv", index=False)

    w = d[d.year.between(YEAR_MIN, YEAR_MAX)]
    summary = {
        "analysis": "11_violation_vs_incident",
        "source_version": d.attrs.get("source_version", "v2_hardened"),
        "n_total": int(len(d)), "n_window": int(len(w)),
        "window": [YEAR_MIN, YEAR_MAX],
        "quick_mode": quick, "mcmc": cfg,
        "four_way_by_period": per.set_index("period")[
            [f"share_{c}" for c in CLASSES] + ["n"]].to_dict(orient="index"),
        "raw_shares": {
            "violation_2000_09": float(d[d.year.between(2000, 2009)].is_violation.mean()),
            "violation_2010_19": float(d[d.year.between(2010, 2019)].is_violation.mean()),
            "violation_2020_25": float(d[d.year.between(2020, 2025)].is_violation.mean()),
            "violation_2020_26": float(d[d.year.between(2020, 2026)].is_violation.mean()),
            "injury_2000_09": float(d[d.year.between(2000, 2009)].injuries_reported.mean()),
            "injury_2010_19": float(d[d.year.between(2010, 2019)].injuries_reported.mean()),
            "injury_2020_25": float(d[d.year.between(2020, 2025)].injuries_reported.mean()),
            "injury_2020_26": float(d[d.year.between(2020, 2026)].injuries_reported.mean()),
        },
        "bayes": bayes,
        "violation_x_injury": xt["violation_x_injury_overall"],
        "violation_share_by_hazard": {
            r["hazard"]: {"share": r["share"], "n": r["n"],
                          "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]}
            for _, r in vbh[vbh.series == "violation"].iterrows()},
        "violation_share_by_channel": {
            k: {"violation_share": v["violation_share"],
                "injury_share": v["injury_share"], "n": v["n"]}
            for k, v in xt["by_channel"]["strata"].items()},
        "units_weighted": uw.to_dict(orient="records"),
    }
    D.write_json(summary, RES / "summary.json")

    b = bayes
    print(f"[11] n = {len(d)} recalls, {len(w)} in {YEAR_MIN}-{YEAR_MAX}")
    print("[11] raw violation share: 2000-09 "
          f"{summary['raw_shares']['violation_2000_09']:.1%}  2010-19 "
          f"{summary['raw_shares']['violation_2010_19']:.1%}  2020-25 "
          f"{summary['raw_shares']['violation_2020_25']:.1%}")
    print("[11] raw injury    share: 2000-09 "
          f"{summary['raw_shares']['injury_2000_09']:.1%}  2010-19 "
          f"{summary['raw_shares']['injury_2010_19']:.1%}  2020-25 "
          f"{summary['raw_shares']['injury_2020_25']:.1%}")
    pm = b["period_means"]
    print(f"[11] posterior P(violation): 2000-09 {pm['violation_2000_09']['mean']:.3f} "
          f"-> 2020-25 {pm['violation_2020_25']['mean']:.3f};  "
          f"P(injury): {pm['injury_2000_09']['mean']:.3f} -> "
          f"{pm['injury_2020_25']['mean']:.3f}")
    dv = b["violation_share_2020_25_minus_2000_09"]
    print(f"[11] difference {dv['mean']:+.3f} "
          f"[{dv['hdi_lo']:+.3f},{dv['hdi_hi']:+.3f}]  "
          f"P(2020-25 > 2000-09) = "
          f"{b['posterior_prob_violation_2020_25_exceeds_2000_09']:.4f}")
    dc = b["decoupling"]
    print(f"[11] latent correlation r = {dc['correlation_of_latent_series']['mean']:+.3f} "
          f"[{dc['correlation_of_latent_series']['hdi_lo']:+.3f},"
          f"{dc['correlation_of_latent_series']['hdi_hi']:+.3f}];  "
          f"gap {dc['gap_2000_09']['mean']:+.3f} -> {dc['gap_2020_25']['mean']:+.3f} "
          f"(change {dc['gap_change']['mean']:+.3f} "
          f"[{dc['gap_change']['hdi_lo']:+.3f},{dc['gap_change']['hdi_hi']:+.3f}])")
    print(f"[11] MCMC (2 hyperparameters): rhat_max {b['rhat_max']:.4f}  "
          f"ess_min {b['ess_min']:.0f}  {b['seconds']:.1f}s;  importance-weight "
          f"ESS {b['importance_weight_ess']:.0f} "
          f"({b['importance_weight_ess_fraction']:.0%} of draws)")
    sp = b["sigma_posterior"]; gc = b["grid_integration_crosscheck"]
    print(f"[11] walk scale sigma: violation {sp['violation']['mean']:.3f} "
          f"(grid check {gc['violation']['mean']:.3f}), injury "
          f"{sp['injury']['mean']:.3f} (grid check {gc['injury']['mean']:.3f})")
    vx = xt["violation_x_injury_overall"]
    print(f"[11] violation x injury: OR = {vx['odds_ratio_haldane']:.3f} "
          f"[{vx['or_ci_lo']:.3f},{vx['or_ci_hi']:.3f}]  Fisher p = {vx['fisher_p']:.2g}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="500 draws / 500 tune instead of 6000 / 6000")
    main(**vars(ap.parse_args()))
