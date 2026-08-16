from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import data as D

RES = D.results_dir(__file__)

YEAR_MIN, YEAR_MAX = 2000, 2026

PERIOD_BINS = [1999, 2009, 2019, 2026]
PERIODS = ["2000-09", "2010-19", "2020-26"]

HAZ_KEEP = ["flammability_burn", "choking_small_parts", "drawstring_strangulation",
            "chemical", "fall_slip"]
HAZARDS = HAZ_KEEP + ["other"]
CHANNELS = ["online_only", "mixed", "store_only"]

ROLL_HALFWIDTH = 3
ROLL_MIN_DENOM = 8

N_BOOT_FULL, N_BOOT_QUICK = 10_000, 500
N_PERM_FULL, N_PERM_QUICK = 10_000, 500
N_SIM_FULL, N_SIM_QUICK = 4_000, 200


def build_frame() -> pd.DataFrame:
    df = D.load_recalls()
    d = df[df.year.between(YEAR_MIN, YEAR_MAX)].copy()
    d["period"] = pd.cut(d.year, bins=PERIOD_BINS, labels=PERIODS)
    d["hazard"] = np.where(d.hazard_category.isin(HAZ_KEEP),
                           d.hazard_category, "other")
    d["channel"] = d.sales_channel
    d["online_only"] = (d.channel == "online_only").astype(int)
    d["is_flam"] = (d.hazard == "flammability_burn").astype(int)
    return d


def g_test(obs: np.ndarray, exp: np.ndarray) -> float:
    o, e = np.asarray(obs, float), np.asarray(exp, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(o > 0, o * np.log(np.where(e > 0, o / e, 1.0)), 0.0)
    return float(2 * term.sum())


def expected_indep(tab: np.ndarray) -> np.ndarray:
    n = tab.sum()
    return np.outer(tab.sum(1), tab.sum(0)) / n if n else np.zeros_like(tab, float)


def std_residuals(tab: np.ndarray) -> np.ndarray:
    tab = np.asarray(tab, float)
    n = tab.sum()
    exp = expected_indep(tab)
    pr, pc = tab.sum(1) / n, tab.sum(0) / n
    denom = np.sqrt(exp * np.outer(1 - pr, 1 - pc))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, (tab - exp) / denom, 0.0)


def permutation_p(rows: np.ndarray, cols: np.ndarray, n_perm: int,
                  rng: np.random.Generator) -> float:
    nr, nc = rows.max() + 1, cols.max() + 1
    def chi2(r, c):
        t = np.zeros((nr, nc))
        np.add.at(t, (r, c), 1.0)
        e = expected_indep(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.where(e > 0, (t - e) ** 2 / e, 0.0).sum())
    obs = chi2(rows, cols)
    hits = 0
    c = cols.copy()
    for _ in range(n_perm):
        rng.shuffle(c)
        hits += chi2(rows, c) >= obs - 1e-12
    return (hits + 1) / (n_perm + 1)


def two_way_tests(d: pd.DataFrame, n_perm: int, rng) -> tuple[dict, dict]:
    hi = {h: i for i, h in enumerate(HAZARDS)}
    ci = {c: i for i, c in enumerate(CHANNELS)}
    out, tables = {}, {}
    for p in PERIODS:
        w = d[(d.period == p) & d.channel.isin(CHANNELS)]
        tab = np.zeros((len(CHANNELS), len(HAZARDS)))
        np.add.at(tab, (w.channel.map(ci).to_numpy(), w.hazard.map(hi).to_numpy()), 1.0)
        tables[p] = tab
        exp = expected_indep(tab)
        keep_r = tab.sum(1) > 0
        keep_c = tab.sum(0) > 0
        sub, sube = tab[np.ix_(keep_r, keep_c)], exp[np.ix_(keep_r, keep_c)]
        df_ = (keep_r.sum() - 1) * (keep_c.sum() - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = float(np.where(sube > 0, (sub - sube) ** 2 / sube, 0.0).sum())
        g = g_test(sub, sube)
        n = tab.sum()
        v = float(np.sqrt(chi2 / (n * (min(keep_r.sum(), keep_c.sum()) - 1)))) if n else np.nan
        out[p] = {
            "n": int(n),
            "chi2": chi2, "df": int(df_),
            "p_chi2": float(stats.chi2.sf(chi2, df_)),
            "G": g, "p_G": float(stats.chi2.sf(g, df_)),
            "cramers_v": v,
            "p_permutation": permutation_p(
                w.channel.map(ci).to_numpy(), w.hazard.map(hi).to_numpy(), n_perm, rng),
            "cells_expected_lt5": int((exp < 5).sum()),
            "cells_total": int(exp.size),
        }
    return out, tables


def ipf(obs: np.ndarray, margins: list[tuple], tol=1e-11, max_iter=5000):
    fit = np.full(obs.shape, obs.sum() / obs.size, dtype=float)
    for it in range(max_iter):
        prev = fit.copy()
        for m in margins:
            ax = tuple(a for a in range(obs.ndim) if a not in m)
            o = obs.sum(axis=ax, keepdims=True)
            f = fit.sum(axis=ax, keepdims=True)
            ratio = np.divide(o, f, out=np.zeros_like(f), where=f > 0)
            fit = fit * ratio
        if np.max(np.abs(fit - prev)) < tol:
            break
    return fit, it + 1


def n_params(shape: tuple, margins: list[tuple]) -> int:
    from itertools import combinations
    dims = {i: shape[i] - 1 for i in range(len(shape))}
    terms = set()
    for m in margins:
        for k in range(len(m) + 1):
            for sub in combinations(sorted(m), k):
                terms.add(sub)
    return int(sum(np.prod([dims[i] for i in t]) if t else 1 for t in terms))


def poisson_irls(y: np.ndarray, X: np.ndarray, max_iter=200, tol=1e-11):
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-6))
    for _ in range(max_iter):
        eta = X @ beta
        mu = np.exp(np.clip(eta, -30, 30))
        W = mu
        z = eta + (y - mu) / np.maximum(mu, 1e-10)
        XtW = X.T * W
        beta_new = np.linalg.solve(XtW @ X + 1e-10 * np.eye(X.shape[1]), XtW @ z)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    mu = np.exp(np.clip(X @ beta, -30, 30))
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(np.where(mu > 0, y / mu, 1.0)), 0.0)
    dev = float(2 * (term - (y - mu)).sum())
    return beta, mu, dev


def design_for(shape: tuple, margins: list[tuple]) -> np.ndarray:
    from itertools import combinations, product
    idx = np.array(list(product(*[range(s) for s in shape])))
    terms = set()
    for m in margins:
        for k in range(1, len(m) + 1):
            for sub in combinations(sorted(m), k):
                terms.add(sub)
    cols = [np.ones(len(idx))]
    for t in sorted(terms, key=lambda s: (len(s), s)):
        levels = [range(1, shape[a]) for a in t]
        for combo in product(*levels):
            col = np.ones(len(idx))
            for a, lv in zip(t, combo):
                col = col * (idx[:, a] == lv)
            cols.append(col.astype(float))
    return np.column_stack(cols)


def parametric_bootstrap_G2(tab: np.ndarray, margins: list[tuple], n_sim: int,
                            rng: np.random.Generator) -> dict:
    exp, _ = ipf(tab, margins)
    g_obs = g_test(tab, exp)
    p = (exp / exp.sum()).ravel()
    n = int(tab.sum())
    sims = np.empty(n_sim)
    for b in range(n_sim):
        sim = rng.multinomial(n, p).reshape(tab.shape).astype(float)
        e_sim, _ = ipf(sim, margins)
        sims[b] = g_test(sim, e_sim)
    return {"G2_observed": float(g_obs), "n_sim": int(n_sim),
            "p_parametric_bootstrap": float(((sims >= g_obs - 1e-12).sum() + 1) / (n_sim + 1)),
            "sim_mean_G2": float(sims.mean()), "sim_sd_G2": float(sims.std(ddof=1)),
            "sim_q95_G2": float(np.percentile(sims, 95))}


def decomposition(d: pd.DataFrame, n_boot: int, seed=99) -> dict:
    ch = CHANNELS + ["unknown"]
    cidx = {c: i for i, c in enumerate(ch)}
    d = d[d.period.isin(["2000-09", "2020-26"])]
    code = (d.period.eq("2020-26").to_numpy().astype(int) * len(ch) * 2
            + d.channel.map(cidx).to_numpy() * 2 + d.is_flam.to_numpy())
    size = 2 * len(ch) * 2

    def split(counts):
        c = counts.reshape(2, len(ch), 2)
        tot = c.sum(-1)
        n = tot.sum(-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(n > 0, tot / np.maximum(n, 1), 0.0)
            r = np.where(tot > 0, c[..., 1] / np.maximum(tot, 1), 0.0)
        comp = float(((w[1] - w[0]) * (r[0] + r[1]) / 2).sum())
        rate = float(((w[0] + w[1]) / 2 * (r[1] - r[0])).sum())
        return comp, rate, float((w[1] * r[1]).sum() - (w[0] * r[0]).sum())

    obs = np.bincount(code, minlength=size)
    comp, rate, total = split(obs)
    rng = np.random.default_rng(seed)
    n = len(d)
    bs = np.empty((n_boot, 3))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        bs[b] = split(np.bincount(code[idx], minlength=size))
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return {
        "total_change_in_flammability_share": {"estimate": total,
                                               "ci_lo": float(lo[2]), "ci_hi": float(hi[2])},
        "composition_component": {"estimate": comp, "ci_lo": float(lo[0]),
                                  "ci_hi": float(hi[0]),
                                  "share_of_total": comp / total if total else None},
        "within_channel_rate_component": {"estimate": rate, "ci_lo": float(lo[1]),
                                          "ci_hi": float(hi[1]),
                                          "share_of_total": rate / total if total else None},
        "periods": ["2000-09", "2020-26"],
        "reading": ("composition = how much of the flammability rise is explained "
                    "purely by the channel mix moving toward online-only, holding "
                    "each channel's own flammability rate at its period average"),
    }


def mantel_haenszel(d: pd.DataFrame) -> dict:
    w = d[d.channel.isin(CHANNELS)]
    strata, rows = [], []
    for p in PERIODS:
        s = w[w.period == p]
        a = int(((s.online_only == 1) & (s.is_flam == 1)).sum())
        b = int(((s.online_only == 1) & (s.is_flam == 0)).sum())
        c = int(((s.online_only == 0) & (s.is_flam == 1)).sum())
        e = int(((s.online_only == 0) & (s.is_flam == 0)).sum())
        strata.append((a, b, c, e))
        or_hat = (a + .5) * (e + .5) / ((b + .5) * (c + .5))
        se = np.sqrt(1 / (a + .5) + 1 / (b + .5) + 1 / (c + .5) + 1 / (e + .5))
        rows.append({"period": p, "online_flam": a, "online_other": b,
                     "storemix_flam": c, "storemix_other": e,
                     "odds_ratio_haldane": float(or_hat),
                     "ci_lo": float(np.exp(np.log(or_hat) - 1.959964 * se)),
                     "ci_hi": float(np.exp(np.log(or_hat) + 1.959964 * se))})

    num = sum(a * e / (a + b + c + e) for a, b, c, e in strata)
    den = sum(b * c / (a + b + c + e) for a, b, c, e in strata)
    or_mh = num / den
    P = [(a + e) / (a + b + c + e) for a, b, c, e in strata]
    Q = [(b + c) / (a + b + c + e) for a, b, c, e in strata]
    R = [a * e / (a + b + c + e) for a, b, c, e in strata]
    S = [b * c / (a + b + c + e) for a, b, c, e in strata]
    var = (sum(p_ * r for p_, r in zip(P, R)) / (2 * sum(R) ** 2)
           + sum(p_ * s + q * r for p_, q, r, s in zip(P, Q, R, S)) / (2 * sum(R) * sum(S))
           + sum(q * s for q, s in zip(Q, S)) / (2 * sum(S) ** 2))
    se_mh = np.sqrt(var)

    A = sum(a for a, b, c, e in strata)
    EA = sum((a + b) * (a + c) / (a + b + c + e) for a, b, c, e in strata)
    VA = sum((a + b) * (c + e) * (a + c) * (b + e)
             / ((a + b + c + e) ** 2 * (a + b + c + e - 1))
             for a, b, c, e in strata if (a + b + c + e) > 1)
    mh_chi2 = (abs(A - EA) - 0.5) ** 2 / VA

    bd = 0.0
    for a, b, c, e in strata:
        n1, n2, m1 = a + b, c + e, a + c
        n = a + b + c + e
        lo, hi = max(0.0, m1 - n2), min(n1, m1)
        f = lambda x: (x * (n2 - m1 + x)) - or_mh * ((n1 - x) * (m1 - x))
        x0, x1 = lo + 1e-9, hi - 1e-9
        if f(x0) * f(x1) < 0:
            for _ in range(200):
                mid = (x0 + x1) / 2
                if f(x0) * f(mid) <= 0:
                    x1 = mid
                else:
                    x0 = mid
            Ea = (x0 + x1) / 2
            v = 1 / (1 / max(Ea, 1e-9) + 1 / max(n1 - Ea, 1e-9)
                     + 1 / max(m1 - Ea, 1e-9) + 1 / max(n2 - m1 + Ea, 1e-9))
            bd += (a - Ea) ** 2 / v
    bd_df = len(strata) - 1
    return {
        "by_period": rows,
        "common_odds_ratio_mh": float(or_mh),
        "ci_lo": float(np.exp(np.log(or_mh) - 1.959964 * se_mh)),
        "ci_hi": float(np.exp(np.log(or_mh) + 1.959964 * se_mh)),
        "mh_chi2": float(mh_chi2), "mh_p": float(stats.chi2.sf(mh_chi2, 1)),
        "breslow_day_chi2": float(bd), "breslow_day_df": bd_df,
        "breslow_day_p": float(stats.chi2.sf(bd, bd_df)),
        "reading": ("common OR = how many times more likely an online-only recall "
                    "is to be a flammability recall than a store/mixed recall; a "
                    "large Breslow-Day p means that multiple has not changed"),
    }


def loglinear(d: pd.DataFrame, n_sim: int = 0, rng=None) -> dict:
    hi = {h: i for i, h in enumerate(HAZARDS)}
    ci = {c: i for i, c in enumerate(CHANNELS)}
    pi = {p: i for i, p in enumerate(PERIODS)}
    w = d[d.channel.isin(CHANNELS)]
    tab = np.zeros((len(HAZARDS), len(CHANNELS), len(PERIODS)))
    np.add.at(tab, (w.hazard.map(hi).to_numpy(), w.channel.map(ci).to_numpy(),
                    w.period.map(pi).to_numpy()), 1.0)

    models = {
        "[H][C][P] mutual independence": [(0,), (1,), (2,)],
        "[HP][CP] hazard-channel independent given period": [(0, 2), (1, 2)],
        "[HC][HP][CP] homogeneous association": [(0, 1), (0, 2), (1, 2)],
        "[HCP] saturated": [(0, 1, 2)],
    }
    fits = {}
    for name, margins in models.items():
        exp, iters = ipf(tab, margins)
        k = n_params(tab.shape, margins)
        dfree = int(tab.size - k)
        g = g_test(tab, exp)
        with np.errstate(divide="ignore", invalid="ignore"):
            x2 = float(np.where(exp > 0, (tab - exp) ** 2 / exp, 0.0).sum())
        X = design_for(tab.shape, margins)
        _, mu, dev = poisson_irls(tab.ravel(), X)
        fits[name] = {
            "margins": [list(m) for m in margins],
            "n_params": k, "df": dfree,
            "G2_ipf": g, "X2": x2,
            "p_value": float(stats.chi2.sf(g, dfree)) if dfree > 0 else None,
            "G2_poisson_irls": dev,
            "irls_design_rank": int(np.linalg.matrix_rank(X)),
            "ipf_iterations": iters,
            "ipf_vs_irls_abs_diff": abs(g - dev),
            "aic": g - 2 * dfree,
            "bic": g - dfree * np.log(tab.sum()),
        }

    m2 = fits["[HP][CP] hazard-channel independent given period"]
    m3 = fits["[HC][HP][CP] homogeneous association"]
    comparisons = {
        "is_there_any_hazard_channel_association": {
            "compare": "[HP][CP]  vs  [HC][HP][CP]",
            "delta_G2": m2["G2_ipf"] - m3["G2_ipf"],
            "delta_df": m2["df"] - m3["df"],
            "p_value": float(stats.chi2.sf(m2["G2_ipf"] - m3["G2_ipf"],
                                           m2["df"] - m3["df"])),
        },
        "did_the_association_change_over_periods": {
            "compare": "[HC][HP][CP]  vs  [HCP] saturated",
            "delta_G2": m3["G2_ipf"],
            "delta_df": m3["df"],
            "p_value": float(stats.chi2.sf(m3["G2_ipf"], m3["df"])),
            "reading": ("a small p means the hazard x channel association is NOT "
                        "constant across periods - the three-way term is needed"),
        },
    }
    if n_sim:
        comparisons["did_the_association_change_over_periods"]["sparse_table_check"] = \
            parametric_bootstrap_G2(tab, models["[HC][HP][CP] homogeneous association"],
                                    n_sim, rng or np.random.default_rng(11))
    return {"table_dims": {"hazard": HAZARDS, "channel": CHANNELS, "period": PERIODS},
            "table_counts": tab.tolist(), "n": int(tab.sum()),
            "models": fits, "comparisons": comparisons}


def multinomial_fit(X: np.ndarray, y: np.ndarray, K: int, *, ref: int,
                    prior_sd_slope=2.5, prior_sd_intercept=10.0,
                    max_iter=200, tol=1e-10):
    n, p = X.shape
    others = [k for k in range(K) if k != ref]
    m = len(others)
    lam = np.full(p, 1.0 / prior_sd_slope ** 2)
    lam[0] = 1.0 / prior_sd_intercept ** 2
    LAM = np.diag(np.tile(lam, m))
    Y = np.zeros((n, K))
    Y[np.arange(n), y] = 1.0
    Yo = Y[:, others]

    beta = np.zeros((m, p))
    def negll(b):
        eta = np.zeros((n, K))
        eta[:, others] = X @ b.T
        mx = eta.max(axis=1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(eta - mx).sum(axis=1))
        ll = float((eta[np.arange(n), y] - lse).sum())
        pen = 0.5 * float(b.ravel() @ LAM @ b.ravel())
        return -ll + pen, ll

    obj, ll = negll(beta)
    converged = False
    for _ in range(max_iter):
        eta = np.zeros((n, K))
        eta[:, others] = X @ beta.T
        mx = eta.max(axis=1, keepdims=True)
        e = np.exp(eta - mx)
        P = e / e.sum(axis=1, keepdims=True)
        Po = P[:, others]
        grad = (X.T @ (Yo - Po)).T.ravel() - LAM @ beta.ravel()
        H = np.zeros((m * p, m * p))
        for a in range(m):
            for b_ in range(m):
                wgt = Po[:, a] * ((a == b_) - Po[:, b_])
                H[a * p:(a + 1) * p, b_ * p:(b_ + 1) * p] = (X.T * wgt) @ X
        H = H + LAM
        step = np.linalg.solve(H + 1e-9 * np.eye(m * p), grad)
        t = 1.0
        for _ in range(40):
            cand = beta + t * step.reshape(m, p)
            o, l2 = negll(cand)
            if o < obj:
                break
            t /= 2
        if np.max(np.abs(t * step)) < tol:
            beta, obj, ll = cand, o, l2
            converged = True
            break
        beta, obj, ll = cand, o, l2
    cov = np.linalg.inv(H + 1e-9 * np.eye(m * p))
    return beta, cov, ll, converged, others


def multinomial_analysis(d: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    w = d[d.channel.isin(CHANNELS)].copy()
    hi = {h: i for i, h in enumerate(HAZARDS)}
    y = w.hazard.map(hi).to_numpy()
    K = len(HAZARDS)
    ref = hi["choking_small_parts"]

    p10 = (w.period == "2010-19").astype(float).to_numpy()
    p20 = (w.period == "2020-26").astype(float).to_numpy()
    oo = w.online_only.to_numpy(float)
    kid = w.is_childrens.astype(float).to_numpy()

    specs = {
        "main_effects": (["intercept", "online_only", "period_2010_19",
                          "period_2020_26", "is_childrens"],
                         np.column_stack([np.ones(len(w)), oo, p10, p20, kid])),
        "with_online_x_period": (["intercept", "online_only", "period_2010_19",
                                  "period_2020_26", "is_childrens",
                                  "online_x_2010_19", "online_x_2020_26"],
                                 np.column_stack([np.ones(len(w)), oo, p10, p20, kid,
                                                  oo * p10, oo * p20])),
    }
    fitted, rows = {}, []
    for name, (cols, X) in specs.items():
        beta, cov, ll, conv, others = multinomial_fit(X, y, K, ref=ref)
        se = np.sqrt(np.clip(np.diag(cov), 0, None)).reshape(beta.shape)
        fitted[name] = dict(beta=beta, cov=cov, ll=ll, conv=conv,
                            cols=cols, X=X, others=others,
                            n_params=beta.size)
        for a, k in enumerate(others):
            for j, c in enumerate(cols):
                b, s = beta[a, j], se[a, j]
                rows.append({
                    "model": name, "outcome": HAZARDS[k],
                    "reference_outcome": HAZARDS[ref], "term": c,
                    "coef": b, "se": s, "z": b / s if s > 0 else np.nan,
                    "p_wald": float(2 * stats.norm.sf(abs(b / s))) if s > 0 else np.nan,
                    "ci_lo": b - 1.959964 * s, "ci_hi": b + 1.959964 * s,
                    "rrr": float(np.exp(b)),
                    "rrr_lo": float(np.exp(b - 1.959964 * s)),
                    "rrr_hi": float(np.exp(b + 1.959964 * s)),
                })
    coef_df = pd.DataFrame(rows)

    m1, m2 = fitted["main_effects"], fitted["with_online_x_period"]
    lr = {"statistic": 2 * (m2["ll"] - m1["ll"]),
          "df": m2["n_params"] - m1["n_params"]}
    lr["p_value"] = float(stats.chi2.sf(max(lr["statistic"], 0), lr["df"]))

    def predict(model, x):
        beta, others = model["beta"], model["others"]
        eta = np.zeros(K)
        eta[others] = beta @ np.asarray(x, float)
        e = np.exp(eta - eta.max())
        return e / e.sum()

    scen = []
    for per, (a, b) in {"2000-09": (0., 0.), "2010-19": (1., 0.),
                        "2020-26": (0., 1.)}.items():
        for online in (1., 0.):
            for kidv in (1., 0.):
                x = [1., online, a, b, kidv, online * a, online * b]
                pr = predict(m2, x)
                scen.append({"period": per,
                             "channel": "online_only" if online else "store_or_mixed",
                             "is_childrens": bool(kidv),
                             **{f"p_{HAZARDS[k]}": float(pr[k]) for k in range(K)}})
    return ({
        "outcome_levels": HAZARDS,
        "reference_outcome": HAZARDS[ref],
        "n": int(len(w)),
        "converged": {k: bool(v["conv"]) for k, v in fitted.items()},
        "loglik": {k: float(v["ll"]) for k, v in fitted.items()},
        "n_coefficients": {k: int(v["n_params"]) for k, v in fitted.items()},
        "lr_test_online_x_period": lr,
        "ridge_prior_sd": {"slopes": 2.5, "intercepts": 10.0},
        "predicted_probabilities": scen,
    }, coef_df)


def share_engine(d: pd.DataFrame):
    years = np.arange(YEAR_MIN, YEAR_MAX + 1)
    yidx = {y: i for i, y in enumerate(years)}
    ch_all = CHANNELS + ["unknown"]
    cidx = {c: i for i, c in enumerate(ch_all)}
    pidx = {p: i for i, p in enumerate(PERIODS)}
    code_y = (d.year.map(yidx).to_numpy() * len(ch_all) * 2
              + d.channel.map(cidx).to_numpy() * 2 + d.is_flam.to_numpy())
    code_p = (d.period.map(pidx).to_numpy() * len(ch_all) * 2
              + d.channel.map(cidx).to_numpy() * 2 + d.is_flam.to_numpy())
    ny, nc = len(years), len(ch_all)
    return years, ch_all, code_y, code_p, ny, nc


def _shares_from_counts(cnt, nc):
    flam = cnt[..., 1]
    tot = cnt.sum(axis=-1)
    oo, so = 0, 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "flam_given_online_only": np.where(tot[..., oo] > 0,
                                               flam[..., oo] / np.maximum(tot[..., oo], 1), np.nan),
            "flam_given_store_only": np.where(tot[..., so] > 0,
                                              flam[..., so] / np.maximum(tot[..., so], 1), np.nan),
            "online_only_given_flam": np.where(flam.sum(-1) > 0,
                                               flam[..., oo] / np.maximum(flam.sum(-1), 1), np.nan),
            "store_only_given_flam": np.where(flam.sum(-1) > 0,
                                              flam[..., so] / np.maximum(flam.sum(-1), 1), np.nan),
        }


def bootstrap_shares(d: pd.DataFrame, n_boot: int, seed=20260813):
    rng = np.random.default_rng(seed)
    years, ch_all, code_y, code_p, ny, nc = share_engine(d)
    n = len(d)
    size_y, size_p = ny * nc * 2, len(PERIODS) * nc * 2

    W = np.zeros((ny, ny))
    for i in range(ny):
        lo, hi = max(0, i - ROLL_HALFWIDTH), min(ny - 1, i + ROLL_HALFWIDTH)
        W[i, lo:hi + 1] = 1.0

    obs_y = np.bincount(code_y, minlength=size_y).reshape(ny, nc, 2)
    obs_p = np.bincount(code_p, minlength=size_p).reshape(len(PERIODS), nc, 2)
    obs_roll = np.tensordot(W, obs_y, axes=(1, 0))

    keys = ["flam_given_online_only", "flam_given_store_only",
            "online_only_given_flam", "store_only_given_flam"]
    boot_p = {k: np.empty((n_boot, len(PERIODS))) for k in keys}
    boot_r = {k: np.empty((n_boot, ny)) for k in keys}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        cy = np.bincount(code_y[idx], minlength=size_y).reshape(ny, nc, 2)
        cp = np.bincount(code_p[idx], minlength=size_p).reshape(len(PERIODS), nc, 2)
        sp = _shares_from_counts(cp, nc)
        sr = _shares_from_counts(np.tensordot(W, cy, axes=(1, 0)), nc)
        for k in keys:
            boot_p[k][b] = sp[k]
            boot_r[k][b] = sr[k]

    def pct(a, q):
        with np.errstate(invalid="ignore"):
            return np.nanpercentile(a, q, axis=0)

    obs_sp, obs_sr = _shares_from_counts(obs_p, nc), _shares_from_counts(obs_roll, nc)
    period_rows = []
    for i, p in enumerate(PERIODS):
        for k in keys:
            period_rows.append({
                "period": p, "quantity": k,
                "estimate": float(obs_sp[k][i]),
                "ci_lo": float(pct(boot_p[k], 2.5)[i]),
                "ci_hi": float(pct(boot_p[k], 97.5)[i]),
                "denominator": int(obs_p[i, 0].sum() if k == "flam_given_online_only"
                                   else obs_p[i, 2].sum() if k == "flam_given_store_only"
                                   else obs_p[i, :, 1].sum()),
            })
    roll_rows = []
    denom_oo = obs_roll[:, 0].sum(-1)
    denom_so = obs_roll[:, 2].sum(-1)
    denom_fl = obs_roll[:, :, 1].sum(-1)
    dens = {"flam_given_online_only": denom_oo, "flam_given_store_only": denom_so,
            "online_only_given_flam": denom_fl, "store_only_given_flam": denom_fl}
    for k in keys:
        lo, hi = pct(boot_r[k], 3.0), pct(boot_r[k], 97.0)
        for i, y in enumerate(years):
            roll_rows.append({
                "year": int(y), "quantity": k,
                "estimate": float(obs_sr[k][i]),
                "ci_lo": float(lo[i]), "ci_hi": float(hi[i]),
                "denominator": int(dens[k][i]),
                "plotted": bool(dens[k][i] >= ROLL_MIN_DENOM),
            })

    diffs = {}
    for k in keys:
        dd = boot_p[k][:, 2] - boot_p[k][:, 0]
        diffs[k] = {
            "estimate": float(obs_sp[k][2] - obs_sp[k][0]),
            "ci_lo": float(np.nanpercentile(dd, 2.5)),
            "ci_hi": float(np.nanpercentile(dd, 97.5)),
            "p_boot_ge_0": float(np.nanmean(dd <= 0)),
        }
    return pd.DataFrame(period_rows), pd.DataFrame(roll_rows), diffs


def main(quick=False):
    n_boot = N_BOOT_QUICK if quick else N_BOOT_FULL
    n_perm = N_PERM_QUICK if quick else N_PERM_FULL
    n_sim = N_SIM_QUICK if quick else N_SIM_FULL
    rng = np.random.default_rng(7)

    d = build_frame()
    n_unknown = int((d.channel == "unknown").sum())

    tests, tables = two_way_tests(d, n_perm, rng)
    long_rows = []
    for p in PERIODS:
        tab = tables[p]
        for ci_, c in enumerate(CHANNELS):
            for hi_, h in enumerate(HAZARDS):
                n_cp = tab[ci_].sum()
                n_hp = tab[:, hi_].sum()
                long_rows.append({
                    "period": p, "channel": c, "hazard": h,
                    "count": int(tab[ci_, hi_]),
                    "share_within_channel": float(tab[ci_, hi_] / n_cp) if n_cp else np.nan,
                    "share_within_hazard": float(tab[ci_, hi_] / n_hp) if n_hp else np.nan,
                    "share_of_period": float(tab[ci_, hi_] / tab.sum()) if tab.sum() else np.nan,
                })
    cont = pd.DataFrame(long_rows)
    cont.to_csv(RES / "contingency.csv", index=False)

    d2 = d[d.channel.isin(CHANNELS)].copy()
    d2["channel_group"] = np.where(d2.online_only == 1, "online_only", "store_or_mixed")
    comp = (d2.groupby(["period", "channel_group", "hazard"], observed=True)
              .size().rename("count").reset_index())
    tot = comp.groupby(["period", "channel_group"], observed=True)["count"].transform("sum")
    comp["share"] = comp["count"] / tot
    comp["group_n"] = tot
    comp.to_csv(RES / "composition_panelA.csv", index=False)

    D.write_json({"by_period": tests,
                  "note": ("expected counts are small in several cells, so the "
                           "permutation p-value is the one to quote"),
                  "channel_unknown_excluded": n_unknown},
                 RES / "contingency_tests.json")

    ll = loglinear(d, n_sim=n_sim, rng=rng)
    D.write_json(ll, RES / "loglinear.json")

    decomp = decomposition(d, n_boot)
    mh = mantel_haenszel(d)
    D.write_json({"kitagawa_decomposition": decomp,
                  "mantel_haenszel_stability": mh}, RES / "decomposition.json")
    pd.DataFrame(mh["by_period"]).to_csv(RES / "odds_ratios_by_period.csv", index=False)

    mn, coef = multinomial_analysis(d)
    coef.to_csv(RES / "multinomial_coefficients.csv", index=False)
    D.write_json(mn, RES / "multinomial.json")

    period_sh, roll_sh, diffs = bootstrap_shares(d, n_boot)
    period_sh.to_csv(RES / "key_shares.csv", index=False)
    roll_sh.to_csv(RES / "rolling_shares.csv", index=False)

    r = std_residuals(tables["2020-26"])
    res_rows = [{"channel": c, "hazard": h,
                 "observed": float(tables["2020-26"][i, j]),
                 "expected": float(expected_indep(tables["2020-26"])[i, j]),
                 "std_residual": float(r[i, j])}
                for i, c in enumerate(CHANNELS) for j, h in enumerate(HAZARDS)]
    pd.DataFrame(res_rows).to_csv(RES / "residuals_2020_26.csv", index=False)

    def sh(period, q):
        row = period_sh[(period_sh.period == period) & (period_sh.quantity == q)]
        return {k: float(row.iloc[0][k]) for k in ["estimate", "ci_lo", "ci_hi"]} | \
               {"n": int(row.iloc[0]["denominator"])}

    m = d[d.year.between(2020, 2025)]
    m_oo = m[m.channel == "online_only"]
    m_so = m[m.channel == "store_only"]
    manuscript = {
        "n_2020_25": int(len(m)),
        "flam_share_2020_25": float(m.is_flam.mean()),
        "online_only_share_2020_25": float((m.channel == "online_only").mean()),
        "flam_given_online_only_2020_25": float(m_oo.is_flam.mean()),
        "n_online_only_2020_25": int(len(m_oo)),
        "flam_given_store_only_2020_25": float(m_so.is_flam.mean()) if len(m_so) else None,
        "n_store_only_2020_25": int(len(m_so)),
        "online_only_given_flam_2020_25":
            float((m[m.is_flam == 1].channel == "online_only").mean()),
    }
    base = d[d.year.between(2000, 2009)]
    summary = {
        "analysis": "10_hazard_channel_regime",
        "source_version": D.load_recalls().attrs.get("source_version"),
        "n_records_2000_2026": int(len(d)),
        "n_channel_unknown_excluded": n_unknown,
        "hazard_levels": HAZARDS, "channel_levels": CHANNELS, "periods": PERIODS,
        "quick_mode": quick, "n_bootstrap": n_boot, "n_permutations": n_perm,
        "headline": {
            "flam_share_by_period":
                {p: float(d[d.period == p].is_flam.mean()) for p in PERIODS},
            "online_only_share_by_period":
                {p: float((d[d.period == p].channel == "online_only").mean())
                 for p in PERIODS},
            "flam_given_online_only": {p: sh(p, "flam_given_online_only") for p in PERIODS},
            "flam_given_store_only": {p: sh(p, "flam_given_store_only") for p in PERIODS},
            "online_only_given_flam": {p: sh(p, "online_only_given_flam") for p in PERIODS},
            "store_only_given_flam": {p: sh(p, "store_only_given_flam") for p in PERIODS},
            "modern_minus_baseline": diffs,
        },
        "two_way_tests": tests,
        "loglinear_comparisons": ll["comparisons"],
        "decomposition": decomp,
        "mantel_haenszel_stability": mh,
        "loglinear_fit": {k: {kk: v[kk] for kk in ("G2_ipf", "df", "p_value",
                                                   "G2_poisson_irls", "bic")}
                          for k, v in ll["models"].items()},
        "multinomial": {
            "lr_test_online_x_period": mn["lr_test_online_x_period"],
            "p_flammability_by_scenario": [
                {k: s[k] for k in ("period", "channel", "is_childrens")}
                | {"p_flammability": s["p_flammability_burn"]}
                for s in mn["predicted_probabilities"]],
        },
        "manuscript_2020_25_window": manuscript,
        "baseline_2000_09": {
            "n": int(len(base)),
            "flam_share": float(base.is_flam.mean()),
            "online_only_share": float((base.channel == "online_only").mean()),
        },
    }
    D.write_json(summary, RES / "summary.json")

    h = summary["headline"]
    print(f"[10] n = {len(d)} recalls 2000-2026 ({n_unknown} channel-unknown kept "
          f"in descriptives, dropped from models)")
    print(f"[10] flammability share: " + "  ".join(
        f"{p} {h['flam_share_by_period'][p]:.1%}" for p in PERIODS))
    print(f"[10] online-only share : " + "  ".join(
        f"{p} {h['online_only_share_by_period'][p]:.1%}" for p in PERIODS))
    for q in ("flam_given_online_only", "flam_given_store_only", "online_only_given_flam"):
        print(f"[10] {q:24s}: " + "  ".join(
            f"{p} {h[q][p]['estimate']:.1%} [{h[q][p]['ci_lo']:.1%},{h[q][p]['ci_hi']:.1%}]"
            for p in PERIODS))
    a = ll["comparisons"]["is_there_any_hazard_channel_association"]
    print(f"[10] hazard x channel association  G2 = {a['delta_G2']:.1f}  "
          f"df = {a['delta_df']}  p = {a['p_value']:.2g}")
    c = ll["comparisons"]["did_the_association_change_over_periods"]
    print(f"[10] three-way test  G2 = {c['delta_G2']:.1f}  df = {c['delta_df']}  "
          f"p = {c['p_value']:.2g}  (parametric bootstrap p = "
          f"{c['sparse_table_check']['p_parametric_bootstrap']:.3f})")
    print(f"[10] MH common OR (online-only -> flammability) = "
          f"{mh['common_odds_ratio_mh']:.2f} [{mh['ci_lo']:.2f},{mh['ci_hi']:.2f}]  "
          f"Breslow-Day p = {mh['breslow_day_p']:.2f} (homogeneous across periods)")
    dc = decomp
    print(f"[10] flammability share rose {dc['total_change_in_flammability_share']['estimate']:+.1%}"
          f" 2000-09 -> 2020-26: composition "
          f"{dc['composition_component']['estimate']:+.1%} "
          f"({dc['composition_component']['share_of_total']:.0%}), within-channel rate "
          f"{dc['within_channel_rate_component']['estimate']:+.1%} "
          f"({dc['within_channel_rate_component']['share_of_total']:.0%})")
    lr = mn["lr_test_online_x_period"]
    print(f"[10] multinomial online x period LR = {lr['statistic']:.1f} "
          f"df = {lr['df']} p = {lr['p_value']:.2g}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="500 bootstrap / permutation draws instead of 10,000")
    main(**vars(ap.parse_args()))
