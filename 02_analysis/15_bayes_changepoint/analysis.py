from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import data as D
from _common import mcmc as M

RES = D.results_dir(__file__)
VIOLATION_CSV = (D.ANALYSIS / "11_violation_vs_incident" / "results"
                 / "annual_classification.csv")

HAZARD_ORDER = [
    "flammability_burn", "choking_small_parts", "drawstring_strangulation",
    "chemical", "fall_slip", "laceration_puncture", "entrapment_entanglement",
    "protective_failure", "other",
]

MIN_SEG = 3
MIN_N_YEAR = 3
KS = (0, 1, 2)


@dataclass
class Series:
    key: str
    label: str
    years: np.ndarray
    counts: np.ndarray
    cats: list
    ref: int
    note: str = ""
    fits: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return len(self.years)

    @property
    def K(self) -> int:
        return self.counts.shape[1]


def hazard_series(df: pd.DataFrame) -> Series:
    cats = [c for c in HAZARD_ORDER if c in set(df.hazard_category)]
    ct = (pd.crosstab(df.year, df.hazard_category)
            .reindex(columns=cats, fill_value=0).sort_index())
    ct = ct[ct.sum(axis=1) > 0]
    counts = ct.to_numpy(float)
    return Series("hazard", "Hazard composition", ct.index.to_numpy(int), counts,
                  cats, int(np.argmax(counts.sum(axis=0))),
                  note=f"all {int(counts.sum())} recalls, "
                       f"{len(cats)} hazard categories")


def channel_series(df: pd.DataFrame) -> Series:
    kn = df[df.sales_channel.ne("unknown")]
    g = (kn.assign(online=kn.sales_channel.eq("online_only"))
           .groupby("year").agg(n=("online", "size"), k=("online", "sum")))
    g = g[g.n >= MIN_N_YEAR]
    counts = np.column_stack([g.k.to_numpy(float), (g.n - g.k).to_numpy(float)])
    return Series("channel", "Online-only share", g.index.to_numpy(int), counts,
                  ["online_only", "not_online_only"], 1,
                  note=(f"{int(counts.sum())} recalls with a known sales channel, "
                        f"years with >= {MIN_N_YEAR} such records"))


def violation_series(df: pd.DataFrame) -> tuple[Series, str]:
    if VIOLATION_CSV.exists():
        a = pd.read_csv(VIOLATION_CSV)
        g = pd.DataFrame({"year": a.year.to_numpy(int),
                          "k": (a.violation_only + a["both"]).to_numpy(float),
                          "n": a.n.to_numpy(float)}).set_index("year")
        src = "11_violation_vs_incident/results/annual_classification.csv"
    else:
        v = df.groupby("year").agg(n=("is_violation", "size"),
                                   k=("is_violation", "sum")).astype(float)
        g = v[v.index >= 2000]
        src = "recomputed from apparel_recalls_v2.is_violation (build 11 absent)"
    g = g[g.n >= MIN_N_YEAR]
    counts = np.column_stack([g.k.to_numpy(float), (g.n - g.k).to_numpy(float)])
    return Series("violation", "Violation-detected share", g.index.to_numpy(int),
                  counts, ["violation", "no_violation"], 1,
                  note=f"{int(counts.sum())} recalls, source: {src}"), src


def unpack(theta: np.ndarray, R: int, K: int, ref: int) -> np.ndarray:
    th = theta.reshape(R, K)
    a0 = np.exp(np.clip(th[:, 0], -8, 12))[:, None]
    z = np.zeros((R, K))
    free = [k for k in range(K) if k != ref]
    z[:, free] = th[:, 1:]
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return np.maximum(a0 * p, 1e-8)


def seg_loglik(counts: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.array([M.log_dirichlet_multinomial(
        counts, np.broadcast_to(a, counts.shape)) for a in alpha])


def cp_weights(L: np.ndarray, K: int, min_seg: int):
    T = L.shape[1]
    if K == 0:
        return float(L[0].sum()), None
    cum = np.concatenate([np.zeros((L.shape[0], 1)), np.cumsum(L, axis=1)], axis=1)
    if K == 1:
        s = np.arange(min_seg, T - min_seg + 1)
        tot = cum[0][s] + (cum[1][T] - cum[1][s])
        lw = np.full(T + 1, -np.inf)
        lw[s] = tot
        return float(M.log_sum_exp(tot) - np.log(len(s))), lw
    a = cum[0] - cum[1]
    b = cum[1] - cum[2]
    grid = a[:, None] + b[None, :] + cum[2][T]
    s1 = np.arange(T + 1)[:, None]
    s2 = np.arange(T + 1)[None, :]
    ok = ((s1 >= min_seg) & (s2 >= s1 + min_seg) & (s2 <= T - min_seg))
    lw = np.where(ok, grid, -np.inf)
    return float(M.log_sum_exp(lw[ok]) - np.log(int(ok.sum()))), lw


def make_log_post(series: Series, K: int, min_seg=MIN_SEG):
    R, Kc, ref, counts = K + 1, series.K, series.ref, series.counts

    def log_post(theta):
        alpha = unpack(theta, R, Kc, ref)
        ll, _ = cp_weights(seg_loglik(counts, alpha), K, min_seg)
        th = theta.reshape(R, Kc)
        lp = (np.sum(M.log_normal(th[:, 0], np.log(20.0), 1.5))
              + np.sum(M.log_normal(th[:, 1:], 0.0, 2.0)))
        return float(ll + lp)

    names = []
    for r in range(R):
        names.append(f"log_conc[{r}]")
        names += [f"z[{r}]:{c}" for k, c in enumerate(series.cats) if k != ref]
    blocks = []
    for r in range(R):
        o = r * Kc
        blocks.append(np.array([o]))
        idx = np.arange(o + 1, o + Kc)
        blocks += [idx[i:i + 4] for i in range(0, len(idx), 4)]
    return log_post, names, blocks


def init_theta(series: Series, K: int) -> np.ndarray:
    R, Kc, ref = K + 1, series.K, series.ref
    edges = np.linspace(0, series.T, R + 1).astype(int)
    th = np.zeros((R, Kc))
    for r in range(R):
        seg = series.counts[edges[r]:edges[r + 1]].sum(axis=0) + 0.5
        p = seg / seg.sum()
        z = np.log(p) - np.log(p[ref])
        th[r, 0] = np.log(20.0)
        th[r, 1:] = np.clip([z[k] for k in range(Kc) if k != ref], -6, 6)
    return th.ravel()


def fit_model(series: Series, K: int, *, n_draws, n_tune, n_chains, seed,
              rhat_max, ess_min, max_attempts):
    log_post, names, blocks = make_log_post(series, K)
    init = init_theta(series, K)
    attempts, thin, tune = [], 1, n_tune
    post = None
    for a in range(max_attempts):
        t0 = time.time()
        if len(init) <= 6:
            post = M.sample(log_post, init, n_draws=n_draws, n_tune=tune,
                            n_chains=n_chains, seed=seed + 13 * a, names=names,
                            thin=thin, verbose=True)
            sampler = "sample"
        else:
            post = M.sample_blocked(log_post, init, blocks=blocks,
                                    n_draws=n_draws, n_tune=tune,
                                    n_chains=n_chains, seed=seed + 13 * a,
                                    names=names, thin=thin, target=0.35,
                                    verbose=True)
            sampler = "sample_blocked"
        r, e = post.rhat(), post.ess()
        rec = {"attempt": a + 1, "n_tune": tune, "thin": thin,
               "max_rhat": float(np.nanmax(r)), "min_ess": float(np.nanmin(e)),
               "worst_rhat_param": names[int(np.nanargmax(r))],
               "seconds": time.time() - t0,
               "accept_rate": [float(x) for x in post.accept_rate]}
        rec["passed"] = bool(rec["max_rhat"] <= rhat_max
                             and rec["min_ess"] >= ess_min)
        attempts.append(rec)
        print(f"    [{series.key} K={K}] attempt {a + 1}: max Rhat "
              f"{rec['max_rhat']:.3f} ({rec['worst_rhat_param']}), min ESS "
              f"{rec['min_ess']:.0f}, {rec['seconds']:.0f}s "
              f"-> {'PASS' if rec['passed'] else 'RETUNE'}")
        if rec["passed"]:
            break
        tune, thin = int(tune * 2), thin * 2
    info = {"series": series.key, "K": K, "n_params": len(init),
            "sampler": sampler, "n_chains": n_chains, "n_draws": n_draws,
            "attempts": attempts, "max_rhat": attempts[-1]["max_rhat"],
            "min_ess": attempts[-1]["min_ess"],
            "mean_accept_rate": float(np.mean(attempts[-1]["accept_rate"])),
            "seconds_total": float(sum(x["seconds"] for x in attempts)),
            "reportable": attempts[-1]["passed"],
            "rhat_gate": rhat_max, "ess_gate": ess_min}
    return post, log_post, names, info


def hdi1(x, prob=0.94) -> tuple[float, float]:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    k = max(int(np.floor(prob * n)), 1)
    if k >= n:
        return float(x[0]), float(x[-1])
    w = x[k:] - x[:n - k]
    i = int(np.argmin(w))
    return float(x[i]), float(x[i + k])


def process(series: Series, K: int, post, n_keep=6000, seed=7):
    R, Kc, ref, T = K + 1, series.K, series.ref, series.T
    f = post.flat()
    rng = np.random.default_rng(seed)
    idx = (rng.choice(len(f), size=n_keep, replace=False)
           if len(f) > n_keep else np.arange(len(f)))

    w1 = np.zeros(T + 1)
    w2 = np.zeros((T + 1, T + 1))
    comp = np.empty((len(idx), R, Kc))
    conc = np.empty((len(idx), R))
    ll_point = np.empty((len(idx), T))
    cps = np.empty((len(idx), max(K, 1)), dtype=int)

    for i, s in enumerate(idx):
        theta = f[s]
        alpha = unpack(theta, R, Kc, ref)
        comp[i] = alpha / alpha.sum(axis=1, keepdims=True)
        conc[i] = alpha.sum(axis=1)
        L = seg_loglik(series.counts, alpha)
        _, lw = cp_weights(L, K, MIN_SEG)
        if K == 0:
            ll_point[i] = L[0]
            cps[i] = -1
            continue
        flat = lw.ravel()
        p = np.exp(flat - M.log_sum_exp(flat[np.isfinite(flat)]))
        p = np.where(np.isfinite(p), p, 0.0)
        p /= p.sum()
        pick = rng.choice(len(p), p=p)
        if K == 1:
            w1 += p
            s1 = int(pick)
            cps[i] = [s1]
            ll_point[i] = np.concatenate([L[0, :s1], L[1, s1:]])
        else:
            P = p.reshape(T + 1, T + 1)
            w2 += P
            w1 += P.sum(axis=1)
            s1, s2 = int(pick // (T + 1)), int(pick % (T + 1))
            cps[i] = [s1, s2]
            ll_point[i] = np.concatenate([L[0, :s1], L[1, s1:s2], L[2, s2:]])

    out = {"n_draws_used": int(len(idx))}
    if K >= 1:
        w1 /= w1.sum()
        out["cp1_prob"] = w1
        out["cp1_mode_year"] = int(series.years[int(np.argmax(w1))])
        out["cp1_mean_year"] = float(np.sum(w1[:T] * series.years))
        out["cp1_hdi"] = discrete_hdi(w1[:T], series.years)
        out["cp1_draw_years"] = series.years[np.clip(cps[:, 0], 0, T - 1)]
    if K == 2:
        w2 /= w2.sum()
        m2 = w2.sum(axis=0)
        out["cp1_prob"] = w2.sum(axis=1) / w2.sum()
        out["cp2_prob"] = m2
        out["cp2_mode_year"] = int(series.years[int(np.argmax(m2))])
        out["cp2_mean_year"] = float(np.sum(m2[:T] * series.years))
        out["cp2_hdi"] = discrete_hdi(m2[:T], series.years)
        out["joint"] = w2
        out["cp1_mode_year"] = int(series.years[int(np.argmax(out["cp1_prob"]))])
        out["cp1_mean_year"] = float(np.sum(out["cp1_prob"][:T] * series.years))
        out["cp1_hdi"] = discrete_hdi(out["cp1_prob"][:T], series.years)
        out["cp2_draw_years"] = series.years[np.clip(cps[:, 1], 0, T - 1)]
    out["composition"] = comp
    out["concentration"] = conc
    out["loglik_pointwise"] = ll_point
    return out


def discrete_hdi(w: np.ndarray, years: np.ndarray, prob=0.94) -> list:
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    n = len(w)
    best = None
    for i in range(n):
        c = 0.0
        for j in range(i, n):
            c += w[j]
            if c >= prob:
                if best is None or (j - i) < (best[1] - best[0]):
                    best = (i, j, c)
                break
    if best is None:
        return [int(years[0]), int(years[-1]), 1.0]
    return [int(years[best[0]]), int(years[best[1]]), float(best[2])]


def log_marginal_is(log_post, draws: np.ndarray, n_is=8000, df=4, infl=1.15,
                    seed=17) -> dict:
    rng = np.random.default_rng(seed)
    mean = draws.mean(axis=0)
    cov = np.cov(draws.T) * infl ** 2
    d = len(mean)
    cov = np.atleast_2d(cov) + 1e-8 * np.eye(d)
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(np.diag(np.maximum(np.diag(cov), 1e-8)))
    sign, logdet = np.linalg.slogdet(cov)
    from scipy.special import gammaln
    z = rng.standard_normal((n_is, d))
    g = rng.chisquare(df, n_is)
    x = mean + (z @ chol.T) / np.sqrt(g / df)[:, None]
    dev = x - mean
    sol = np.linalg.solve(chol, dev.T).T
    q = np.sum(sol ** 2, axis=1)
    logq = (gammaln((df + d) / 2) - gammaln(df / 2) - 0.5 * d * np.log(df * np.pi)
            - 0.5 * logdet - 0.5 * (df + d) * np.log1p(q / df))
    lp = np.array([log_post(xi) for xi in x])
    lw = lp - logq
    lw = lw[np.isfinite(lw)]
    logZ = M.log_sum_exp(lw) - np.log(len(lw))
    ess = float(np.exp(2 * M.log_sum_exp(lw) - M.log_sum_exp(2 * lw)))
    return {"log_marginal_likelihood": float(logZ), "is_ess": ess,
            "is_ess_frac": ess / len(lw), "n_is": int(len(lw))}


def main(quick: bool = False):
    t_start = time.time()
    if quick:
        n_draws, n_tune, n_chains, n_keep, n_is = 400, 400, 2, 800, 1500
        rhat_max, ess_min, max_attempts = 99.0, 0, 1
    else:
        n_draws, n_tune, n_chains, n_keep, n_is = 8000, 8000, 4, 6000, 20000
        rhat_max, ess_min, max_attempts = 1.05, 200, 3

    df = D.load_recalls()
    haz = hazard_series(df)
    cha = channel_series(df)
    vio, vio_src = violation_series(df)
    series = [haz, cha, vio]

    hm = pd.DataFrame(haz.counts.astype(int), columns=haz.cats)
    hm.insert(0, "year", haz.years)
    hm["n"] = hm[haz.cats].sum(axis=1)
    for c in haz.cats:
        hm[f"share_{c}"] = hm[c] / hm.n
    hm.to_csv(RES / "hazard_year_matrix.csv", index=False)
    sec = pd.concat([
        pd.DataFrame({"series": s.key, "year": s.years,
                      "k": s.counts[:, 0].astype(int),
                      "n": s.counts.sum(axis=1).astype(int),
                      "share": s.counts[:, 0] / s.counts.sum(axis=1)})
        for s in (cha, vio)])
    sec.to_csv(RES / "secondary_series.csv", index=False)
    print(f"[15] hazard series: {haz.T} years {haz.years.min()}-{haz.years.max()}, "
          f"{haz.K} categories, {int(haz.counts.sum())} recalls")
    print(f"[15] channel series: {cha.T} years; violation series: {vio.T} years "
          f"({vio_src})")

    diags, results, comp_rows = [], {}, []
    for s in series:
        for K in KS:
            print(f"[15] fitting {s.key} K={K} ({(K + 1) * s.K} params)")
            post, log_post, names, info = fit_model(
                s, K, n_draws=n_draws, n_tune=n_tune, n_chains=n_chains,
                seed=31 + 5 * K, rhat_max=rhat_max, ess_min=ess_min,
                max_attempts=max_attempts)
            diags.append(info)
            if not info["reportable"]:
                print(f"    REFUSED {s.key} K={K}: Rhat {info['max_rhat']:.3f}, "
                      f"ESS {info['min_ess']:.0f}")
                continue
            out = process(s, K, post, n_keep=n_keep)
            out["info"] = info
            out["names"] = names
            out.update(log_marginal_is(log_post, post.flat()[
                np.linspace(0, len(post.flat()) - 1, min(4000, len(post.flat()))
                            ).astype(int)], n_is=n_is))
            results[(s.key, K)] = out
            s.fits[K] = out

    for s in series:
        have = [K for K in KS if (s.key, K) in results]
        lls = {f"K={K}": results[(s.key, K)]["loglik_pointwise"] for K in have}
        rows = M.compare(lls)
        best_ll = results[(s.key, int(rows[0]["model"][2:]))]["loglik_pointwise"]
        e_best = (M.log_sum_exp(best_ll, axis=0) - np.log(best_ll.shape[0])
                  - best_ll.var(axis=0, ddof=1))
        z0 = results[(s.key, 0)]["log_marginal_likelihood"] if (s.key, 0) in results else np.nan
        for r in rows:
            K = int(r["model"][2:])
            res = results[(s.key, K)]
            ll = res["loglik_pointwise"]
            e = (M.log_sum_exp(ll, axis=0) - np.log(ll.shape[0])
                 - ll.var(axis=0, ddof=1))
            d = e - e_best
            comp_rows.append({
                "series": s.key, "K": K, "n_params": res["info"]["n_params"],
                "elpd_loo": r["elpd_loo"], "se": r["se"], "d_elpd": r["d_elpd"],
                "d_elpd_se": (0.0 if r["d_elpd"] == 0
                              else float(np.sqrt(len(d) * d.var(ddof=1)))),
                "p_loo": r["p_loo"], "looic": r["looic"],
                "elpd_waic": r["elpd_waic"],
                "khat_max_shared_helper": r["khat_max"],
                "log_marginal_likelihood": res["log_marginal_likelihood"],
                "log_bf_vs_K0": res["log_marginal_likelihood"] - z0,
                "is_ess": res["is_ess"], "is_ess_frac": res["is_ess_frac"],
                "max_rhat": res["info"]["max_rhat"],
                "min_ess": res["info"]["min_ess"],
                "n_obs_years": s.T})
    cmp = pd.DataFrame(comp_rows)
    cmp.to_csv(RES / "model_comparison.csv", index=False)
    pd.DataFrame([{k: v for k, v in d.items() if k != "attempts"}
                  for d in diags]).to_csv(RES / "model_diagnostics.csv", index=False)
    print(cmp[["series", "K", "elpd_loo", "se", "d_elpd", "d_elpd_se", "p_loo",
               "log_bf_vs_K0", "is_ess_frac", "max_rhat"]].round(2)
          .to_string(index=False))

    best_K = {s.key: int(cmp[cmp.series == s.key]
                         .sort_values("elpd_loo", ascending=False).K.iloc[0])
              for s in series}
    print(f"[15] best K by elpd_loo: {best_K}")

    cp_rows = []
    for s in series:
        for K in [k for k in KS if k >= 1 and (s.key, k) in results]:
            res = results[(s.key, K)]
            for which in (1, 2):
                key = f"cp{which}_prob"
                if key not in res:
                    continue
                w = res[key][:s.T]
                for y, p in zip(s.years, w):
                    cp_rows.append({"series": s.key, "K": K, "cp": which,
                                    "year": int(y), "prob": float(p),
                                    "is_best_K": bool(K == best_K[s.key])})
    cp = pd.DataFrame(cp_rows)
    cp.to_csv(RES / "changepoint_posterior.csv", index=False)

    if ("hazard", 2) in results:
        J = results[("hazard", 2)]["joint"][:haz.T, :haz.T]
        jj = pd.DataFrame(J, index=haz.years, columns=haz.years)
        jj.to_csv(RES / "changepoint_joint_k2.csv")

    comp_out, delta_out = [], []
    for s in series:
        for K in [k for k in KS if k >= 1 and (s.key, k) in results]:
            comp = results[(s.key, K)]["composition"]
            conc = results[(s.key, K)]["concentration"]
            for r in range(K + 1):
                for k, c in enumerate(s.cats):
                    lo, hi = hdi1(comp[:, r, k])
                    comp_out.append({"series": s.key, "K": K, "regime": r + 1,
                                     "category": c,
                                     "mean": float(comp[:, r, k].mean()),
                                     "hdi_lo": lo, "hdi_hi": hi,
                                     "is_best_K": bool(K == best_K[s.key])})
                lo, hi = hdi1(conc[:, r])
                comp_out.append({"series": s.key, "K": K, "regime": r + 1,
                                 "category": "_concentration",
                                 "mean": float(conc[:, r].mean()),
                                 "hdi_lo": lo, "hdi_hi": hi,
                                 "is_best_K": bool(K == best_K[s.key])})
            for r in range(K):
                for k, c in enumerate(s.cats):
                    d = comp[:, r + 1, k] - comp[:, r, k]
                    lo, hi = hdi1(d)
                    delta_out.append({"series": s.key, "K": K,
                                      "boundary": f"{r + 1}->{r + 2}",
                                      "category": c,
                                      "from_mean": float(comp[:, r, k].mean()),
                                      "to_mean": float(comp[:, r + 1, k].mean()),
                                      "delta_mean": float(d.mean()),
                                      "hdi_lo": lo, "hdi_hi": hi,
                                      "p_increase": float((d > 0).mean()),
                                      "is_best_K": bool(K == best_K[s.key])})
    pd.DataFrame(comp_out).to_csv(RES / "regime_composition.csv", index=False)
    dl = pd.DataFrame(delta_out)
    dl.to_csv(RES / "composition_change.csv", index=False)

    def cp_block(skey, K):
        res = results.get((skey, K))
        if res is None:
            return None
        out = {"K": K,
               "cp1_mode_year": res["cp1_mode_year"],
               "cp1_mean_year": res["cp1_mean_year"],
               "cp1_hdi94": res["cp1_hdi"],
               "cp1_top_years": top_years(res["cp1_prob"], skey)}
        if K == 2:
            out.update({"cp2_mode_year": res["cp2_mode_year"],
                        "cp2_mean_year": res["cp2_mean_year"],
                        "cp2_hdi94": res["cp2_hdi"],
                        "cp2_top_years": top_years(res["cp2_prob"], skey)})
        return out

    years_of = {s.key: s.years for s in series}

    def top_years(w, skey, n=5):
        yrs = years_of[skey]
        w = np.asarray(w[:len(yrs)], dtype=float)
        o = np.argsort(-w)[:n]
        return [{"year": int(yrs[i]), "prob": float(w[i])} for i in o]

    coincide = {}
    for a, b in [("hazard", "channel"), ("hazard", "violation"),
                 ("channel", "violation")]:
        ka, kb = best_K[a], best_K[b]
        ra, rb = results.get((a, ka)), results.get((b, kb))
        if not ra or not rb or ka < 1 or kb < 1:
            continue
        rngc = np.random.default_rng(99)
        rows = []
        for i in (1, 2):
            ya = ra.get(f"cp{i}_draw_years")
            if ya is None:
                continue
            for j in (1, 2):
                yb = rb.get(f"cp{j}_draw_years")
                if yb is None:
                    continue
                n = min(len(ya), len(yb))
                d = (rngc.permutation(ya)[:n].astype(float)
                     - rngc.permutation(yb)[:n].astype(float))
                lo, hi = hdi1(d)
                rows.append({"cp_a": i, "cp_b": j,
                             "mean_years": float(d.mean()),
                             "hdi_lo": lo, "hdi_hi": hi,
                             "p_within_1_year": float((np.abs(d) <= 1).mean()),
                             "p_within_2_years": float((np.abs(d) <= 2).mean()),
                             "p_a_earlier": float((d < 0).mean())})
        if rows:
            closest = max(rows, key=lambda r: r["p_within_2_years"])
            coincide[f"{a}_vs_{b}"] = {"all_pairs": rows, "closest": closest}

    best = best_K["hazard"]
    hb = results[("hazard", best)]
    big = (dl[(dl.series == "hazard") & (dl.K == best)]
           .reindex(dl[(dl.series == "hazard") & (dl.K == best)]
                    .delta_mean.abs().sort_values(ascending=False).index))

    summary = {
        "analysis": "15_bayes_changepoint",
        "quick_mode": quick,
        "quick_mode_warning": (
            "convergence gating is disabled under --quick (400 draws); no number "
            "from a quick run may be quoted" if quick else None),
        "source": str(D.V2_CSV.relative_to(D.ROOT)),
        "violation_series_source": vio_src,
        "series": {s.key: {"label": s.label, "n_years": s.T,
                           "year_range": [int(s.years.min()), int(s.years.max())],
                           "n_records": int(s.counts.sum()),
                           "categories": s.cats,
                           "reference_category": s.cats[s.ref],
                           "note": s.note} for s in series},
        "method": {
            "likelihood": "Dirichlet-multinomial per year",
            "changepoint_handling": (
                "exact marginalisation over discrete year positions by prefix "
                "sums, uniform prior over admissible positions, minimum segment "
                f"{MIN_SEG} years; the reported posterior is the "
                "Rao-Blackwellised average of the exact conditional weights, and "
                "LOO uses one segmentation drawn from that conditional per "
                "posterior draw (a Gibbs step)."),
            "priors": {"log_concentration": "Normal(log 20, 1.5)",
                       "softmax_coordinates": "Normal(0, 2), reference category "
                                              "pinned at 0"},
            "sampler": ("mcmc.sample at 6 parameters or fewer, "
                        "mcmc.sample_blocked above that (one block per regime "
                        "concentration, blocks of "
                        "4 for composition coordinates, target 0.35)"),
            "rhat_gate": rhat_max, "ess_gate": ess_min,
            "n_draws": n_draws, "n_tune": n_tune, "n_chains": n_chains,
        },
        "model_comparison": cmp.to_dict(orient="records"),
        "khat_note": (
            "khat_max_shared_helper is None for every model because "
            "_common/mcmc._gpd_fit returns NaN (sign bug in the "
            "Zhang-Stephens profile likelihood, documented in build 14); "
            "M.loo_psis therefore reports unsmoothed importance-sampling "
            "LOO. Build 14 quantifies the difference and finds it "
            "negligible on data of this shape."),
        "best_K_by_loo": best_K,
        "models_refused": [f"{d['series']} K={d['K']}" for d in diags
                           if not d["reportable"]],
        "changepoints": {s.key: {f"K{K}": cp_block(s.key, K)
                                 for K in KS if K >= 1 and (s.key, K) in results}
                         for s in series},
        "headline": {
            "series": "hazard",
            "best_K": best,
            "cp1": cp_block("hazard", best),
        },
        "biggest_component_changes_hazard": big.head(6).to_dict(orient="records"),
        "regime_composition_best_K": [
            r for r in comp_out if r["series"] == "hazard" and r["K"] == best],
        "changepoint_coincidence": coincide,
        "limitation": (
            "Recall counts are enforcement outputs, not incidence. A change point "
            "here dates a change in what CPSC detected and published - a testing "
            "programme, a port-targeting rule, a marketplace initiative - and not "
            "necessarily a change in the products on the market. Build 11 shows "
            "the modern series is violation-detected rather than harm-detected, "
            "which makes the surveillance reading the more likely one. The "
            "coincidence of the hazard, channel and detection breaks is "
            "consistent with a single enforcement shift, and the paper should say "
            "so rather than claim a change in product safety."),
        "model_diagnostics": diags,
        "runtime_seconds": time.time() - t_start,
    }
    D.write_json(summary, RES / "changepoint_summary.json")

    for s in series:
        K = best_K[s.key]
        if K == 0:
            print(f"[15] {s.key}: LOO prefers NO change point")
            continue
        b = cp_block(s.key, K)
        line = (f"[15] {s.key}: best K={K}, change point {b['cp1_mode_year']} "
                f"(mean {b['cp1_mean_year']:.1f}, 94% HDI "
                f"{b['cp1_hdi94'][0]}-{b['cp1_hdi94'][1]})")
        if K == 2:
            line += (f"; second {b['cp2_mode_year']} (94% HDI "
                     f"{b['cp2_hdi94'][0]}-{b['cp2_hdi94'][1]})")
        print(line)
    for k, v in coincide.items():
        c = v["closest"]
        print(f"[15] {k}: closest pair cp{c['cp_a']} vs cp{c['cp_b']} "
              f"{c['mean_years']:+.1f} yr [{c['hdi_lo']:.0f}, {c['hdi_hi']:.0f}], "
              f"P(|diff| <= 2 yr) = {c['p_within_2_years']:.2f}")
    print("[15] biggest hazard composition changes: " +
          "; ".join(f"{r['category']} {r['boundary']} "
                    f"{r['from_mean']:.2f}->{r['to_mean']:.2f}"
                    for r in big.head(3).to_dict(orient="records")))
    print(f"[15] total runtime {summary['runtime_seconds'] / 60:.1f} min")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fast settings for a smoke run (<90s)")
    main(**vars(ap.parse_args()))
