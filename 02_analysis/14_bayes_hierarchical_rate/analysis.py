from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import gammaln
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import data as D
from _common import mcmc as M
RES = D.results_dir(__file__)
YEAR_MIN, YEAR_MAX = (1990, 2025)
PERIOD_EDGES = [1989, 1999, 2009, 2019, 2025]
PERIOD_LABELS = ['1990-99', '2000-09', '2010-19', '2020-25']
MIN_BN_SME = 0.5
REFERENCE = 'posterior_grand_mean'
REPORT_COUNTRIES = ['China', 'India', 'Pakistan', 'Indonesia', 'Hong Kong', 'Philippines', 'South Korea', 'Taiwan', 'Thailand', 'Vietnam', 'Bangladesh', 'Cambodia', 'Mexico', 'Peru', 'Italy', 'Romania', 'Portugal', 'Sri Lanka']

def build_panel(drop_years: tuple[int, ...]=()) -> tuple[pd.DataFrame, dict]:
    df = D.load_recalls()
    ex = D.explode_countries(df)
    n_exploded = len(ex)
    n_us = int((ex.country == 'United States').sum())
    ex = ex[~ex.country.isin(D.COUNTRY_NO_DENOMINATOR)]
    n_out_of_window = int((~ex.year.between(YEAR_MIN, YEAR_MAX)).sum())
    ex = ex[ex.year.between(YEAR_MIN, YEAR_MAX)]
    ex = ex[~ex.year.isin(drop_years)]
    imp = D.load_imports()
    n_world_rows = int((imp.country == 'World').sum())
    imp = imp[(imp.country != 'World') & imp.year.between(YEAR_MIN, YEAR_MAX)]
    imp = imp[~imp.year.isin(drop_years)]
    cum = imp.groupby('country').sme.sum() / 1000000000.0
    named = set(ex.country.unique())
    keep = sorted(named | set(cum[cum >= MIN_BN_SME].index))
    unmatched = sorted(named - set(imp.country.unique()))
    counts = ex.groupby(['country', 'year'], as_index=False).recall_id.nunique().rename(columns={'recall_id': 'recalls'})
    p = imp[imp.country.isin(keep)].copy()
    p['bn_sme'] = p.sme / 1000000000.0
    p['bn_usd'] = p.usd / 1000000000.0
    p = p.merge(counts, on=['country', 'year'], how='left')
    p['recalls'] = p.recalls.fillna(0.0)
    p = p[p.bn_sme > 0].sort_values(['country', 'year']).reset_index(drop=True)
    p['period'] = pd.cut(p.year, bins=PERIOD_EDGES, labels=PERIOD_LABELS)
    info = {'n_recall_country_rows_exploded': n_exploded, 'n_rows_united_states_dropped': n_us, 'n_rows_outside_window_dropped': n_out_of_window, 'n_otexa_world_aggregate_rows_dropped': n_world_rows, 'n_countries_modelled': int(p.country.nunique()), 'n_countries_with_recalls': int((p.groupby('country').recalls.sum() > 0).sum()), 'n_panel_rows': int(len(p)), 'n_recalls_modelled': int(p.recalls.sum()), 'share_zero_cells': float((p.recalls == 0).mean()), 'total_bn_sme': float(p.bn_sme.sum()), 'pooled_rate_per_bn_sme': float(p.recalls.sum() / p.bn_sme.sum()), 'country_inclusion_rule': f'>=1 recall OR >= {MIN_BN_SME} bn SME cumulative', 'unmatched_recall_countries': unmatched, 'years': [int(p.year.min()), int(p.year.max())], 'dropped_years': [int(y) for y in drop_years]}
    return (p, info)

class PanelArrays:

    def __init__(self, p: pd.DataFrame):
        self.countries = sorted(p.country.unique())
        self.years = sorted((int(y) for y in p.year.unique()))
        c_idx = {c: i for i, c in enumerate(self.countries)}
        t_idx = {t: i for i, t in enumerate(self.years)}
        self.ci = p.country.map(c_idx).to_numpy(int)
        self.ti = p.year.map(t_idx).to_numpy(int)
        self.y = p.recalls.to_numpy(float)
        self.E = p.bn_sme.to_numpy(float)
        self.logE = np.log(self.E)
        self.C, self.T, self.n = (len(self.countries), len(self.years), len(p))
        self.Y_c = np.bincount(self.ci, weights=self.y, minlength=self.C)
        self.S_c = np.bincount(self.ci, weights=self.E, minlength=self.C)
        self.Y_tot = float(self.y.sum())
        self.g_y1 = gammaln(self.y + 1)
        self.const = float(np.sum(self.y * self.logE) - np.sum(self.g_y1))
        self.uy, self.uy_inv = np.unique(self.y, return_inverse=True)
        self.pi = pd.Categorical(pd.cut(p.year, bins=PERIOD_EDGES, labels=PERIOD_LABELS), categories=PERIOD_LABELS).codes.astype(int)
        self.P = len(PERIOD_LABELS)

    def ll_poisson(self, log_mu: np.ndarray) -> float:
        lm = np.clip(log_mu, -40, 40)
        return float(np.sum(self.y * lm) - np.sum(np.exp(lm)) - np.sum(self.g_y1))

    def ll_nb(self, log_mu: np.ndarray, phi: float) -> float:
        mu = np.exp(np.clip(log_mu, -40, 40))
        lpm = np.log(phi + mu)
        gy = gammaln(self.uy + phi)[self.uy_inv]
        return float(np.sum(gy) - self.n * gammaln(phi) - np.sum(self.g_y1) + phi * (self.n * np.log(phi) - np.sum(lpm)) + np.sum(self.y * (np.log(np.maximum(mu, 1e-300)) - lpm)))

    def pointwise_poisson(self, log_mu: np.ndarray) -> np.ndarray:
        return M.log_poisson(self.y, np.exp(np.clip(log_mu, -40, 40)))

    def pointwise_nb(self, log_mu: np.ndarray, phi: float) -> np.ndarray:
        return M.log_nb(self.y, np.exp(np.clip(log_mu, -40, 40)), phi)

def check_nb(A: PanelArrays) -> float:
    rng = np.random.default_rng(0)
    lm = rng.normal(-1.0, 1.0, A.n)
    worst = 0.0
    for phi in (0.3, 1.0, 7.5):
        a = A.ll_nb(lm, phi)
        b = float(np.sum(M.log_nb(A.y, np.exp(lm), phi)))
        worst = max(worst, abs(a - b))
        assert abs(a - b) < 1e-06 * max(1.0, abs(b)), (phi, a, b)
    return worst

def lp_hyper(mu, ltau):
    return M.log_normal(mu, 0.0, 3.0) + M.log_halfnormal(np.exp(ltau), 1.0) + ltau

def lp_phi(lphi):
    return M.log_normal(lphi, 0.0, 2.0)

def lp_sigma(lsig):
    return M.log_halfnormal(np.exp(lsig), 0.5) + lsig

def rw1_basis(T: int) -> tuple[np.ndarray, np.ndarray]:
    R = np.zeros((T, T))
    for t in range(T - 1):
        R[t, t] += 1.0
        R[t + 1, t + 1] += 1.0
        R[t, t + 1] -= 1.0
        R[t + 1, t] -= 1.0
    lam, U = np.linalg.eigh(R)
    return (lam[1:], U[:, 1:])

class Model:

    def __init__(self, key, label, A, log_post, init, names, blocks, kind):
        self.key, self.label, self.A = (key, label, A)
        self.log_post, self.init, self.names = (log_post, init, names)
        self.blocks, self.kind = (blocks, kind)
        self.post = None
        self.fit_info = {}

    @property
    def n_params(self) -> int:
        return len(self.names)

    def log_mu(self, th):
        raise NotImplementedError

    def phi(self, th) -> float:
        return float('inf')

    def country_lograte(self, th):
        raise NotImplementedError

def make_models(A: PanelArrays) -> 'list[Model]':
    C, T, P = (A.C, A.T, A.P)
    ci, ti, pi, logE = (A.ci, A.ti, A.pi, A.logE)
    a0 = float(np.log(A.Y_tot / A.S_c.sum()))
    th0 = np.clip(np.log((A.Y_c + 0.5) / np.maximum(A.S_c, 1e-06)), a0 - 2.5, a0 + 2.5)
    cnames = [f'theta:{c}' for c in A.countries]
    models = []

    class M1(Model):

        def log_mu(self, th):
            return th[0] + logE

        def country_lograte(self, th):
            return np.full(C, th[0])

    def lp1(th):
        return float(A.ll_poisson(th[0] + logE) + M.log_normal(th[0], 0.0, 3.0))
    models.append(M1('M1', 'Pooled Poisson', A, lp1, np.array([a0]), ['mu'], None, 'poisson'))

    class M2(Model):

        def log_mu(self, th):
            return th[2:][ci] + logE

        def country_lograte(self, th):
            return th[2:]

    def lp2(th):
        mu, ltau, theta = (th[0], th[1], th[2:])
        ll = float(theta @ A.Y_c) - float(np.exp(np.clip(theta, -40, 40)) @ A.S_c) + A.const
        return float(ll + lp_hyper(mu, ltau) + np.sum(M.log_normal(theta, mu, np.exp(ltau))))
    blocks2 = [np.array([0, 1])] + [np.array([2 + c]) for c in range(C)]
    models.append(M2('M2', 'Hierarchical Poisson', A, lp2, np.concatenate([[a0, np.log(0.7)], th0]), ['mu', 'log_tau'] + cnames, blocks2, 'poisson'))

    class M3(Model):

        def log_mu(self, th):
            return th[3:][ci] + logE

        def phi(self, th):
            return float(np.exp(th[2]))

        def country_lograte(self, th):
            return th[3:]

    def lp3(th):
        mu, ltau, lphi, theta = (th[0], th[1], th[2], th[3:])
        ll = A.ll_nb(theta[ci] + logE, float(np.exp(lphi)))
        return float(ll + lp_hyper(mu, ltau) + lp_phi(lphi) + np.sum(M.log_normal(theta, mu, np.exp(ltau))))
    blocks3 = [np.array([0, 1]), np.array([2])] + [np.array([3 + c]) for c in range(C)]
    models.append(M3('M3', 'Hierarchical negative binomial', A, lp3, np.concatenate([[a0, np.log(0.7), 0.0], th0]), ['mu', 'log_tau', 'log_phi'] + cnames, blocks3, 'nb'))
    lam_w, U_w = rw1_basis(T)
    sq_w = np.sqrt(lam_w)
    K0 = 4
    U0 = K0 + (T - 1)

    class M4(Model):

        def log_mu(self, th):
            return th[U0:][ci] + (U_w @ th[K0:U0])[ti] + logE

        def phi(self, th):
            return float(np.exp(th[2]))

        def country_lograte(self, th):
            return th[U0:]

        def year_effects(self, th):
            return U_w @ th[K0:U0]

    def lp4(th):
        mu, ltau, lphi, lsig = (th[0], th[1], th[2], th[3])
        c, theta = (th[K0:U0], th[U0:])
        ll = A.ll_nb(theta[ci] + (U_w @ c)[ti] + logE, float(np.exp(lphi)))
        return float(ll + lp_hyper(mu, ltau) + lp_phi(lphi) + lp_sigma(lsig) + np.sum(M.log_normal(theta, mu, np.exp(ltau))) + np.sum(M.log_normal(c, 0.0, np.exp(lsig) / sq_w)))
    blocks4 = [np.array([0, 1]), np.array([2]), np.array([3])] + [np.array([K0 + k]) for k in range(T - 1)] + [np.array([U0 + c]) for c in range(C)]
    models.append(M4('M4', 'Hierarchical NB + year random walk', A, lp4, np.concatenate([[a0, np.log(0.7), 0.0, np.log(0.15)], np.zeros(T - 1), th0]), ['mu', 'log_tau', 'log_phi', 'log_sigma_w'] + [f'w_mode:{k}' for k in range(T - 1)] + cnames, blocks4, 'nb'))
    G0 = 3
    U5 = G0 + (P - 1)

    def period_effects(g_free):
        return np.concatenate([g_free, [-g_free.sum()]])

    class M5(Model):

        def log_mu(self, th):
            return th[U5:][ci] + period_effects(th[G0:U5])[pi] + logE

        def phi(self, th):
            return float(np.exp(th[2]))

        def country_lograte(self, th):
            return th[U5:]

    def lp5(th):
        mu, ltau, lphi = (th[0], th[1], th[2])
        g_free, theta = (th[G0:U5], th[U5:])
        ll = A.ll_nb(theta[ci] + period_effects(g_free)[pi] + logE, float(np.exp(lphi)))
        return float(ll + lp_hyper(mu, ltau) + lp_phi(lphi) + np.sum(M.log_normal(theta, mu, np.exp(ltau))) + np.sum(M.log_normal(g_free, 0.0, 1.0)))
    blocks5 = [np.array([0, 1]), np.array([2]), np.arange(G0, U5)] + [np.array([U5 + c]) for c in range(C)]
    models.append(M5('M5', 'Hierarchical NB + period effects', A, lp5, np.concatenate([[a0, np.log(0.7), 0.0], np.zeros(P - 1), th0]), ['mu', 'log_tau', 'log_phi'] + [f'g:{p}' for p in PERIOD_LABELS[:-1]] + cnames, blocks5, 'nb'))
    return models

def fit_once(model: Model, *, n_draws, n_tune, n_chains, seed, thin=1):
    if model.blocks is None:
        return M.sample(model.log_post, model.init, n_draws=n_draws, n_tune=n_tune, n_chains=n_chains, seed=seed, names=model.names, thin=thin, verbose=True)
    return M.sample_blocked(model.log_post, model.init, blocks=model.blocks, n_draws=n_draws, n_tune=n_tune, n_chains=n_chains, seed=seed, names=model.names, thin=thin, target=0.4, verbose=True)

def fit_with_retune(model: Model, *, n_draws, n_tune, n_chains, seed, rhat_max, ess_min, max_attempts=3) -> dict:
    attempts, thin, tune = ([], 1, n_tune)
    for k in range(max_attempts):
        t0 = time.time()
        post = fit_once(model, n_draws=n_draws, n_tune=tune, n_chains=n_chains, seed=seed + 7 * k, thin=thin)
        r, e = (post.rhat(), post.ess())
        rec = {'attempt': k + 1, 'n_tune': tune, 'thin': thin, 'max_rhat': float(np.nanmax(r)), 'min_ess': float(np.nanmin(e)), 'worst_rhat_param': model.names[int(np.nanargmax(r))], 'worst_ess_param': model.names[int(np.nanargmin(e))], 'seconds': time.time() - t0, 'accept_rate': [float(a) for a in post.accept_rate]}
        rec['passed'] = bool(rec['max_rhat'] <= rhat_max and rec['min_ess'] >= ess_min)
        attempts.append(rec)
        model.post = post
        print(f"    [{model.key}] attempt {k + 1}: max Rhat {rec['max_rhat']:.3f} ({rec['worst_rhat_param']}), min ESS {rec['min_ess']:.0f} ({rec['worst_ess_param']}), {rec['seconds']:.0f}s -> {('PASS' if rec['passed'] else 'RETUNE')}")
        if rec['passed']:
            break
        tune, thin = (int(tune * 2), thin * 2)
    info = {'model': model.key, 'label': model.label, 'n_params': model.n_params, 'n_chains': n_chains, 'n_draws': n_draws, 'attempts': attempts, 'max_rhat': attempts[-1]['max_rhat'], 'min_ess': attempts[-1]['min_ess'], 'mean_accept_rate': float(np.mean(attempts[-1]['accept_rate'])), 'seconds_total': float(sum((a['seconds'] for a in attempts))), 'reportable': attempts[-1]['passed'], 'rhat_gate': rhat_max, 'ess_gate': ess_min, 'sampler': 'sample_blocked' if model.blocks is not None else 'sample', 'n_blocks': len(model.blocks) if model.blocks is not None else 1}
    model.fit_info = info
    return info

def pointwise(model: Model, n_ll: int, rng) -> np.ndarray:
    f = model.post.flat()
    idx = rng.choice(len(f), size=min(n_ll, len(f)), replace=False)
    out = np.empty((len(idx), model.A.n))
    for i, s in enumerate(idx):
        th = f[s]
        lm = model.log_mu(th)
        out[i] = model.A.pointwise_poisson(lm) if model.kind == 'poisson' else model.A.pointwise_nb(lm, model.phi(th))
    return out

def elpd_pointwise(ll: np.ndarray) -> np.ndarray:
    lppd = M.log_sum_exp(ll, axis=0) - np.log(ll.shape[0])
    return lppd - ll.var(axis=0, ddof=1)

def gpd_fit(x: np.ndarray) -> tuple[float, float]:
    x = np.sort(np.asarray(x, dtype=float))
    x = x[x > 0]
    n = len(x)
    if n < 5:
        return (np.nan, np.nan)
    m = 30 + int(np.sqrt(n))
    prior = 3.0 / x[int(n / 4 + 0.5) - 1]
    theta = 1.0 / x[-1] + (1 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))) * prior
    with np.errstate(all='ignore'):
        k_t = -np.mean(np.log1p(-np.outer(theta, x)), axis=1)
        k_safe = np.where(np.abs(k_t) < 1e-12, np.copysign(1e-12, theta), k_t)
        ell = n * (np.log(theta / k_safe) + k_safe - 1.0)
        bad = ~np.isfinite(ell)
        if bad.all():
            return (np.nan, np.nan)
        ell = np.where(bad, ell[~bad].min() - 50.0, ell)
        w = 1.0 / np.sum(np.exp(ell[None, :] - ell[:, None]), axis=1)
        theta_hat = float(np.sum(theta * w))
        xi = float(np.mean(np.log1p(-theta_hat * x)))
        sigma = -xi / theta_hat if theta_hat != 0 else np.nan
    return (xi, sigma)

def psis_loo(loglik: np.ndarray, tail_frac=0.2) -> dict:
    S, N = loglik.shape
    m = max(int(min(tail_frac * S, 3 * np.sqrt(S))), 10)
    elpd, khat = (np.empty(N), np.empty(N))
    for i in range(N):
        lw = -loglik[:, i]
        lw = lw - M.log_sum_exp(lw)
        order = np.argsort(lw)
        tail = order[-m:]
        x = np.exp(lw[tail])
        u = np.exp(lw[order[-m - 1]]) if S > m else x.min()
        xi, sigma = gpd_fit(np.sort(x) - u)
        khat[i] = xi
        if np.isfinite(xi) and xi < 1 and np.isfinite(sigma) and (sigma > 0):
            q = (np.arange(m) + 0.5) / m
            smoothed = u + (sigma * ((1 - q) ** (-xi) - 1) / xi if abs(xi) > 1e-08 else -sigma * np.log1p(-q))
            lw = lw.copy()
            lw[tail[np.argsort(lw[tail])]] = np.log(np.maximum(smoothed, 1e-300))
            lw = lw - M.log_sum_exp(lw)
        elpd[i] = M.log_sum_exp(lw + loglik[:, i])
    lppd = M.log_sum_exp(loglik, axis=0) - np.log(S)
    return {'elpd_loo_psis': float(elpd.sum()), 'p_loo_psis': float((lppd - elpd).sum()), 'se_psis': float(np.sqrt(N * elpd.var(ddof=1))), 'khat_max': float(np.nanmax(khat)), 'khat_median': float(np.nanmedian(khat)), 'khat_gt_0p5': int(np.sum(khat > 0.5)), 'khat_gt_0p7': int(np.sum(khat > 0.7)), 'khat_nan': int(np.sum(~np.isfinite(khat)))}

def hdi1(x, prob=0.94) -> tuple[float, float]:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    k = max(int(np.floor(prob * n)), 1)
    if k >= n:
        return (float(x[0]), float(x[-1]))
    w = x[k:] - x[:n - k]
    i = int(np.argmin(w))
    return (float(x[i]), float(x[i + k]))

def summarise(x, prob=0.94) -> dict:
    lo, hi = hdi1(x, prob)
    return {'mean': float(np.mean(x)), 'sd': float(np.std(x, ddof=1)), 'median': float(np.median(x)), 'hdi_lo': lo, 'hdi_hi': hi}

def country_rate_table(model: Model, panel: pd.DataFrame, A: PanelArrays, grand: float, max_draws=6000, seed=3):
    f = model.post.flat()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(f), size=max_draws, replace=False) if len(f) > max_draws else np.arange(len(f))
    lr = np.array([model.country_lograte(f[s]) for s in idx])
    rates = np.exp(lr)
    mu_draws = np.exp(model.post.get('mu'))[idx]
    rr = rates / mu_draws[:, None]
    obs = panel.groupby('country').agg(recalls=('recalls', 'sum'), bn_sme=('bn_sme', 'sum'), bn_usd=('bn_usd', 'sum'), n_years=('year', 'nunique'))
    rows = []
    for i, c in enumerate(A.countries):
        r, rrc = (rates[:, i], rr[:, i])
        lo, hi = hdi1(r)
        rlo, rhi = hdi1(rrc)
        raw = float(obs.loc[c, 'recalls'] / obs.loc[c, 'bn_sme'])
        pm = float(r.mean())
        dev = np.log(raw / grand) if raw > 0 else np.nan
        shrink = 100 * (1 - np.log(pm / grand) / dev) if np.isfinite(dev) and abs(dev) > 0.05 else np.nan
        rows.append({'country': c, 'recalls': int(obs.loc[c, 'recalls']), 'bn_sme': float(obs.loc[c, 'bn_sme']), 'bn_usd': float(obs.loc[c, 'bn_usd']), 'raw_rate': raw, 'post_rate_mean': pm, 'post_rate_median': float(np.median(r)), 'post_rate_sd': float(r.std(ddof=1)), 'hdi_lo': lo, 'hdi_hi': hi, 'shrinkage_pct_log_scale': float(shrink) if shrink == shrink else None, 'rr_vs_reference_mean': float(rrc.mean()), 'rr_vs_reference_median': float(np.median(rrc)), 'rr_hdi_lo': rlo, 'rr_hdi_hi': rhi, 'p_rate_below_reference': float((rrc < 1).mean())})
    tab = pd.DataFrame(rows).sort_values('post_rate_mean', ascending=False)
    return (tab, rr)

def posterior_predictive(model: Model, A: PanelArrays, n_rep=1000, seed=5):
    f = model.post.flat()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(f), size=min(n_rep, len(f)), replace=False)
    big = A.countries.index('China')

    def stats(v):
        vc = np.bincount(A.ci, weights=v, minlength=A.C)
        return {'total_recalls': float(v.sum()), 'share_zero_cells': float((v == 0).mean()), 'max_cell_count': float(v.max()), 'var_mean_ratio': float(v.var(ddof=1) / max(v.mean(), 1e-12)), 'n_countries_with_recalls': float((vc > 0).sum()), 'china_total': float(vc[big]), 'share_recalls_largest_country': float(vc.max() / max(vc.sum(), 1e-12))}
    obs = stats(A.y)
    reps = {k: [] for k in obs}
    country_rep = np.empty((len(idx), A.C))
    for i, s in enumerate(idx):
        th = f[s]
        mu = np.exp(np.clip(model.log_mu(th), -40, 40))
        if model.kind == 'poisson':
            yr = rng.poisson(mu).astype(float)
        else:
            phi = model.phi(th)
            yr = rng.poisson(rng.gamma(phi, mu / phi)).astype(float)
        for k, v in stats(yr).items():
            reps[k].append(v)
        country_rep[i] = np.bincount(A.ci, weights=yr, minlength=A.C)
    rows = []
    for k, v in obs.items():
        arr = np.asarray(reps[k], dtype=float)
        lo, hi = hdi1(arr)
        rows.append({'statistic': k, 'observed': v, 'rep_mean': float(arr.mean()), 'rep_sd': float(arr.std(ddof=1)), 'rep_hdi_lo': lo, 'rep_hdi_hi': hi, 'bayes_p': float((arr >= v).mean()), 'inside_hdi': bool(lo <= v <= hi)})
    ppc = pd.DataFrame(rows)
    obs_c = np.bincount(A.ci, weights=A.y, minlength=A.C)
    crows = []
    for i, c in enumerate(A.countries):
        lo, hi = hdi1(country_rep[:, i])
        crows.append({'country': c, 'observed': float(obs_c[i]), 'rep_mean': float(country_rep[:, i].mean()), 'rep_lo': lo, 'rep_hi': hi, 'inside': bool(lo <= obs_c[i] <= hi)})
    ppc_c = pd.DataFrame(crows).sort_values('observed', ascending=False)
    dist = pd.DataFrame({k: reps[k] for k in ['share_zero_cells', 'var_mean_ratio', 'max_cell_count']})
    return (ppc, ppc_c, dist, {'n_rep': int(len(idx)), 'country_coverage_94': float(ppc_c.inside.mean())})

def main(quick: bool=False):
    t_start = time.time()
    if quick:
        n_draws, n_tune, n_chains, n_ll, n_rep = (250, 250, 2, 150, 200)
        rhat_max, ess_min, max_attempts = (99.0, 0, 1)
    else:
        n_draws, n_tune, n_chains, n_ll, n_rep = (8000, 8000, 4, 1500, 1500)
        rhat_max, ess_min, max_attempts = (1.05, 200, 3)
    panel, panel_info = build_panel()
    A = PanelArrays(panel)
    nb_err = check_nb(A)
    panel.to_csv(RES / 'panel_country_year.csv', index=False)
    print(f"[14] panel: {A.n} country-year cells, {A.C} countries, {int(A.Y_tot)} recalls, {A.S_c.sum():.0f} bn SME, {panel_info['share_zero_cells']:.1%} zero cells")
    suspect = D.suspect_import_years(D.load_imports())
    print(f'[14] suspect OTEXA years (world total duplicates previous): {suspect}')
    models, diags = ({}, [])
    for m in make_models(A):
        print(f'[14] fitting {m.key} ({m.label}, {m.n_params} params)')
        diags.append(fit_with_retune(m, n_draws=n_draws, n_tune=n_tune, n_chains=n_chains, seed=101, rhat_max=rhat_max, ess_min=ess_min, max_attempts=max_attempts))
        models[m.key] = m
    pd.DataFrame([{k: v for k, v in d.items() if k != 'attempts'} for d in diags]).to_csv(RES / 'model_diagnostics.csv', index=False)
    rng = np.random.default_rng(11)
    lls = {k: pointwise(m, n_ll, rng) for k, m in models.items() if m.fit_info['reportable']}
    refused = [k for k, m in models.items() if not m.fit_info['reportable']]
    if refused:
        print(f'[14] REFUSED (did not converge): {refused}')
    cmp_rows = M.compare(lls)
    best_key = cmp_rows[0]['model']
    e_best = elpd_pointwise(lls[best_key])
    for r in cmp_rows:
        m = models[r['model']]
        e = elpd_pointwise(lls[r['model']])
        d = e - e_best
        r.update({'label': m.label, 'n_params': m.n_params, 'd_elpd_se': float(np.sqrt(len(d) * d.var(ddof=1))) if r['model'] != best_key else 0.0, 'max_rhat': m.fit_info['max_rhat'], 'min_ess': m.fit_info['min_ess'], 'khat_max_shared_helper': r.pop('khat_max'), 'khat_bad_shared_helper': r.pop('khat_bad')})
        r.update(psis_loo(lls[r['model']]))
    cmp = pd.DataFrame(cmp_rows)[['model', 'label', 'n_params', 'elpd_loo', 'se', 'd_elpd', 'd_elpd_se', 'p_loo', 'looic', 'elpd_waic', 'elpd_loo_psis', 'p_loo_psis', 'khat_max', 'khat_median', 'khat_gt_0p5', 'khat_gt_0p7', 'khat_max_shared_helper', 'max_rhat', 'min_ess']]
    cmp.to_csv(RES / 'model_comparison.csv', index=False)
    print(cmp.round(2).to_string(index=False))
    best = models[best_key]
    print(f'[14] best model by elpd_loo: {best_key} ({best.label})')
    grand = float(np.exp(best.post.get('mu')).mean())
    rates, rr = country_rate_table(best, panel, A, grand)
    rates.to_csv(RES / 'country_rates.csv', index=False)
    keep = [c for c in REPORT_COUNTRIES if c in A.countries]
    sub = pd.DataFrame({c: rr[:, A.countries.index(c)] for c in keep})
    sub.sample(min(4000, len(sub)), random_state=0).to_csv(RES / 'rate_ratio_draws.csv', index=False)
    hyper = {'grand_mean_rate_per_bn_sme': summarise(np.exp(best.post.get('mu')))}
    if 'log_tau' in best.names:
        hyper['tau_country_sd_of_log_rate'] = summarise(np.exp(best.post.get('log_tau')))
    if 'log_phi' in best.names:
        hyper['phi_nb_dispersion'] = summarise(np.exp(best.post.get('log_phi')))
    if 'log_sigma_w' in best.names:
        hyper['sigma_year_walk'] = summarise(np.exp(best.post.get('log_sigma_w')))
    year_eff = None
    if hasattr(best, 'year_effects'):
        f = best.post.flat()
        step = max(len(f) // 4000, 1)
        W = np.exp(np.array([best.year_effects(th) for th in f[::step]]))
        year_eff = pd.DataFrame([dict(year=int(yy), **{k: v for k, v in summarise(W[:, i]).items() if k in ('mean', 'hdi_lo', 'hdi_hi')}) for i, yy in enumerate(A.years)])
        year_eff.to_csv(RES / 'year_effects.csv', index=False)
    period_eff = None
    if models['M5'].fit_info['reportable']:
        p5 = models['M5'].post
        gf = np.column_stack([p5.get(f'g:{p}') for p in PERIOD_LABELS[:-1]])
        g = np.exp(np.column_stack([gf, -gf.sum(axis=1)]))
        period_eff = pd.DataFrame([dict(period=lab, **{k: v for k, v in summarise(g[:, i]).items() if k in ('mean', 'hdi_lo', 'hdi_hi')}) for i, lab in enumerate(PERIOD_LABELS)])
        period_eff.to_csv(RES / 'period_effects.csv', index=False)
    ppc, ppc_c, ppc_dist, ppc_info = posterior_predictive(best, A, n_rep=n_rep)
    ppc.to_csv(RES / 'ppc_stats.csv', index=False)
    ppc_c.to_csv(RES / 'ppc_country.csv', index=False)
    ppc_dist.to_csv(RES / 'ppc_replicates.csv', index=False)
    sens_panel, sens_info = build_panel(drop_years=tuple(suspect))
    A2 = PanelArrays(sens_panel)
    m2 = {m.key: m for m in make_models(A2)}[best_key]
    print(f'[14] sensitivity refit of {best_key} without OTEXA year(s) {suspect}')
    sens_fit = fit_with_retune(m2, n_draws=n_draws, n_tune=n_tune, n_chains=n_chains, seed=202, rhat_max=rhat_max, ess_min=ess_min, max_attempts=max_attempts)
    grand2 = float(np.exp(m2.post.get('mu')).mean())
    sens_rates, _ = country_rate_table(m2, sens_panel, A2, grand2)
    j = sens_rates.merge(rates[['country', 'post_rate_mean', 'rr_vs_reference_mean', 'p_rate_below_reference']], on='country', suffixes=('_no_suspect', '_main'))
    j['pct_change_rate'] = 100 * (j.post_rate_mean_no_suspect / j.post_rate_mean_main - 1)
    j.to_csv(RES / 'sensitivity_no2024.csv', index=False)
    sens_summary = {'dropped_years': suspect, 'n_panel_rows': int(len(sens_panel)), 'n_recalls': int(sens_panel.recalls.sum()), 'grand_mean_rate': grand2, 'max_abs_pct_change_in_country_rate': float(j.pct_change_rate.abs().max()), 'median_abs_pct_change': float(j.pct_change_rate.abs().median()), 'spearman_with_main': float(j.post_rate_mean_no_suspect.corr(j.post_rate_mean_main, method='spearman')), 'converged': sens_fit['reportable'], 'max_rhat': sens_fit['max_rhat'], 'min_ess': sens_fit['min_ess'], 'seconds': sens_fit['seconds_total'], 'china_rate': float(j.loc[j.country == 'China', 'post_rate_mean_no_suspect'].iloc[0]), 'biggest_movers': j.reindex(j.pct_change_rate.abs().sort_values(ascending=False).index).head(6)[['country', 'post_rate_mean_main', 'post_rate_mean_no_suspect', 'pct_change_rate']].to_dict(orient='records')}
    idx_tab = rates.set_index('country')
    small = idx_tab[(idx_tab.bn_sme < 3.0) & (idx_tab.recalls >= 1)].sort_values('raw_rate', ascending=False)
    shrink = {'grand_mean_rate_per_bn_sme': grand, 'n_countries': int(len(idx_tab)), 'raw_rate_range': [float(idx_tab.raw_rate.min()), float(idx_tab.raw_rate.max())], 'posterior_rate_range': [float(idx_tab.post_rate_mean.min()), float(idx_tab.post_rate_mean.max())], 'raw_spread_over_posterior_spread': float((idx_tab.raw_rate.max() - idx_tab.raw_rate.min()) / (idx_tab.post_rate_mean.max() - idx_tab.post_rate_mean.min())), 'small_supplier_examples': small.reset_index()[['country', 'recalls', 'bn_sme', 'raw_rate', 'post_rate_mean', 'hdi_lo', 'hdi_hi', 'shrinkage_pct_log_scale', 'p_rate_below_reference']].head(12).to_dict(orient='records'), 'reference': 'posterior grand mean exp(mu), per draw'}
    summary = {'analysis': '14_bayes_hierarchical_rate', 'quick_mode': quick, 'quick_mode_warning': 'convergence gating is disabled under --quick (250 draws); no number from a quick run may be quoted' if quick else None, 'source': str(D.V2_CSV.relative_to(D.ROOT)), 'exposure_source': str(D.OTEXA_SME.relative_to(D.ROOT)), 'panel': panel_info, 'nb_likelihood_max_abs_error_vs_shared_log_nb': nb_err, 'suspect_import_years': suspect, 'suspect_year_note': 'The OTEXA export repeats the 2023 column as 2024: the world totals differ by 984,315 SME out of 25.7 bn (0.004%), and the two columns are identical for 149 of 240 reporting countries. The main panel keeps 2024 (treating it as a carried-forward value); results/sensitivity_no2024.csv refits the best model with the year removed from both numerator and denominator.', 'sampler': {'why_blocked': '_common/mcmc.sample is a random-walk Metropolis and mixes badly above ~10 parameters (Rhat 9.1 at d=54); every model with country effects uses sample_blocked - hierarchical mean+scale in one block, dispersion and walk scale in their own, each country effect and each random-walk mode in a single-parameter block, acceptance target 0.40.', 'reparameterisations': '(1) country effects carried as absolute log rates theta_c ~ N(mu, tau), not as offsets u_c from a global intercept: the offset form puts mu and the 58 offsets on a likelihood ridge and returned Rhat 1.9-2.2 with min ESS ~15. (2) the year walk carried in the spectral (Demmler-Reinsch) basis of the RW1 structure matrix, not as increments: increments gave Rhat 1.27 / ESS 10 on a 1500-draw pilot, the spectral basis Rhat 1.01 / ESS 89. Both are exact reparameterisations - same posterior.', 'rhat_gate': rhat_max, 'ess_gate': ess_min, 'max_attempts': max_attempts, 'n_draws': n_draws, 'n_tune': n_tune, 'n_chains': n_chains, 'n_loglik_draws_for_loo': n_ll, 'n_posterior_predictive_draws': n_rep}, 'priors': {'mu': 'Normal(0, 3) on the log rate per bn SME', 'u_c': 'Normal(0, tau)', 'tau': 'HalfNormal(1), sampled as log tau with Jacobian', 'phi': 'LogNormal(0, 2) (NB dispersion; phi -> inf is Poisson)', 'sigma_w': 'HalfNormal(0.5), sampled as log sigma with Jacobian', 'g_p': 'Normal(0, 1), sum-to-zero over the four periods'}, 'models': {k: {'label': m.label, 'n_params': m.n_params} for k, m in models.items()}, 'model_diagnostics': diags, 'models_refused_for_nonconvergence': refused, 'model_comparison': cmp.to_dict(orient='records'), 'best_model': best_key, 'best_model_label': best.label, 'hyperparameters': hyper, 'country_rates': rates.to_dict(orient='records'), 'shrinkage': shrink, 'year_effects': year_eff.to_dict(orient='records') if year_eff is not None else None, 'period_effects': period_eff.to_dict(orient='records') if period_eff is not None else None, 'ppc': ppc.to_dict(orient='records'), 'ppc_country_coverage': ppc_info, 'sensitivity_no_suspect_year': sens_summary, 'sensitivity_panel': sens_info, 'runtime_seconds': time.time() - t_start}
    D.write_json(summary, RES / 'bayes_rate_summary.json')
    print(f"[14] grand-mean rate {grand:.3f} per bn SME; tau {hyper.get('tau_country_sd_of_log_rate', {}).get('mean', float('nan')):.2f}")
    show = rates[rates.country.isin(['China', 'India', 'Pakistan', 'Indonesia', 'Vietnam', 'Bangladesh', 'Peru', 'Italy', 'Romania', 'Sweden'])]
    print(show[['country', 'recalls', 'bn_sme', 'raw_rate', 'post_rate_mean', 'hdi_lo', 'hdi_hi', 'rr_vs_reference_mean', 'p_rate_below_reference']].round(3).to_string(index=False))
    print(f"[14] PPC country coverage at 94%: {ppc_info['country_coverage_94']:.2f}")
    print(f"[14] sensitivity: max |change| in a country rate {sens_summary['max_abs_pct_change_in_country_rate']:.1f}%, Spearman {sens_summary['spearman_with_main']:.3f}")
    print(f"[14] total runtime {summary['runtime_seconds'] / 60:.1f} min")
    return summary
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='fast settings for a smoke run (<90s)')
    main(**vars(ap.parse_args()))
