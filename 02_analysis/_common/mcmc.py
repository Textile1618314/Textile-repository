from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.special import gammaln


@dataclass
class Posterior:
    draws: np.ndarray
    names: list[str]
    accept_rate: np.ndarray
    seconds: float
    extra: dict = field(default_factory=dict)

    @property
    def n_chains(self) -> int:
        return self.draws.shape[0]

    @property
    def n_draws(self) -> int:
        return self.draws.shape[1]

    def flat(self) -> np.ndarray:
        return self.draws.reshape(-1, self.draws.shape[2])

    def get(self, name: str) -> np.ndarray:
        return self.flat()[:, self.names.index(name)]

    def block(self, prefix: str) -> np.ndarray:
        idx = [i for i, n in enumerate(self.names) if n.startswith(prefix)]
        return self.flat()[:, idx]

    def rhat(self) -> np.ndarray:
        d = self.draws
        m, n, p = d.shape
        if n < 4:
            return np.full(p, np.nan)
        half = n // 2
        s = np.concatenate([d[:, :half, :], d[:, half:2 * half, :]], axis=0)
        M, N = s.shape[0], s.shape[1]
        chain_mean = s.mean(axis=1)
        chain_var = s.var(axis=1, ddof=1)
        W = chain_var.mean(axis=0)
        B = N * chain_mean.var(axis=0, ddof=1)
        var_hat = (N - 1) / N * W + B / N
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sqrt(np.where(W > 0, var_hat / W, np.nan))

    def ess(self) -> np.ndarray:
        d = self.draws
        m, n, p = d.shape
        out = np.empty(p)
        for j in range(p):
            x = d[:, :, j]
            W = x.var(axis=1, ddof=1).mean()
            if not np.isfinite(W) or W <= 0:
                out[j] = np.nan
                continue
            acov = np.zeros(n)
            for c in range(m):
                y = x[c] - x[c].mean()
                f = np.fft.rfft(y, 2 * n)
                a = np.fft.irfft(f * np.conjugate(f), 2 * n)[:n]
                acov += a / n
            acov /= m
            if acov[0] <= 0:
                out[j] = np.nan
                continue
            rho = acov / acov[0]
            t, s = 1, 0.0
            while t + 1 < n:
                pair = rho[t] + rho[t + 1]
                if pair <= 0:
                    break
                s += pair
                t += 2
            tau = 1 + 2 * s
            out[j] = m * n / max(tau, 1e-8)
        return out

    def hdi(self, prob: float = 0.94) -> np.ndarray:
        f = self.flat()
        n = f.shape[0]
        k = max(int(np.floor(prob * n)), 1)
        out = np.empty((f.shape[1], 2))
        for j in range(f.shape[1]):
            x = np.sort(f[:, j])
            widths = x[k:] - x[:n - k]
            i = int(np.argmin(widths))
            out[j] = (x[i], x[i + k])
        return out

    def summary(self, transform=None) -> "list[dict]":
        f = self.flat()
        h = self.hdi()
        r, e = self.rhat(), self.ess()
        rows = []
        for j, name in enumerate(self.names):
            x = f[:, j]
            lo, hi = h[j]
            row = {"param": name, "mean": float(x.mean()),
                   "sd": float(x.std(ddof=1)),
                   "hdi_3%": float(lo), "hdi_97%": float(hi),
                   "rhat": float(r[j]), "ess": float(e[j])}
            if transform is not None:
                row.update({f"t_{k}": v for k, v in
                            transform(name, x, lo, hi).items()})
            rows.append(row)
        return rows

    def converged(self, rhat_max=1.01, ess_min=400) -> bool:
        r, e = self.rhat(), self.ess()
        return bool(np.nanmax(r) <= rhat_max and np.nanmin(e) >= ess_min)


def _adapt_chain(log_post, x0, n_draws, n_tune, rng, target=0.234,
                 thin=1, progress=None):
    p = len(x0)
    x = np.asarray(x0, dtype=float).copy()
    lp = log_post(x)
    if not np.isfinite(lp):
        raise ValueError("initial point has non-finite log-posterior")

    log_scale = np.log(2.38 / np.sqrt(p))
    mu = x.copy()
    cov = np.eye(p) * 0.01
    chol = np.linalg.cholesky(cov)
    eps = 1e-10

    total = n_tune + n_draws * thin
    kept = np.empty((n_draws, p))
    n_acc = 0
    k = 0
    for i in range(total):
        prop = x + np.exp(log_scale) * (chol @ rng.standard_normal(p))
        lp_prop = log_post(prop)
        a = lp_prop - lp
        accepted = np.log(rng.random() + 1e-300) < a
        if accepted:
            x, lp = prop, lp_prop
            if i >= n_tune:
                n_acc += 1
        elif i >= n_tune:
            pass

        if i < n_tune:
            g = 1.0 / (i // 50 + 2) ** 0.7
            log_scale += g * (min(np.exp(min(a, 0.0)), 1.0) - target)
            d = x - mu
            mu = mu + g * d
            cov = cov + g * (np.outer(d, d) - cov)
            if i % 100 == 0 and i > 200:
                try:
                    chol = np.linalg.cholesky(cov + eps * np.eye(p))
                except np.linalg.LinAlgError:
                    chol = np.linalg.cholesky(np.diag(np.diag(cov)) +
                                              eps * np.eye(p))
        else:
            if (i - n_tune) % thin == 0 and k < n_draws:
                kept[k] = x
                k += 1
        if progress is not None and i % max(total // 20, 1) == 0:
            progress(i / total)
    return kept, n_acc / max(n_draws * thin, 1)


def sample(log_post, init, *, n_draws=4000, n_tune=4000, n_chains=4, seed=0,
           names=None, thin=1, jitter=0.25, verbose=True) -> Posterior:
    init = np.atleast_2d(np.asarray(init, dtype=float))
    p = init.shape[1]
    if init.shape[0] == 1:
        init = np.repeat(init, n_chains, axis=0)
    names = list(names) if names is not None else [f"p{i}" for i in range(p)]
    assert len(names) == p, f"{len(names)} names for {p} params"

    t0 = time.time()
    draws = np.empty((n_chains, n_draws, p))
    acc = np.empty(n_chains)
    for c in range(n_chains):
        rng = np.random.default_rng(seed + 1000 * c)
        x0 = init[c] + jitter * rng.standard_normal(p)
        tries = 0
        while not np.isfinite(log_post(x0)) and tries < 50:
            x0 = init[c] + jitter * rng.standard_normal(p)
            tries += 1
        cb = None
        if verbose:
            def cb(frac, c=c):
                print(f"\r    chain {c + 1}/{n_chains}  {frac:5.0%}",
                      end="", flush=True)
        draws[c], acc[c] = _adapt_chain(log_post, x0, n_draws, n_tune, rng,
                                        thin=thin, progress=cb)
        if verbose:
            print(f"\r    chain {c + 1}/{n_chains}  done  "
                  f"accept={acc[c]:.2f}", flush=True)
    return Posterior(draws, names, acc, time.time() - t0)


def log_sum_exp(a, axis=None):
    a = np.asarray(a, dtype=float)
    if axis is None:
        m = np.max(a)
        return float(m + np.log(np.sum(np.exp(a - m))))
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def waic(loglik: np.ndarray) -> dict:
    lppd = log_sum_exp(loglik, axis=0) - np.log(loglik.shape[0])
    p_waic = loglik.var(axis=0, ddof=1)
    elpd = lppd - p_waic
    return {"elpd_waic": float(elpd.sum()),
            "p_waic": float(p_waic.sum()),
            "waic": float(-2 * elpd.sum()),
            "se": float(np.sqrt(len(elpd) * elpd.var(ddof=1)))}


def loo_psis(loglik: np.ndarray, tail_frac=0.2) -> dict:
    S, N = loglik.shape
    M = max(int(min(tail_frac * S, 3 * np.sqrt(S))), 10)
    elpd = np.empty(N)
    khat = np.empty(N)
    for i in range(N):
        lw = -loglik[:, i]
        lw = lw - log_sum_exp(lw)
        order = np.argsort(lw)
        tail_idx = order[-M:]
        x = np.exp(lw[tail_idx])
        u = np.exp(lw[order[-M - 1]]) if S > M else x.min()
        k, sigma = _gpd_fit(np.sort(x) - u)
        khat[i] = k
        if np.isfinite(k) and k < 1:
            q = (np.arange(M) + 0.5) / M
            smoothed = u + _gpd_icdf(q, k, sigma)
            lw_new = lw.copy()
            lw_new[tail_idx[np.argsort(lw[tail_idx])]] = np.log(
                np.maximum(smoothed, 1e-300))
            lw = lw_new - log_sum_exp(lw_new)
        elpd[i] = log_sum_exp(lw + loglik[:, i])
    lppd = log_sum_exp(loglik, axis=0) - np.log(S)
    p_loo = float((lppd - elpd).sum())
    return {"elpd_loo": float(elpd.sum()), "p_loo": p_loo,
            "looic": float(-2 * elpd.sum()),
            "se": float(np.sqrt(N * elpd.var(ddof=1))),
            "khat_max": (float(np.nanmax(khat)) if np.any(np.isfinite(khat)) else None),
            "khat_bad": int(np.sum(np.nan_to_num(khat, nan=0.0) > 0.7))}


def _gpd_fit(x):
    x = np.asarray(x, dtype=float)
    x = np.sort(x[np.isfinite(x) & (x > 0)])
    n = len(x)
    if n < 5:
        return np.nan, np.nan
    m = 30 + int(np.sqrt(n))
    x_star = x[max(int(n / 4 + 0.5) - 1, 0)]
    prior = 1.0 / (3.0 * x_star) if x_star > 0 else 1.0
    j = np.arange(1, m + 1)
    theta = 1.0 / x[-1] + (1.0 - np.sqrt(m / (j - 0.5))) * prior
    with np.errstate(all="ignore"):
        k_t = -np.mean(np.log1p(-np.outer(theta, x)), axis=1)
        ok = np.isfinite(k_t) & (k_t != 0) & np.isfinite(theta) & (theta != 0)
        if not ok.any():
            return np.nan, np.nan
        theta, k_t = theta[ok], k_t[ok]
        ratio = theta / k_t
        ok2 = ratio > 0
        if not ok2.any():
            return np.nan, np.nan
        theta, k_t, ratio = theta[ok2], k_t[ok2], ratio[ok2]
        ell = n * (np.log(ratio) + k_t - 1.0)
        ell = np.where(np.isfinite(ell), ell, -np.inf)
        if not np.isfinite(ell).any():
            return np.nan, np.nan
        w = np.exp(ell - log_sum_exp(ell))
        theta_hat = float(np.sum(theta * w))
        k = float(-np.mean(np.log1p(-theta_hat * x)))
        sigma = k / theta_hat if theta_hat != 0 else np.nan
    if not (np.isfinite(k) and np.isfinite(sigma)):
        return np.nan, np.nan
    return -k, float(sigma)


def _gpd_icdf(q, k, sigma):
    if not np.isfinite(k) or not np.isfinite(sigma):
        return np.full_like(q, np.nan)
    if abs(k) < 1e-8:
        return -sigma * np.log1p(-q)
    return sigma * ((1 - q) ** (-k) - 1) / k


def compare(models: dict) -> list[dict]:
    rows = []
    for name, ll in models.items():
        r = loo_psis(ll)
        r["model"] = name
        r.update({"elpd_waic": waic(ll)["elpd_waic"]})
        rows.append(r)
    rows.sort(key=lambda r: -r["elpd_loo"])
    best = rows[0]["elpd_loo"]
    for r in rows:
        r["d_elpd"] = r["elpd_loo"] - best
    return rows


def log_nb(y, mu, phi):
    mu = np.maximum(mu, 1e-12)
    phi = np.maximum(phi, 1e-9)
    return (gammaln(y + phi) - gammaln(phi) - gammaln(y + 1)
            + phi * (np.log(phi) - np.log(phi + mu))
            + y * (np.log(mu) - np.log(phi + mu)))


def log_poisson(y, mu):
    mu = np.maximum(mu, 1e-12)
    return y * np.log(mu) - mu - gammaln(y + 1)


def log_dirichlet_multinomial(counts, alpha):
    counts = np.atleast_2d(counts)
    alpha = np.atleast_2d(alpha)
    a0 = alpha.sum(axis=-1)
    n = counts.sum(axis=-1)
    out = (gammaln(a0) - gammaln(n + a0)
           + np.sum(gammaln(counts + alpha) - gammaln(alpha), axis=-1)
           + gammaln(n + 1) - np.sum(gammaln(counts + 1), axis=-1))
    return out


def log_normal(x, mu, sd):
    sd = np.maximum(sd, 1e-9)
    return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi)


def log_halfnormal(x, sd=1.0):
    return np.where(x > 0, log_normal(x, 0.0, sd) + np.log(2.0), -np.inf)


def log_student_t(x, mu, sd, nu=3.0):
    return stats.t.logpdf((x - mu) / sd, nu) - np.log(sd)


def _adapt_blocked_chain(log_post, x0, blocks, n_draws, n_tune, rng,
                         target=0.30, thin=1, progress=None):
    x = np.asarray(x0, dtype=float).copy()
    lp = log_post(x)
    if not np.isfinite(lp):
        raise ValueError("initial point has non-finite log-posterior")

    nb = len(blocks)
    log_scale = [np.log(2.38 / np.sqrt(len(b))) for b in blocks]
    mu = [x[b].copy() for b in blocks]
    cov = [np.eye(len(b)) * 0.04 for b in blocks]
    chol = [np.linalg.cholesky(c) for c in cov]
    eps = 1e-10

    total = n_tune + n_draws * thin
    kept = np.empty((n_draws, len(x)))
    acc = np.zeros(nb)
    tries = np.zeros(nb)
    k = 0
    for i in range(total):
        for j, b in enumerate(blocks):
            prop = x.copy()
            step = np.exp(log_scale[j]) * (chol[j] @ rng.standard_normal(len(b)))
            prop[b] = x[b] + step
            lp_prop = log_post(prop)
            a = lp_prop - lp
            if np.log(rng.random() + 1e-300) < a:
                x, lp = prop, lp_prop
                if i >= n_tune:
                    acc[j] += 1
            if i >= n_tune:
                tries[j] += 1
            if i < n_tune:
                g = 1.0 / (i // 50 + 2) ** 0.7
                log_scale[j] += g * (min(np.exp(min(a, 0.0)), 1.0) - target)
                d = x[b] - mu[j]
                mu[j] = mu[j] + g * d
                cov[j] = cov[j] + g * (np.outer(d, d) - cov[j])
                if i % 200 == 0 and i > 400:
                    try:
                        chol[j] = np.linalg.cholesky(
                            cov[j] + eps * np.eye(len(b)))
                    except np.linalg.LinAlgError:
                        chol[j] = np.linalg.cholesky(
                            np.diag(np.maximum(np.diag(cov[j]), eps)))
        if i >= n_tune and (i - n_tune) % thin == 0 and k < n_draws:
            kept[k] = x
            k += 1
        if progress is not None and i % max(total // 20, 1) == 0:
            progress(i / total)
    return kept, float(np.mean(acc / np.maximum(tries, 1)))


def make_blocks(n_params, block_map=None, max_block=4):
    if block_map is not None:
        return [np.asarray(b, dtype=int) for b in block_map]
    idx = np.arange(n_params)
    return [idx[i:i + max_block] for i in range(0, n_params, max_block)]


def sample_blocked(log_post, init, *, blocks=None, max_block=4, n_draws=4000,
                   n_tune=4000, n_chains=4, seed=0, names=None, thin=1,
                   jitter=0.2, target=0.30, verbose=True) -> Posterior:
    init = np.atleast_2d(np.asarray(init, dtype=float))
    p = init.shape[1]
    if init.shape[0] == 1:
        init = np.repeat(init, n_chains, axis=0)
    names = list(names) if names is not None else [f"p{i}" for i in range(p)]
    blocks = make_blocks(p, blocks, max_block)

    t0 = time.time()
    draws = np.empty((n_chains, n_draws, p))
    acc = np.empty(n_chains)
    for c in range(n_chains):
        rng = np.random.default_rng(seed + 1000 * c)
        x0 = init[c] + jitter * rng.standard_normal(p)
        t = 0
        while not np.isfinite(log_post(x0)) and t < 100:
            x0 = init[c] + jitter * rng.standard_normal(p)
            t += 1
        cb = None
        if verbose:
            def cb(frac, c=c):
                print(f"\r    chain {c + 1}/{n_chains}  {frac:5.0%}",
                      end="", flush=True)
        draws[c], acc[c] = _adapt_blocked_chain(
            log_post, x0, blocks, n_draws, n_tune, rng, target=target,
            thin=thin, progress=cb)
        if verbose:
            print(f"\r    chain {c + 1}/{n_chains}  done  "
                  f"accept={acc[c]:.2f}", flush=True)
    return Posterior(draws, names, acc, time.time() - t0,
                     extra={"sampler": "blocked", "n_blocks": len(blocks)})
