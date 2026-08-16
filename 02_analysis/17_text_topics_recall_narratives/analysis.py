from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import (CountVectorizer, ENGLISH_STOP_WORDS,
                                             TfidfTransformer)
from sklearn.model_selection import KFold
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import data as D
from _common import mcmc as M

RES = D.results_dir(__file__)
SEED = 20260813

PERIOD_BINS = [1973, 1989, 1999, 2009, 2019, 2026]
PERIOD_LABELS = ["1974-89", "1990-99", "2000-09", "2010-19", "2020-26"]

BOILERPLATE = {
    "recall", "recalls", "recalled", "recalling", "announce", "announces",
    "announced", "announcement", "cpsc", "consumer", "consumers", "product",
    "products", "safety", "commission", "due", "sold", "exclusively",
    "imported", "importer", "importers", "distributed", "distributor",
    "manufactured", "manufacturer", "company", "companies", "inc", "llc",
    "ltd", "corp", "co", "usa", "us", "america", "american", "issues", "issue",
    "issued", "expands", "expanded", "expansion", "reannounces", "certain",
    "new", "alert", "alerts", "voluntary", "voluntarily", "stop", "using",
    "urges", "warns", "warning", "warnings", "notice", "brand", "brands",
    "national", "international", "group", "trading", "enterprises", "store",
    "stores", "sells", "selling", "sale", "sales", "may", "can", "could",
    "posed", "poses", "pose", "reported", "reports", "report", "unit", "units",
    "style", "styles", "model", "models", "number", "numbers", "size", "sizes",
    "one", "two", "three", "sr", "jr", "www", "com", "http", "https",
}
STOPWORDS = list(ENGLISH_STOP_WORDS | BOILERPLATE)

TOKEN_RE = re.compile(r"[a-z][a-z\-]{1,}")

_RECALL_TOKEN = re.compile(r"\b(?:recalls?|recalling|recalled|announce\s+"
                           r"(?:the\s+)?recall\s+of|halt\s+sale\s+of|"
                           r"stops?\s+(?:importing|selling|sale))\b", re.I)
_AFTER_STRIP = re.compile(r"^(?:of|to\s+repair|to\s+all|the|voluntary|expanded|"
                          r"consumer|its|two|certain|and\s+selling)\b[\s:,-]*", re.I)
_PASSIVE_MARK = re.compile(r"^(?:by\b|due\b|because\b|for\b|after\b|over\b|"
                           r"from\b|following\b|and\b|linked\b|amid\b|on\b|"
                           r"in\b|at\b|as\b|;|:|$)", re.I)
_TAIL = re.compile(
    r"\s*(?:;|:|\bdue\s+to\b|\bbecause\s+of\b|\bfor\s+risk\b|\bposes?\b"
    r"|\bfor\s+\w+\s+hazard\b|\bhazard\b|\bviolat\w*|\brisk\s+of\b"
    r"|\bsold\s+(?:exclusively\s+)?(?:at|by|on|in|through|with)\b"
    r"|\bimported\s+by\b|\bmanufactured\s+by\b|\bdistributed\s+by\b"
    r"|\brecalled\b|\bmay\s+not\b|\bafter\b|\bover\b|\bwarning\b)", re.I)
_CPSC_ANN = re.compile(r"^cpsc[^a-z]{0,3}\s*(?:and\s+)?.*?\bannounces?\b\s*", re.I)


def product_phrase(title: str) -> str:
    t = re.sub(r"\s+", " ", str(title or "")).strip()
    m = _RECALL_TOKEN.search(t)
    if m:
        after = _AFTER_STRIP.sub("", t[m.end():].strip())
        seg = (_CPSC_ANN.sub("", t[:m.start()])
               if (_PASSIVE_MARK.match(after) or not after) else after)
    else:
        seg = _CPSC_ANN.sub("", t)
    return re.sub(r"\s+", " ", _TAIL.split(seg)[0]).strip(" .,-:'\"")
TOP_COUNTRIES = ["China", "Vietnam", "India", "Pakistan", "Bangladesh",
                 "Indonesia", "United States"]

SETTINGS = {
    "quick": dict(k_range=list(range(2, 8)), n_restarts=2, perp_folds=2,
                  lda_max_iter=25, nmf_max_iter=400, n_draws=400, n_tune=400,
                  n_chains=2, top_terms=10, stab_runs_nmf=4, stab_runs_lda=2),
    "production": dict(k_range=list(range(2, 17)), n_restarts=5, perp_folds=3,
                       lda_max_iter=120, nmf_max_iter=3000, n_draws=8000,
                       n_tune=8000, n_chains=4, top_terms=12,
                       stab_runs_nmf=8, stab_runs_lda=3),
}


def clean_doc(title: str, firm: str | None, extra: str = "") -> str:
    t = f"{title or ''} {extra or ''}"
    t = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t).lower()
    if isinstance(firm, str) and len(firm) > 2:
        t = t.replace(firm.lower(), " ")
        for piece in firm.lower().split():
            if len(piece) > 3:
                t = re.sub(rf"\b{re.escape(piece)}\b", " ", t)
    t = re.sub(r"['\u2019]s\b", "", t)
    t = re.sub(r"s['\u2019](?![a-z])", "s", t)
    t = re.sub(r"[^a-z\s\-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def build_corpus():
    df = D.load_recalls()
    meta = {"source_version": df.attrs.get("source_version"),
            "n_records": int(len(df))}
    raw = D.load_raw_json()
    meta["raw_json_present"] = raw is not None
    extra = pd.Series("", index=df.index)
    if raw is not None:
        keep = raw[["recall_id", "description", "hazard_text", "products"]]
        df = df.merge(keep, on="recall_id", how="left", suffixes=("", "_raw"))
        extra = (df.get("products", pd.Series("", index=df.index)).fillna("")
                 + " " + df.get("hazard_text", pd.Series("", index=df.index)).fillna("")
                 + " " + df.get("description", pd.Series("", index=df.index)).fillna(""))
        meta["text_fields"] = ["title", "products", "hazard_text", "description"]
    else:
        meta["text_fields"] = ["title"]

    df = df.copy()
    df["period"] = pd.cut(df.year, bins=PERIOD_BINS, labels=PERIOD_LABELS)
    df["doc"] = [clean_doc(t, f, e) for t, f, e in
                 zip(df.title, df.firm, extra)]
    prod_extra = (df.get("products", pd.Series("", index=df.index)).fillna("")
                  if raw is not None else pd.Series("", index=df.index))
    df["product_phrase"] = df.title.map(product_phrase)
    df["doc_product"] = [clean_doc(t, f, e) for t, f, e in
                         zip(df.product_phrase, df.firm, prod_extra)]
    df["n_tokens"] = df.doc.map(lambda s: len(TOKEN_RE.findall(s)))
    df["n_tokens_product"] = df.doc_product.map(
        lambda s: len(TOKEN_RE.findall(s)))
    meta["n_documents_before_filter"] = int(len(df))
    df = df[(df.n_tokens >= 3) & (df.n_tokens_product >= 2)].reset_index(
        drop=True)
    meta["n_documents"] = int(len(df))
    meta["mean_tokens"] = float(df.n_tokens.mean())
    meta["median_tokens"] = float(df.n_tokens.median())
    meta["total_tokens"] = int(df.n_tokens.sum())
    meta["mean_tokens_product"] = float(df.n_tokens_product.mean())
    meta["total_tokens_product"] = int(df.n_tokens_product.sum())
    return df, meta


def vectorise(docs, min_df, max_df=0.6):
    cv = CountVectorizer(stop_words=STOPWORDS, ngram_range=(1, 2),
                         min_df=min_df, max_df=max_df,
                         token_pattern=r"(?u)\b[a-z][a-z\-]+\b")
    X = cv.fit_transform(docs)
    tfidf = TfidfTransformer(sublinear_tf=True, norm="l2").fit_transform(X)
    return cv, X, tfidf


def npmi_coherence(top_idx: list[np.ndarray], Xbin: np.ndarray) -> float:
    n_docs = Xbin.shape[0]
    scores = []
    for idx in top_idx:
        cols = Xbin[:, idx]
        df_i = cols.sum(axis=0) + 1.0
        co = cols.T @ cols + 1.0
        vals = []
        m = len(idx)
        for a in range(m):
            for b in range(a + 1, m):
                p_ij = co[a, b] / (n_docs + 1.0)
                p_i = df_i[a] / (n_docs + 1.0)
                p_j = df_i[b] / (n_docs + 1.0)
                pmi = np.log(p_ij / (p_i * p_j))
                vals.append(pmi / (-np.log(p_ij)))
        if vals:
            scores.append(float(np.mean(vals)))
    return float(np.mean(scores)) if scores else np.nan


def top_indices(components: np.ndarray, n_top: int) -> list[np.ndarray]:
    return [np.argsort(c)[::-1][:n_top] for c in components]


def consensus_cophenetic(fit_labels, n_docs, n_runs, seed, subsample=0.8):
    C = np.zeros((n_docs, n_docs))
    N = np.zeros((n_docs, n_docs))
    for b in range(n_runs):
        rng = np.random.default_rng(seed + 977 * b)
        idx = np.sort(rng.choice(n_docs, size=int(subsample * n_docs),
                                 replace=False))
        lab = fit_labels(idx, seed + b)
        same = (lab[:, None] == lab[None, :]).astype(float)
        C[np.ix_(idx, idx)] += same
        N[np.ix_(idx, idx)] += 1.0
    M = C / np.maximum(N, 1.0)
    d = squareform(np.clip(1.0 - M, 0, 1), checks=False)
    if not np.isfinite(d).all() or d.std() == 0:
        return np.nan
    Z = linkage(d, method="average")
    return float(cophenet(Z, d)[0])


def sweep(X, tfidf, Xbin, cfg):
    rows = []
    kf = KFold(cfg["perp_folds"], shuffle=True, random_state=SEED)
    for k in cfg["k_range"]:
        t0 = time.time()
        recon, coh_nmf = [], []
        for r in range(cfg["n_restarts"]):
            nmf = NMF(n_components=k, init="nndsvda", random_state=SEED + r,
                      max_iter=cfg["nmf_max_iter"], tol=1e-5)
            W = nmf.fit_transform(tfidf)
            recon.append(float(nmf.reconstruction_err_))
            coh_nmf.append(npmi_coherence(top_indices(nmf.components_, 10),
                                          Xbin))
        heldout = []
        for tr, te in kf.split(np.arange(X.shape[0])):
            nmf = NMF(n_components=k, init="nndsvda", random_state=SEED,
                      max_iter=cfg["nmf_max_iter"], tol=1e-5)
            nmf.fit(tfidf[tr])
            Wte = nmf.transform(tfidf[te])
            resid = np.asarray(tfidf[te].todense()) - Wte @ nmf.components_
            heldout.append(float(np.sqrt((resid ** 2).sum())))
        perp, coh_lda = [], []
        for tr, te in kf.split(np.arange(X.shape[0])):
            lda = LatentDirichletAllocation(
                n_components=k, learning_method="batch",
                max_iter=cfg["lda_max_iter"], random_state=SEED,
                doc_topic_prior=None, topic_word_prior=None)
            lda.fit(X[tr])
            perp.append(float(lda.perplexity(X[te])))
        for r in range(cfg["n_restarts"]):
            lda = LatentDirichletAllocation(
                n_components=k, learning_method="batch",
                max_iter=cfg["lda_max_iter"], random_state=SEED + r)
            lda.fit(X)
            coh_lda.append(npmi_coherence(top_indices(lda.components_, 10),
                                          Xbin))
        def nmf_labels(idx, sd, k=k):
            m = NMF(n_components=k, init="nndsvdar", random_state=sd,
                    max_iter=cfg["nmf_max_iter"], tol=1e-4)
            return m.fit_transform(tfidf[idx]).argmax(axis=1)

        def lda_labels(idx, sd, k=k):
            m = LatentDirichletAllocation(n_components=k,
                                          learning_method="batch",
                                          max_iter=cfg["lda_max_iter"],
                                          random_state=sd)
            return m.fit_transform(X[idx]).argmax(axis=1)

        stab_nmf = consensus_cophenetic(nmf_labels, X.shape[0],
                                        cfg["stab_runs_nmf"], SEED)
        stab_lda = consensus_cophenetic(lda_labels, X.shape[0],
                                        cfg["stab_runs_lda"], SEED)
        rows.append({
            "k": k,
            "nmf_stability_cophenetic": stab_nmf,
            "lda_stability_cophenetic": stab_lda,
            "nmf_reconstruction_err": float(np.mean(recon)),
            "nmf_reconstruction_sd": float(np.std(recon, ddof=1)
                                           if len(recon) > 1 else 0.0),
            "nmf_heldout_err": float(np.mean(heldout)),
            "nmf_coherence_npmi": float(np.mean(coh_nmf)),
            "nmf_coherence_sd": float(np.std(coh_nmf, ddof=1)
                                      if len(coh_nmf) > 1 else 0.0),
            "lda_perplexity": float(np.mean(perp)),
            "lda_perplexity_sd": float(np.std(perp, ddof=1)
                                       if len(perp) > 1 else 0.0),
            "lda_coherence_npmi": float(np.mean(coh_lda)),
            "lda_coherence_sd": float(np.std(coh_lda, ddof=1)
                                      if len(coh_lda) > 1 else 0.0),
            "seconds": time.time() - t0,
        })
        print(f"  [17] k = {k:2d}  NMF coh {rows[-1]['nmf_coherence_npmi']:+.3f}"
              f"  stab {stab_nmf:.3f}  err {rows[-1]['nmf_reconstruction_err']:.2f}"
              f" | LDA coh {rows[-1]['lda_coherence_npmi']:+.3f}"
              f"  stab {stab_lda:.3f}  perp {rows[-1]['lda_perplexity']:.0f}"
              f"   ({rows[-1]['seconds']:.0f}s)")
    return pd.DataFrame(rows)


def log_odds_prior(counts_a, counts_b, prior_counts, a0=500.0):
    ya = np.asarray(counts_a, dtype=float)
    yb = np.asarray(counts_b, dtype=float)
    p = np.asarray(prior_counts, dtype=float)
    alpha = a0 * p / max(p.sum(), 1.0)
    na, nb = ya.sum(), yb.sum()
    oa = (ya + alpha) / np.maximum(na + a0 - ya - alpha, 1e-9)
    ob = (yb + alpha) / np.maximum(nb + a0 - yb - alpha, 1e-9)
    delta = np.log(oa) - np.log(ob)
    var = 1.0 / (ya + alpha) + 1.0 / (yb + alpha)
    return delta, delta / np.sqrt(var), alpha


def contrast_table(vocab, X, mask_a, mask_b, label_a, label_b, a0=500.0):
    ca = np.asarray(X[mask_a].sum(axis=0)).ravel()
    cb = np.asarray(X[mask_b].sum(axis=0)).ravel()
    total = np.asarray(X.sum(axis=0)).ravel()
    delta, z, alpha = log_odds_prior(ca, cb, total, a0)
    doc_freq = np.asarray((X > 0).sum(axis=0)).ravel()
    out = pd.DataFrame({
        "term": vocab, "count_a": ca, "count_b": cb,
        "count_total": total, "doc_freq": doc_freq,
        "freq_per_1k_a": 1000 * ca / max(ca.sum(), 1),
        "freq_per_1k_b": 1000 * cb / max(cb.sum(), 1),
        "log_odds_delta": delta, "z": z,
        "side": np.where(z > 0, label_a, label_b),
    })
    out.attrs["label_a"], out.attrs["label_b"] = label_a, label_b
    return out.sort_values("z", ascending=False).reset_index(drop=True)


def trend_model(counts: np.ndarray, years: np.ndarray, cfg, names):
    T, K = counts.shape
    Km1 = K - 1
    z = (years - years.mean()) / 10.0
    n_par = 3 * Km1 + Km1 * T + 1
    par_names = ([f"alpha_{k}" for k in range(Km1)]
                 + [f"beta_{k}" for k in range(Km1)]
                 + [f"log_sigma_{k}" for k in range(Km1)]
                 + [f"w_{k}_{t}" for k in range(Km1) for t in range(T)]
                 + ["log_phi"])

    def unpack(th):
        i = 0
        a = th[i:i + Km1]; i += Km1
        b = th[i:i + Km1]; i += Km1
        ls = th[i:i + Km1]; i += Km1
        w = th[i:i + Km1 * T].reshape(Km1, T); i += Km1 * T
        return a, b, ls, w, th[i]

    Zt = np.column_stack([np.ones(T), z])
    Proj = np.eye(T) - Zt @ np.linalg.solve(Zt.T @ Zt, Zt.T)

    def eta_of(a, b, ls, w):
        rw = np.exp(ls)[:, None] * (np.cumsum(w, axis=1) @ Proj.T)
        return (a[:, None] + b[:, None] * z[None, :] + rw).T

    def probs(th):
        a, b, ls, w, _ = unpack(th)
        e = np.concatenate([eta_of(a, b, ls, w), np.zeros((T, 1))], axis=1)
        e = e - e.max(axis=1, keepdims=True)
        p = np.exp(e)
        return p / p.sum(axis=1, keepdims=True)

    def log_post(th):
        a, b, ls, w, lphi = unpack(th)
        if not np.all(np.isfinite(th)) or abs(lphi) > 12 or np.any(np.abs(ls) > 6):
            return -np.inf
        p = probs(th)
        ll = float(M.log_dirichlet_multinomial(counts, np.exp(lphi) * p).sum())
        lp = (np.sum(M.log_normal(a, 0.0, 2.0))
              + np.sum(M.log_normal(b, 0.0, 1.5))
              + np.sum(M.log_normal(ls, np.log(0.25), 0.75))
              + np.sum(M.log_normal(w, 0.0, 1.0))
              + float(M.log_normal(lphi, np.log(40.0), 1.5)))
        return ll + lp

    blocks = []
    for k in range(Km1):
        blocks.append(np.array([k, Km1 + k, 2 * Km1 + k]))
    base = 3 * Km1
    for k in range(Km1):
        idx = np.arange(base + k * T, base + (k + 1) * T)
        blocks += [idx[i:i + 4] for i in range(0, T, 4)]
    blocks.append(np.array([n_par - 1]))

    init = np.zeros(n_par)
    share = counts.sum(axis=0) / counts.sum()
    init[:Km1] = np.log(np.maximum(share[:Km1], 1e-3) / max(share[-1], 1e-3))
    init[2 * Km1:3 * Km1] = np.log(0.25)
    init[-1] = np.log(40.0)

    post = M.sample_blocked(log_post, init, blocks=blocks,
                            n_draws=cfg["n_draws"], n_tune=cfg["n_tune"],
                            n_chains=cfg["n_chains"], seed=SEED,
                            names=par_names, verbose=True)

    flat = post.flat()
    sub = flat[np.random.default_rng(SEED).choice(
        len(flat), size=min(2000, len(flat)), replace=False)]
    P = np.stack([probs(th) for th in sub])
    curves = []
    for k in range(K):
        for t in range(T):
            x = P[:, t, k]
            lo, hi = np.percentile(x, [3, 97])
            curves.append({"year": int(years[t]), "topic": names[k],
                           "topic_index": k, "mean": float(x.mean()),
                           "lo94": float(lo), "hi94": float(hi),
                           "observed": float(counts[t, k] / counts[t].sum()),
                           "n": int(counts[t].sum())})
    rows = []
    hdi = post.hdi(0.94)
    rhat, ess = post.rhat(), post.ess()
    for j, nm in enumerate(par_names):
        if nm.startswith("w_"):
            continue
        x = flat[:, j]
        rows.append({"param": nm, "mean": float(x.mean()),
                     "sd": float(x.std(ddof=1)),
                     "hdi_3%": float(hdi[j][0]), "hdi_97%": float(hdi[j][1]),
                     "rhat": float(rhat[j]), "ess": float(ess[j]),
                     "p_positive": float((x > 0).mean())})
    diag = {
        "n_params": n_par, "n_blocks": len(blocks),
        "n_draws": cfg["n_draws"], "n_tune": cfg["n_tune"],
        "n_chains": cfg["n_chains"], "seconds": post.seconds,
        "accept_rate": post.accept_rate.tolist(),
        "max_rhat_all": float(np.nanmax(rhat)),
        "min_ess_all": float(np.nanmin(ess)),
        "max_rhat_trend": float(np.nanmax(
            [r for r, nm in zip(rhat, par_names) if not nm.startswith("w_")])),
        "min_ess_trend": float(np.nanmin(
            [e for e, nm in zip(ess, par_names) if not nm.startswith("w_")])),
    }
    return pd.DataFrame(curves), pd.DataFrame(rows), diag


def cramers_v(tab):
    chi2, p, dof, exp = stats.chi2_contingency(tab)
    n = tab.to_numpy().sum()
    k = min(tab.shape) - 1
    return {"chi2": float(chi2), "dof": int(dof), "p": float(p),
            "cramers_v": float(np.sqrt(chi2 / (n * k))) if n and k else np.nan,
            "min_expected": float(exp.min()), "n": int(n)}


def main(quick: bool = False):
    mode = "quick" if quick else "production"
    cfg = SETTINGS[mode]
    t0 = time.time()

    df, meta = build_corpus()
    min_df = 4 if meta["raw_json_present"] else 3
    cv, X, tfidf = vectorise(df.doc_product.tolist(), min_df=min_df)
    cv_full, X_full, _ = vectorise(df.doc.tolist(), min_df=min_df)
    vocab = np.array(cv.get_feature_names_out())
    vocab_full = np.array(cv_full.get_feature_names_out())
    Xd = np.asarray(X.todense())
    Xbin = (Xd > 0).astype(float)
    meta.update({"vocabulary_product": int(len(vocab)),
                 "vocabulary_full": int(len(vocab_full)), "min_df": min_df,
                 "matrix_density": float((Xd > 0).mean()),
                 "n_unigrams": int(sum(" " not in t for t in vocab)),
                 "n_bigrams": int(sum(" " in t for t in vocab)),
                 "n_bigrams_full": int(sum(" " in t for t in vocab_full)),
                 "mode": mode})
    print(f"[17] corpus: {len(df)} documents; full text {meta['total_tokens']} "
          f"tokens / vocabulary {len(vocab_full)}; product phrases "
          f"{meta['total_tokens_product']} tokens / vocabulary {len(vocab)} "
          f"({meta['n_bigrams']} bigrams); raw JSON: {meta['raw_json_present']}")

    corpus_stats = (df.groupby("period", observed=True)
                    .agg(n_documents=("doc", "size"),
                         mean_tokens=("n_tokens", "mean"),
                         total_tokens=("n_tokens", "sum")).reset_index())
    corpus_stats["distinct_terms"] = [
        int((Xd[(df.period == p).to_numpy()].sum(axis=0) > 0).sum())
        for p in corpus_stats.period]
    corpus_stats.to_csv(RES / "corpus_stats.csv", index=False)

    sel = sweep(X, tfidf, Xbin, cfg)
    sel.to_csv(RES / "topic_selection.csv", index=False)

    def choose_k(stab_col, coh_col):
        best = float(sel[stab_col].max())
        v = sel[stab_col].to_numpy()
        i = 0
        while i + 1 < len(v) and v[i + 1] >= v[i] - 0.005:
            i += 1
        k_sel = int(sel.k.iloc[i])
        return (k_sel, int(sel.loc[sel[stab_col].idxmax(), "k"]),
                float(v[i]), best,
                float(sel.loc[sel.k == k_sel, coh_col].iloc[0]),
                float(sel[coh_col].max()),
                int(sel.loc[sel[coh_col].idxmax(), "k"]))

    (k_nmf, k_nmf_stabmax, stab_nmf_at_k, stab_nmf_best, coh_nmf_at_k,
     coh_nmf_max, k_nmf_cohmax) = choose_k("nmf_stability_cophenetic",
                                           "nmf_coherence_npmi")
    (k_lda, k_lda_stabmax, stab_lda_at_k, stab_lda_best, coh_lda_at_k,
     coh_lda_max, k_lda_cohmax) = choose_k("lda_stability_cophenetic",
                                           "lda_coherence_npmi")
    best_nmf, best_lda = coh_nmf_at_k, coh_lda_at_k
    k_perp = int(sel.loc[sel.lda_perplexity.idxmin(), "k"])
    primary = "nmf" if best_nmf >= best_lda else "lda"
    k = k_nmf if primary == "nmf" else k_lda
    print(f"[17] selection: NMF first stability peak at k = {k_nmf} "
          f"(cophenetic {stab_nmf_at_k:.3f}; global max {stab_nmf_best:.3f} at "
          f"k = {k_nmf_stabmax}), NPMI there {coh_nmf_at_k:+.3f}; NPMI alone "
          f"would say k = {k_nmf_cohmax}. LDA k = {k_lda} "
          f"(cophenetic {stab_lda_at_k:.3f}, NPMI {coh_lda_at_k:+.3f}); "
          f"LDA min perplexity at k = {k_perp}. "
          f"primary = {primary.upper()}, k = {k}")

    nmf = NMF(n_components=k, init="nndsvda", random_state=SEED,
              max_iter=cfg["nmf_max_iter"], tol=1e-6)
    W_nmf = nmf.fit_transform(tfidf)
    lda = LatentDirichletAllocation(n_components=k, learning_method="batch",
                                    max_iter=cfg["lda_max_iter"],
                                    random_state=SEED)
    W_lda = lda.fit_transform(X)
    models = {"nmf": (nmf.components_, W_nmf), "lda": (lda.components_, W_lda)}
    comps, W_all = models[primary]

    assigned = np.asarray((X > 0).sum(axis=1)).ravel() > 0
    df["topic_assigned"] = assigned
    meta["n_documents_topic_assigned"] = int(assigned.sum())
    meta["n_documents_no_topic_terms"] = int((~assigned).sum())

    dfa = df[assigned].reset_index(drop=True)
    Wraw = W_all[assigned]
    Wn = Wraw / np.maximum(Wraw.sum(axis=1, keepdims=True), 1e-12)
    dfa["dominant_topic"] = Wn.argmax(axis=1)
    dfa["dominant_weight"] = Wn.max(axis=1)
    print(f"[17] {int(assigned.sum())} documents carry at least one in-vocabulary "
          f"product term; {int((~assigned).sum())} are unassigned")

    term_rows, names = [], []
    for model_name, (C, _) in models.items():
        for ti, c in enumerate(C):
            order = np.argsort(c)[::-1][:cfg["top_terms"]]
            for rank, j in enumerate(order):
                term_rows.append({"model": model_name, "topic": ti,
                                  "rank": rank, "term": vocab[j],
                                  "weight": float(c[j] / c.sum())})
    for ti, c in enumerate(comps):
        order = np.argsort(c)[::-1]
        picked = []
        for j in order:
            t = vocab[j]
            if any(t in q or q in t for q in picked):
                continue
            picked.append(t)
            if len(picked) == 2:
                break
        names.append(" / ".join(picked))
    pd.DataFrame(term_rows).to_csv(RES / "topic_terms.csv", index=False)

    doc_rows = []
    for ti in range(k):
        order = np.argsort(Wraw[:, ti])[::-1][:5]
        for rank, i in enumerate(order):
            doc_rows.append({"topic": ti, "topic_name": names[ti],
                             "rank": rank, "weight_raw": float(Wraw[i, ti]),
                             "weight_share": float(Wn[i, ti]),
                             "year": int(dfa.year.iloc[i]),
                             "sales_channel": dfa.sales_channel.iloc[i],
                             "title": dfa.title.iloc[i]})
    pd.DataFrame(doc_rows).to_csv(RES / "topic_documents.csv", index=False)

    tp = []
    for p, sub in dfa.groupby("period", observed=True):
        idx = sub.index.to_numpy()
        for ti in range(k):
            tp.append({"period": str(p), "topic": ti, "topic_name": names[ti],
                       "n_period": int(len(sub)),
                       "n_dominant": int((sub.dominant_topic == ti).sum()),
                       "share_dominant": float((sub.dominant_topic == ti).mean()),
                       "mean_weight": float(Wn[idx, ti].mean())})
    topic_period = pd.DataFrame(tp)
    topic_period.to_csv(RES / "topic_by_period.csv", index=False)

    grp_rows, tests = [], {}
    dfa["country_group"] = np.where(dfa.primary_country.isin(TOP_COUNTRIES),
                                    dfa.primary_country,
                                    np.where(dfa.primary_country.isna(),
                                             "unknown", "other"))
    for var in ["sales_channel", "hazard_category", "country_group"]:
        tab = pd.crosstab(dfa.dominant_topic, dfa[var])
        tests[var] = cramers_v(tab)
        share = tab.div(tab.sum(axis=0), axis=1)
        lift = tab.div(tab.sum(axis=1), axis=0).div(
            tab.sum(axis=0) / tab.to_numpy().sum(), axis=1)
        for ti in tab.index:
            for g in tab.columns:
                grp_rows.append({
                    "variable": var, "topic": int(ti),
                    "topic_name": names[int(ti)], "group": g,
                    "n": int(tab.at[ti, g]),
                    "share_of_group": float(share.at[ti, g]),
                    "lift": float(lift.at[ti, g]),
                    "mean_weight": float(Wn[(dfa[var] == g).to_numpy(), ti].mean())})
    pd.DataFrame(grp_rows).to_csv(RES / "topic_by_group.csv", index=False)

    out = df[["recall_id", "year", "period", "title", "product_phrase",
              "sales_channel", "hazard_category", "primary_country",
              "topic_assigned"]].copy()
    out["dominant_topic"] = np.where(assigned, -1, -1)
    out.loc[assigned, "dominant_topic"] = dfa.dominant_topic.to_numpy()
    out["dominant_weight"] = np.nan
    out.loc[assigned, "dominant_weight"] = dfa.dominant_weight.to_numpy()
    for ti in range(k):
        col = np.full(len(df), np.nan)
        col[assigned] = Wn[:, ti]
        out[f"w_topic_{ti}"] = col
    out.to_csv(RES / "doc_topics.csv", index=False)

    ch = df.sales_channel.to_numpy()
    yr = df.year.to_numpy()
    lo_channel = contrast_table(vocab_full, X_full, ch == "online_only",
                                ch == "store_only", "online_only", "store_only")
    lo_era = contrast_table(vocab_full, X_full, yr >= 2015, yr < 2015,
                            "2015-2026", "pre-2015")
    lo_channel.to_csv(RES / "logodds_online_vs_store.csv", index=False)
    lo_era.to_csv(RES / "logodds_recent_vs_early.csv", index=False)

    def head_tail(t, n=15):
        return {"top_a": t.head(n).to_dict(orient="records"),
                "top_b": t.tail(n).iloc[::-1].to_dict(orient="records")}

    size = topic_period.groupby("topic").n_dominant.sum().sort_values(
        ascending=False)
    top3 = [int(t) for t in size.index[:3]]
    years = np.arange(2000, int(dfa.year.max()) + 1)
    sub = dfa[dfa.year >= 2000]
    counts = np.zeros((len(years), 4))
    for i, y in enumerate(years):
        d = sub[sub.year == y]
        for j, ti in enumerate(top3):
            counts[i, j] = int((d.dominant_topic == ti).sum())
        counts[i, 3] = int(len(d)) - counts[i, :3].sum()
    keep = counts.sum(axis=1) > 0
    counts, years_k = counts[keep], years[keep]
    trend_names = [names[t] for t in top3] + ["all other topics"]
    curves, params, diag = trend_model(counts, years_k, cfg, trend_names)
    curves.to_csv(RES / "topic_trend_posterior.csv", index=False)
    params.to_csv(RES / "topic_trend_params.csv", index=False)

    slopes = []
    for j, ti in enumerate(top3):
        r = params[params.param == f"beta_{j}"].iloc[0]
        c0 = curves[(curves.topic_index == j) & (curves.year == years_k.min())]
        c1 = curves[(curves.topic_index == j) & (curves.year == years_k.max())]
        slopes.append({
            "topic": ti, "topic_name": names[ti],
            "beta_per_decade": float(r["mean"]),
            "hdi_3%": float(r["hdi_3%"]), "hdi_97%": float(r["hdi_97%"]),
            "p_positive": float(r["p_positive"]),
            "share_start": float(c0["mean"].iloc[0]),
            "share_end": float(c1["mean"].iloc[0]),
            "rhat": float(r["rhat"]), "ess": float(r["ess"])})

    summary = {
        "analysis": "17_text_topics_recall_narratives",
        "mode": mode, "seconds": time.time() - t0,
        "corpus": meta,
        "settings": {kk: vv for kk, vv in cfg.items()},
        "selection": {
            "k_range": cfg["k_range"], "primary_model": primary, "k": k,
            "rule": "consensus-stability (cophenetic correlation, Brunet "
                    "et al. 2004); coherence and perplexity reported alongside",
            "nmf_k_selected": k_nmf, "nmf_stability_at_k": stab_nmf_at_k,
            "nmf_stability_global_max": stab_nmf_best,
            "nmf_k_if_stability_argmax": k_nmf_stabmax,
            "nmf_coherence_at_k": coh_nmf_at_k,
            "nmf_coherence_max": coh_nmf_max,
            "nmf_k_if_coherence_argmax": k_nmf_cohmax,
            "nmf_best_coherence": best_nmf,
            "lda_k_selected": k_lda, "lda_stability_at_k": stab_lda_at_k,
            "lda_stability_global_max": stab_lda_best,
            "lda_coherence_at_k": coh_lda_at_k,
            "lda_coherence_max": coh_lda_max,
            "lda_k_if_coherence_argmax": k_lda_cohmax,
            "lda_best_coherence": best_lda,
            "lda_min_perplexity_k": k_perp,
            "lda_min_perplexity": float(sel.lda_perplexity.min()),
            "curve": sel.to_dict(orient="records")},
        "topics": [{
            "topic": ti, "name": names[ti],
            "n_dominant": int((dfa.dominant_topic == ti).sum()),
            "share": float((dfa.dominant_topic == ti).mean()),
            "top_terms": [vocab[j] for j in np.argsort(comps[ti])[::-1][:10]],
            "representative_titles":
                dfa.title.iloc[np.argsort(Wraw[:, ti])[::-1][:3]].tolist(),
            "share_by_period": topic_period[topic_period.topic == ti]
                .set_index("period").share_dominant.to_dict(),
        } for ti in range(k)],
        "association_tests": tests,
        "logodds_online_vs_store": head_tail(lo_channel),
        "logodds_recent_vs_early": head_tail(lo_era),
        "logodds_settings": {"prior_a0": 500.0,
                             "n_terms": int(len(vocab_full)),
                             "n_online_only": int((ch == "online_only").sum()),
                             "n_store_only": int((ch == "store_only").sum()),
                             "n_2015plus": int((yr >= 2015).sum()),
                             "n_pre2015": int((yr < 2015).sum())},
        "trend_model": {"topics": trend_names, "years":
                        [int(years_k.min()), int(years_k.max())],
                        "slopes": slopes, "diagnostics": diag},
    }
    D.write_json(summary, RES / "summary.json")

    print(f"[17] primary {primary.upper()} with k = {k}; topics: "
          + "; ".join(f"{n} ({int((dfa.dominant_topic == i).sum())})"
                      for i, n in enumerate(names)))
    print(f"[17] online-only vs stores, top terms: "
          + ", ".join(f"{r.term} (z={r.z:.1f})"
                      for r in lo_channel.head(6).itertuples()))
    print(f"[17] stores vs online-only, top terms: "
          + ", ".join(f"{r.term} (z={r.z:.1f})"
                      for r in lo_channel.tail(6).iloc[::-1].itertuples()))
    for s in slopes:
        print(f"[17] trend {s['topic_name']}: beta/decade {s['beta_per_decade']:+.2f} "
              f"[{s['hdi_3%']:+.2f}, {s['hdi_97%']:+.2f}], "
              f"P(>0) = {s['p_positive']:.3f}, share "
              f"{s['share_start']:.2f} -> {s['share_end']:.2f}")
    print(f"[17] MCMC: max Rhat {diag['max_rhat_all']:.3f} (trend params "
          f"{diag['max_rhat_trend']:.3f}), min ESS {diag['min_ess_all']:.0f}")
    print(f"[17] done in {(time.time() - t0) / 60:.1f} min")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke run, < 90 s")
    main(**vars(ap.parse_args()))
