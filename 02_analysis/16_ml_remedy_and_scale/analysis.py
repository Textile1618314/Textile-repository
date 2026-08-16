from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, log_loss, mean_absolute_error,
                             r2_score, roc_auc_score, root_mean_squared_error)
from sklearn.model_selection import (KFold, RandomizedSearchCV,
                                     StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import data as D

RES = D.results_dir(__file__)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

SEED = 20260813

SETTINGS = {
    "quick": dict(outer_folds=3, outer_repeats=1, inner_folds=3,
                  n_iter={"elastic_net": 4, "random_forest": 3,
                          "hist_gradient_boosting": 4},
                  n_boot=60, perm_repeats=3, perm_folds=2, ice_lines=25,
                  grid_res=8, light=True),
    "mid": dict(outer_folds=5, outer_repeats=2, inner_folds=5,
                n_iter={"elastic_net": 50, "random_forest": 30,
                        "hist_gradient_boosting": 50},
                n_boot=600, perm_repeats=20, perm_folds=5, ice_lines=120,
                grid_res=20, light=False),
    "production": dict(outer_folds=5, outer_repeats=5, inner_folds=5,
                       n_iter={"elastic_net": 200, "random_forest": 200,
                               "hist_gradient_boosting": 240},
                       n_boot=2000, perm_repeats=50, perm_folds=5,
                       ice_lines=200, grid_res=25, light=False),
}

TOP_COUNTRIES = ["China", "Vietnam", "India", "Pakistan", "Bangladesh",
                 "Indonesia", "United States", "Hong Kong"]

FEATURE_LABELS = {
    "year": "Recall year",
    "log10_units": "Units recalled (log10)",
    "log10_price": "Unit price (log10 US$)",
    "n_countries": "Countries of manufacture (count)",
    "firm_prior_recalls": "Prior recalls by the same firm",
    "is_childrens": "Children's product",
    "is_violation": "Standard violation cited",
    "injuries_reported": "Injury reported",
    "sleepwear_standard": "Sleepwear standard cited",
    "category_arbitrage": "Category arbitrage",
    "is_electric_textile": "Electrically heated textile",
    "hazard_category": "Hazard category",
    "sales_channel": "Sales channel",
    "primary_country": "Country of manufacture",
    "archetype": "Garment archetype",
    "boundary_class": "Exemption boundary class",
    "enforcement_mode": "Enforcement mode",
    "segment": "Segment (apparel / home textile)",
    "title_words": "Title length (words)",
    "desc_chars": "Description length (chars)",
    "n_retailers": "Retailers listed",
    "n_products": "Products listed",
    "sold_online_text": "Online marketplace named in text",
}

MODEL_LABELS = {
    "elastic_net": "Elastic-net logistic",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Gradient boosting",
    "dummy_stratified": "Dummy (stratified)",
    "dummy_prior": "Dummy (prior)",
}
MODEL_LABELS_REG = dict(MODEL_LABELS,
                        elastic_net="Elastic net",
                        dummy_stratified="Dummy (median)",
                        dummy_prior="Dummy (mean)")


def build_frame() -> tuple[pd.DataFrame, dict]:
    df = D.load_recalls()
    meta = {"source_version": df.attrs.get("source_version"),
            "n_records": int(len(df))}

    raw = D.load_raw_json()
    meta["raw_json_present"] = raw is not None
    if raw is not None:
        keep = ["recall_id", "description", "hazard_text", "products",
                "retailers"]
        df = df.merge(raw[keep], on="recall_id", how="left",
                      suffixes=("", "_raw"))

    df = df.copy()
    df["log10_units"] = np.log10(pd.to_numeric(df.units, errors="coerce")
                                 .where(lambda s: s > 0))
    df["log10_price"] = np.log10(pd.to_numeric(df.price_usd, errors="coerce")
                                 .where(lambda s: s > 0))
    df["title_words"] = df.title.fillna("").str.split().map(len)

    df["recall_date"] = pd.to_datetime(df.recall_date, errors="coerce")
    order = df.sort_values(["recall_date", "recall_id"]).index
    prior = pd.Series(0, index=df.index, dtype=float)
    seen: dict[str, int] = {}
    for i in order:
        k = df.at[i, "firm_key"]
        if isinstance(k, str) and k:
            prior.at[i] = seen.get(k, 0)
            seen[k] = seen.get(k, 0) + 1
    df["firm_prior_recalls"] = prior

    for c in ["is_childrens", "is_violation", "injuries_reported",
              "sleepwear_standard", "category_arbitrage", "is_electric_textile"]:
        if c in df:
            df[c] = df[c].fillna(False).astype(float)
        else:
            df[c] = 0.0

    df["primary_country"] = np.where(
        df.primary_country.isin(TOP_COUNTRIES), df.primary_country,
        np.where(df.primary_country.isna(), "missing", "other"))
    for c in ["hazard_category", "sales_channel", "archetype",
              "boundary_class", "enforcement_mode", "segment"]:
        df[c] = df[c].fillna("missing").astype(str)

    text_feats = []
    if raw is not None:
        df["desc_chars"] = df.description.fillna("").str.len().astype(float)
        df["n_retailers"] = (df.retailers.fillna("").str.count(r"\|")
                             + df.retailers.fillna("").ne("").astype(int)).astype(float)
        df["n_products"] = (df.products.fillna("").str.count(r"\|")
                            + df.products.fillna("").ne("").astype(int)).astype(float)
        pool = (df.title.fillna("") + " " + df.description.fillna("") + " "
                + df.retailers.fillna(""))
        df["sold_online_text"] = pool.str.contains(
            r"amazon|ebay|etsy|shein|temu|walmart\.com|online", case=False,
            regex=True).astype(float)
        text_feats = ["desc_chars", "n_retailers", "n_products",
                      "sold_online_text"]
    meta["text_features"] = text_feats
    return df, meta


NUM_BASE = ["year", "log10_price", "n_countries", "firm_prior_recalls",
            "is_childrens", "is_violation", "injuries_reported",
            "sleepwear_standard", "category_arbitrage", "is_electric_textile",
            "title_words"]
CAT_BASE = ["hazard_category", "sales_channel", "primary_country", "archetype",
            "boundary_class", "enforcement_mode", "segment"]


def task_matrices(df: pd.DataFrame, task: str, text_feats: list[str]):
    num = list(NUM_BASE) + list(text_feats)
    cat = list(CAT_BASE)
    if task == "refund":
        d = df[(df.year >= 2010) & df.remedy_options.notna()].copy()
        y = d.remedy_refund.astype(int).to_numpy()
        num = num + ["log10_units"]
        note = ("2010-2026, records with a populated remedy field; every "
                "remedy-derived column is excluded from X.")
    else:
        d = df[df.log10_units.notna()].copy()
        y = d.log10_units.to_numpy(float)
        note = ("all records with a positive units count; remedy columns "
                "excluded because the field is empty before 2010.")
    X = d[num + cat].copy()
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    return X, y, d, num, cat, note


def make_pre(num, cat, scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [("num", Pipeline(num_steps), num),
         ("cat", Pipeline([("impute", SimpleImputer(strategy="constant",
                                                    fill_value="missing")),
                           ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                 sparse_output=False,
                                                 min_frequency=3))]), cat)],
        remainder="drop", verbose_feature_names_out=False)


def model_space(kind: str, num, cat, seed: int, light: bool = False):
    clf = kind == "classification"
    n_tree = stats.randint(30, 80) if light else stats.randint(150, 500)
    n_round = stats.randint(30, 80) if light else stats.randint(60, 400)
    sp = {}

    lin = LogisticRegression(solver="saga", max_iter=4000, tol=1e-3,
                             random_state=seed) if clf else \
        ElasticNet(max_iter=20000, tol=1e-4, random_state=seed)
    sp["elastic_net"] = (
        Pipeline([("pre", make_pre(num, cat, scale=True)), ("est", lin)]),
        ({"est__C": stats.loguniform(1e-3, 1e2),
          "est__l1_ratio": stats.uniform(0, 1),
          "est__class_weight": [None, "balanced"]} if clf else
         {"est__alpha": stats.loguniform(1e-4, 1e1),
          "est__l1_ratio": stats.uniform(0.01, 0.99)}))

    rf = (RandomForestClassifier(random_state=seed, n_jobs=1) if clf else
          RandomForestRegressor(random_state=seed, n_jobs=1))
    rf_grid = {"est__n_estimators": n_tree,
               "est__max_depth": [None, 3, 4, 6, 8, 12, 20],
               "est__min_samples_leaf": stats.randint(1, 20),
               "est__min_samples_split": stats.randint(2, 20),
               "est__max_features": stats.uniform(0.1, 0.8)}
    if clf:
        rf_grid["est__class_weight"] = [None, "balanced",
                                        "balanced_subsample"]
    sp["random_forest"] = (
        Pipeline([("pre", make_pre(num, cat, scale=False)), ("est", rf)]),
        rf_grid)

    hgb = (HistGradientBoostingClassifier(random_state=seed,
                                          early_stopping=False) if clf else
           HistGradientBoostingRegressor(random_state=seed,
                                         early_stopping=False))
    sp["hist_gradient_boosting"] = (
        Pipeline([("pre", make_pre(num, cat, scale=False)), ("est", hgb)]),
        {"est__learning_rate": stats.loguniform(0.01, 0.4),
         "est__max_iter": n_round,
         "est__max_leaf_nodes": stats.randint(3, 31),
         "est__min_samples_leaf": stats.randint(3, 40),
         "est__l2_regularization": stats.loguniform(1e-4, 1e1),
         "est__max_features": stats.uniform(0.4, 0.6)})

    if clf:
        sp["dummy_stratified"] = (
            Pipeline([("pre", make_pre(num, cat, scale=False)),
                      ("est", DummyClassifier(strategy="stratified",
                                              random_state=seed))]), {})
        sp["dummy_prior"] = (
            Pipeline([("pre", make_pre(num, cat, scale=False)),
                      ("est", DummyClassifier(strategy="prior"))]), {})
    else:
        sp["dummy_stratified"] = (
            Pipeline([("pre", make_pre(num, cat, scale=False)),
                      ("est", DummyRegressor(strategy="median"))]), {})
        sp["dummy_prior"] = (
            Pipeline([("pre", make_pre(num, cat, scale=False)),
                      ("est", DummyRegressor(strategy="mean"))]), {})
    return sp


def fit_one(pipe, grid, X, y, cfg, kind, name, seed):
    if not grid:
        est = clone(pipe).fit(X, y)
        return est, {}, 1
    n_iter = cfg["n_iter"][name]
    if kind == "classification":
        inner = StratifiedKFold(cfg["inner_folds"], shuffle=True,
                                random_state=seed)
        scoring = "roc_auc"
    else:
        inner = KFold(cfg["inner_folds"], shuffle=True, random_state=seed)
        scoring = "r2"
    search = RandomizedSearchCV(clone(pipe), grid, n_iter=n_iter, cv=inner,
                                scoring=scoring, n_jobs=2, refit=True,
                                random_state=seed, error_score=np.nan)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, n_iter


def predict_scores(est, X, kind):
    if kind == "classification":
        p = est.predict_proba(X)[:, 1]
        return np.clip(p, 1e-6, 1 - 1e-6)
    return est.predict(X)


def score_row(y, p, kind):
    if kind == "classification":
        yhat = (p >= 0.5).astype(int)
        out = {"auc": (roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan),
               "average_precision": (average_precision_score(y, p)
                                     if len(np.unique(y)) > 1 else np.nan),
               "balanced_accuracy": balanced_accuracy_score(y, yhat),
               "accuracy": accuracy_score(y, yhat),
               "brier": brier_score_loss(y, p),
               "log_loss": log_loss(y, p, labels=[0, 1])}
    else:
        out = {"r2": r2_score(y, p), "mae": mean_absolute_error(y, p),
               "rmse": root_mean_squared_error(y, p),
               "spearman": float(stats.spearmanr(y, p).statistic
                                 if np.std(p) > 0 else np.nan)}
    return out


CLF_METRICS = ["auc", "average_precision", "balanced_accuracy", "accuracy",
               "brier", "log_loss"]
REG_METRICS = ["r2", "mae", "rmse", "spearman"]


def nested_cv(X, y, space, cfg, kind, tag):
    rows, oof, params = [], [], []
    metrics = CLF_METRICS if kind == "classification" else REG_METRICS
    for name, (pipe, grid) in space.items():
        t0 = time.time()
        for rep in range(cfg["outer_repeats"]):
            if kind == "classification":
                outer = StratifiedKFold(cfg["outer_folds"], shuffle=True,
                                        random_state=SEED + rep)
                splits = outer.split(X, y)
            else:
                outer = KFold(cfg["outer_folds"], shuffle=True,
                              random_state=SEED + rep)
                splits = outer.split(X)
            for fold, (tr, te) in enumerate(splits):
                est, best, n_cand = fit_one(
                    pipe, grid, X.iloc[tr], y[tr], cfg, kind, name,
                    SEED + 100 * rep + fold)
                p = predict_scores(est, X.iloc[te], kind)
                r = score_row(y[te], p, kind)
                rows.append({"model": name, "repeat": rep, "fold": fold,
                             "n_train": len(tr), "n_test": len(te), **r})
                oof.append(pd.DataFrame({"model": name, "repeat": rep,
                                         "fold": fold, "idx": X.index[te],
                                         "y": y[te], "pred": p}))
                params.append({"model": name, "repeat": rep, "fold": fold,
                               "n_candidates": n_cand,
                               **{k.replace("est__", ""): v
                                  for k, v in best.items()}})
        el = time.time() - t0
        sub = pd.DataFrame([r for r in rows if r["model"] == name])
        key = metrics[0]
        print(f"  [{tag}] {name:<24s} {key} = {sub[key].mean():.3f} "
              f"+/- {sub[key].std(ddof=1):.3f}   ({el:.0f}s)")
    return (pd.DataFrame(rows), pd.concat(oof, ignore_index=True),
            pd.DataFrame(params))


def temporal_split(X, y, years, space, cfg, kind, cut=2019):
    tr = np.where(years <= cut)[0]
    te = np.where(years > cut)[0]
    rows, fitted, preds = [], {}, {}
    for name, (pipe, grid) in space.items():
        est, best, _ = fit_one(pipe, grid, X.iloc[tr], y[tr], cfg, kind, name,
                               SEED + 7)
        p = predict_scores(est, X.iloc[te], kind)
        rows.append({"model": name, "n_train": len(tr), "n_test": len(te),
                     **score_row(y[te], p, kind),
                     **{"best_" + k.replace("est__", ""): v
                        for k, v in best.items()}})
        fitted[name] = est
        preds[name] = p
    return pd.DataFrame(rows), fitted, preds, tr, te


def perm_importance(pipe, grid, X, y, cfg, kind, model_name, seed=SEED):
    scoring = "roc_auc" if kind == "classification" else "r2"
    if kind == "classification":
        cv = StratifiedKFold(cfg["perm_folds"], shuffle=True, random_state=seed)
        splits = list(cv.split(X, y))
    else:
        cv = KFold(cfg["perm_folds"], shuffle=True, random_state=seed)
        splits = list(cv.split(X))
    per_col: dict[str, list] = {c: [] for c in X.columns}
    base = []
    for fold, (tr, te) in enumerate(splits):
        est, _, _ = fit_one(pipe, grid, X.iloc[tr], y[tr], cfg, kind,
                            model_name, seed + fold)
        p = predict_scores(est, X.iloc[te], kind)
        base.append(score_row(y[te], p, kind)["auc" if kind == "classification"
                                              else "r2"])
        r = permutation_importance(est, X.iloc[te], y[te], scoring=scoring,
                                   n_repeats=cfg["perm_repeats"],
                                   random_state=seed + fold, n_jobs=1)
        for j, c in enumerate(X.columns):
            per_col[c].extend(r.importances[j].tolist())
    rng = np.random.default_rng(seed)
    rows = []
    for c, vals in per_col.items():
        v = np.asarray(vals, dtype=float)
        boot = rng.choice(v, size=(1000, len(v)), replace=True).mean(axis=1)
        rows.append({"feature": c, "label": FEATURE_LABELS.get(c, c),
                     "importance": float(v.mean()),
                     "sd": float(v.std(ddof=1)),
                     "lo95": float(np.percentile(boot, 2.5)),
                     "hi95": float(np.percentile(boot, 97.5)),
                     "n_draws": int(len(v)),
                     "p_gt_zero": float((v > 0).mean())})
    out = pd.DataFrame(rows).sort_values("importance", ascending=False)
    out["baseline_score"] = float(np.mean(base))
    return out.reset_index(drop=True)


def pdp_ice(est, X, features, kind, cfg, seed=SEED):
    rng = np.random.default_rng(seed)
    n_ice = min(cfg["ice_lines"], len(X))
    ice_rows_idx = rng.choice(len(X), size=n_ice, replace=False)
    pdp_rows, ice_rows = [], []
    for f in features:
        col = X[f]
        if pd.api.types.is_numeric_dtype(col):
            v = col.dropna().to_numpy(float)
            lo, hi = np.percentile(v, [2.5, 97.5])
            grid = np.unique(np.round(np.linspace(lo, hi, cfg["grid_res"]), 6))
            kindf, labels = "numeric", grid
        else:
            vc = col.value_counts()
            grid = (vc[vc >= 5].index.tolist() or vc.index.tolist())[:6]
            kindf, labels = "categorical", grid
        curves = np.empty((len(X), len(grid)))
        for gi, g in enumerate(grid):
            Xg = X.copy()
            Xg[f] = g
            curves[:, gi] = predict_scores(est, Xg, kind)
        mean = curves.mean(axis=0)
        for gi, g in enumerate(grid):
            pdp_rows.append({
                "feature": f, "label": FEATURE_LABELS.get(f, f), "kind": kindf,
                "grid_index": gi, "grid_value": g,
                "pd_mean": float(mean[gi]),
                "pd_p10": float(np.percentile(curves[:, gi], 10)),
                "pd_p90": float(np.percentile(curves[:, gi], 90))})
        for li, ri in enumerate(ice_rows_idx):
            for gi, g in enumerate(grid):
                ice_rows.append({"feature": f, "line": li, "grid_index": gi,
                                 "grid_value": g,
                                 "pred": float(curves[ri, gi])})
    return pd.DataFrame(pdp_rows), pd.DataFrame(ice_rows)


def linear_coefficients(pipe, grid, X, y, cfg, kind, seed=SEED):
    est, best, _ = fit_one(pipe, grid, X, y, cfg, kind, "elastic_net", seed)
    pre = est.named_steps["pre"]
    Z = pre.transform(X)
    names = list(pre.get_feature_names_out())
    final = clone(est.named_steps["est"])
    final.fit(Z, y)
    coef = np.asarray(final.coef_).ravel()

    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.empty((cfg["n_boot"], len(coef)))
    for b in range(cfg["n_boot"]):
        if kind == "classification":
            idx = np.concatenate([
                rng.choice(np.where(y == c)[0], size=int((y == c).sum()),
                           replace=True) for c in np.unique(y)])
        else:
            idx = rng.integers(0, n, n)
        m = clone(final)
        try:
            m.fit(Z[idx], y[idx])
            boots[b] = np.asarray(m.coef_).ravel()
        except Exception:
            boots[b] = np.nan
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    nz = np.nanmean(np.abs(boots) > 1e-8, axis=0)
    rows = []
    for j, nm in enumerate(names):
        rows.append({"term": nm, "coef": float(coef[j]),
                     "lo95": float(lo[j]), "hi95": float(hi[j]),
                     "selected_frac": float(nz[j]),
                     "boot_mean": float(np.nanmean(boots[:, j])),
                     "excludes_zero": bool(lo[j] > 0 or hi[j] < 0)})
        if kind == "classification":
            rows[-1].update({"odds_ratio": float(np.exp(coef[j])),
                             "or_lo95": float(np.exp(lo[j])),
                             "or_hi95": float(np.exp(hi[j]))})
    out = pd.DataFrame(rows)
    out["abs_coef"] = out.coef.abs()
    out = out.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    return out, {k.replace("est__", ""): v for k, v in best.items()}


def calibration(y, p, n_bins=8):
    q = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    b = np.digitize(p, q[1:-1], right=True)
    rows, ece = [], 0.0
    for k in range(n_bins):
        m = b == k
        if not m.any():
            continue
        obs, pred = float(y[m].mean()), float(p[m].mean())
        lo, hi = _wilson(int(y[m].sum()), int(m.sum()))
        rows.append({"bin": k, "n": int(m.sum()), "pred_mean": pred,
                     "obs_rate": obs, "lo95": lo, "hi95": hi})
        ece += m.mean() * abs(obs - pred)
    return pd.DataFrame(rows), float(ece)


def _wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (max(c - h, 0.0), min(c + h, 1.0))


def agg_metrics(folds: pd.DataFrame, metrics) -> pd.DataFrame:
    g = folds.groupby("model")
    out = []
    for name, sub in g:
        row = {"model": name, "n_folds": int(len(sub))}
        for m in metrics:
            row[f"{m}_mean"] = float(sub[m].mean())
            row[f"{m}_sd"] = float(sub[m].std(ddof=1))
            row[f"{m}_lo"] = float(sub[m].mean() - 1.96 * sub[m].std(ddof=1)
                                   / np.sqrt(len(sub)))
            row[f"{m}_hi"] = float(sub[m].mean() + 1.96 * sub[m].std(ddof=1)
                                   / np.sqrt(len(sub)))
        out.append(row)
    return pd.DataFrame(out)


def run_task(df, task, kind, cfg, text_feats):
    X, y, d, num, cat, note = task_matrices(df, task, text_feats)
    years = d.year.to_numpy(int)
    metrics = CLF_METRICS if kind == "classification" else REG_METRICS
    space = model_space(kind, num, cat, SEED, light=cfg.get("light", False))
    print(f"[16] task {task}: n = {len(y)}, "
          f"{'positives ' + str(int(y.sum())) if kind == 'classification' else 'mean log10 units %.2f' % y.mean()}"
          f", features {len(num)} numeric + {len(cat)} categorical")

    folds, oof, params = nested_cv(X, y, space, cfg, kind, task)
    comp = agg_metrics(folds, metrics)

    temp, fitted, tpreds, tr_idx, te_idx = temporal_split(
        X, y, years, space, cfg, kind)

    key = "auc_mean" if kind == "classification" else "r2_mean"
    real = comp[~comp.model.str.startswith("dummy")]
    best_name = real.sort_values(key, ascending=False).model.iloc[0]
    pipe, grid = space[best_name]

    imp = perm_importance(pipe, grid, X, y, cfg, kind, best_name)

    best_est, best_params, _ = fit_one(pipe, grid, X, y, cfg, kind, best_name,
                                       SEED + 3)
    top5 = imp.feature.head(5).tolist()
    pdp, ice = pdp_ice(best_est, X, top5, kind, cfg)

    lpipe, lgrid = space["elastic_net"]
    coefs, lin_params = linear_coefficients(lpipe, lgrid, X, y, cfg, kind)

    extra = {}
    if kind == "classification":
        pooled = (oof[oof.model == best_name]
                  .groupby("idx", observed=True)
                  .agg(y=("y", "first"), pred=("pred", "mean")).reset_index())
        cal, ece = calibration(pooled.y.to_numpy(), pooled.pred.to_numpy())
        yhat = (pooled.pred >= 0.5).astype(int)
        cm = confusion_matrix(pooled.y, yhat, labels=[0, 1])
        all2010 = df[df.year >= 2010]
        extra = {"calibration": cal.to_dict(orient="records"), "ece": ece,
                 "confusion_matrix": cm.tolist(),
                 "confusion_labels": ["no refund", "refund"],
                 "pooled_auc": float(roc_auc_score(pooled.y, pooled.pred)),
                 "pooled_brier": float(brier_score_loss(pooled.y, pooled.pred)),
                 "n_2010plus_all": int(len(all2010)),
                 "n_dropped_missing_remedy_field":
                     int(all2010.remedy_options.isna().sum()),
                 "dropped_years": {int(k): int(v) for k, v in
                                   all2010.loc[all2010.remedy_options.isna(),
                                               "year"].value_counts().items()},
                 "base_rate_if_missing_scored_negative":
                     float(all2010.remedy_refund.mean())}
        cal.to_csv(RES / "calibration_refund.csv", index=False)
        pooled = pooled.merge(
            d[["year", "sales_channel", "hazard_category"]].reset_index()
            .rename(columns={"index": "idx"}), on="idx", how="left")
        pooled.to_csv(RES / "oof_predictions_refund.csv", index=False)
    else:
        pooled = (oof[oof.model == best_name]
                  .groupby("idx", observed=True)
                  .agg(y=("y", "first"), pred=("pred", "mean")).reset_index())
        pooled["resid"] = pooled.y - pooled.pred
        pooled["split"] = "oof_random_cv"
        t = pd.DataFrame({"idx": X.index[te_idx], "y": y[te_idx],
                          "pred": tpreds[best_name]})
        t["resid"] = t.y - t.pred
        t["split"] = "temporal_2020plus"
        resid = pd.concat([pooled, t], ignore_index=True)
        resid = resid.merge(d[["year", "sales_channel"]].reset_index()
                            .rename(columns={"index": "idx"}),
                            on="idx", how="left")
        resid.to_csv(RES / "residuals_units.csv", index=False)
        extra = {"oof_r2": float(r2_score(pooled.y, pooled.pred)),
                 "oof_mae": float(mean_absolute_error(pooled.y, pooled.pred))}

    suffix = "refund" if task == "refund" else "units"
    folds.to_csv(RES / f"cv_folds_{suffix}.csv", index=False)
    imp.to_csv(RES / f"permutation_importance_{suffix}.csv", index=False)
    pdp.to_csv(RES / f"pdp_{suffix}.csv", index=False)
    ice.to_csv(RES / f"ice_{suffix}.csv", index=False)
    coefs.to_csv(RES / f"coefficients_{suffix}.csv", index=False)
    params.to_csv(RES / f"search_candidates_{suffix}.csv", index=False)

    tkey = "auc" if kind == "classification" else "r2"
    gap = []
    for _, r in temp.iterrows():
        cv_mean = float(comp.loc[comp.model == r.model, f"{tkey}_mean"].iloc[0])
        gap.append({"model": r.model, "cv": cv_mean, "temporal": float(r[tkey]),
                    "gap": cv_mean - float(r[tkey])})

    summary = {
        "task": task, "kind": kind, "note": note,
        "n": int(len(y)), "n_features_numeric": len(num),
        "n_features_categorical": len(cat),
        "features_numeric": num, "features_categorical": cat,
        "year_range": [int(years.min()), int(years.max())],
        "outcome": ("share_refund" if kind == "classification"
                    else "mean_log10_units"),
        "outcome_value": float(y.mean()),
        "cv_design": {"outer_folds": cfg["outer_folds"],
                      "outer_repeats": cfg["outer_repeats"],
                      "inner_folds": cfg["inner_folds"],
                      "n_candidates": cfg["n_iter"],
                      "total_outer_fits": int(len(folds))},
        "model_comparison": comp.to_dict(orient="records"),
        "temporal_split": temp.to_dict(orient="records"),
        "temporal_n_train": int(len(tr_idx)), "temporal_n_test": int(len(te_idx)),
        "cv_vs_temporal": gap,
        "best_model": best_name,
        "best_params_full_fit": {k.replace("est__", ""): (
            float(v) if isinstance(v, (int, float, np.floating, np.integer))
            else v) for k, v in best_params.items()},
        "linear_best_params": {k: (float(v) if isinstance(
            v, (int, float, np.floating, np.integer)) else v)
            for k, v in lin_params.items()},
        "permutation_top10": imp.head(10).to_dict(orient="records"),
        "permutation_baseline": float(imp.baseline_score.iloc[0]),
        "pdp_features": top5,
        "coefficients_top15": coefs.head(15).to_dict(orient="records"),
        "coefficients_excluding_zero": int(coefs.excludes_zero.sum()),
        "n_terms": int(len(coefs)),
        **extra,
    }
    return summary, comp, temp, folds


def main(quick: bool = False, mid: bool = False):
    mode = "quick" if quick else ("mid" if mid else "production")
    cfg = SETTINGS[mode]
    t0 = time.time()
    df, meta = build_frame()
    print(f"[16] mode = {mode}; raw JSON present: {meta['raw_json_present']}")

    a_sum, a_comp, a_temp, a_folds = run_task(df, "refund", "classification",
                                              cfg, meta["text_features"])
    b_sum, b_comp, b_temp, b_folds = run_task(df, "units", "regression",
                                              cfg, meta["text_features"])

    a_comp["task"], b_comp["task"] = "refund", "units"
    a_temp["task"], b_temp["task"] = "refund", "units"
    pd.concat([a_comp, b_comp], ignore_index=True).to_csv(
        RES / "model_comparison.csv", index=False)
    pd.concat([a_temp, b_temp], ignore_index=True).to_csv(
        RES / "temporal_split.csv", index=False)

    elapsed = time.time() - t0
    summary = {
        "analysis": "16_ml_remedy_and_scale",
        "mode": mode,
        "settings": {k: v for k, v in cfg.items()},
        "seconds": elapsed,
        "data": meta,
        "task_refund": a_sum,
        "task_units": b_sum,
    }
    D.write_json(summary, RES / "summary.json")

    ba = a_sum["best_model"]
    ga = [g for g in a_sum["cv_vs_temporal"] if g["model"] == ba][0]
    bb = b_sum["best_model"]
    gb = [g for g in b_sum["cv_vs_temporal"] if g["model"] == bb][0]
    print(f"[16] refund: best {ba}; AUC CV {ga['cv']:.3f} -> temporal "
          f"{ga['temporal']:.3f} (gap {ga['gap']:+.3f}); "
          f"top feature {a_sum['permutation_top10'][0]['feature']}")
    print(f"[16] units:  best {bb}; R2 CV {gb['cv']:.3f} -> temporal "
          f"{gb['temporal']:.3f} (gap {gb['gap']:+.3f}); "
          f"top feature {b_sum['permutation_top10'][0]['feature']}")
    print(f"[16] done in {elapsed / 60:.1f} min")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke run, < 90 s")
    ap.add_argument("--mid", action="store_true", help="mid-size run")
    main(**vars(ap.parse_args()))
