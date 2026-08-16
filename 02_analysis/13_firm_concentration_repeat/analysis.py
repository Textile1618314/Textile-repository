from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import data as D
from _common import mcmc as M

RES = D.results_dir(__file__)

ERA_SPLIT = 2010
SIM_THRESHOLD = 0.60
MAX_DAYS_APART = 730
RARE_DF_FRAC = 0.02
RARE_DF_MIN = 3
PRODUCT_STOP = set("""
nightgown nightgowns nightdress pajama pajamas pyjama pyjamas sleepwear sleeper
sleepers robe robes bathrobe bathrobes loungewear lounge sack sacks swaddle
blanket blankets sweatshirt sweatshirts sweater sweaters hoodie hoodies jacket
jackets coat coats parka anorak vest vests shirt shirts blouse blouses top tops
pant pants trousers jeans shorts skirt skirts dress dresses romper rompers
onesie bodysuit jumpsuit overalls leggings jumper jumpers suit suits set sets
shoe shoes boot boots sandal sandals sneaker sneakers slipper slippers footwear
scarf scarves glove gloves mitten mittens hat hats beanie sock socks tights
jewelry necklace bracelet earrings charm keychain backpack purse handbag
swimwear swimsuit bikini costume costumes towel towels quilt comforter bedding
pillow curtain drape sheet sheets thermal thermals underwear boxers briefs
girls girl boys boy kids kid child children childrens infant infants toddler
toddlers youth women womens woman men mens man baby babies unisex junior
cotton fleece flannel knit knitted woven denim velour terry organic bamboo
hooded drawstring drawstrings zip zippered footed printed print holiday winter
summer spring fall christmas halloween two piece three long short sleeve
sleeves style styles assorted various models model number numbers heated electric weighted infrared cordless rechargeable
chenille velour wool silk satin polyester nylon leather suede metal plastic
wooden waist hood hoods neck cuff cuffs collar zipper zippers snap snaps button
buttons pocket pockets garment garments apparel clothing outfit outfits
teddy bear bears doll dolls toy toys necklace necklaces jewellery athletic
safety water sport sports novelty decorative
""".split())

_CPSC_JOINT = [
    re.compile(r"^cpsc\s*(?:,|and)?\s*(.{2,70}?)\s+announces?\b", re.I),
    re.compile(r"^(.{2,70}?)\s+and\s+cpsc\s+announces?\b", re.I),
]
_ACTIVE = re.compile(r"^(.{2,70}?)\s+(?:recalls|recalling|recall\b(?!\s*(?:of|to))"
                     r"|expands?\s+recall|reannounces?\s+recall|issues?\s+recall"
                     r"|stops?\s+(?:importing|selling|sale)|to\s+halt\s+sale)", re.I)
_BY = re.compile(r"\brecalled\s+by\s+(.{2,70}?)(?:\s+due to\b|\s+because\b|[;,]|$)", re.I)
_IMPORTED = re.compile(r"\bimported\s+by\s+(.{2,70}?)(?:\s+due to\b|[;,]|$)", re.I)
_SOLD_BY = re.compile(r"\bsold\s+(?:exclusively\s+)?(?:on|at|in|through)?\s*"
                      r"(?:[\w.\- ]{0,20}?)?\bby\s+(.{2,70}?)(?:\s+due to\b|[;,]|$)", re.I)
_MANUF = re.compile(r"\bmanufactured\s+by\s+(.{2,70}?)(?:\s+due to\b|[;,]|$)", re.I)
_TRAILING_JUNK = re.compile(
    r"\s*\(recall alert\)|\s*\.com\b|\bexclusively\b|\bannounces?\b|\bcpsc\b", re.I)
_NAME_TAIL = re.compile(
    r"\s+(?:due\b|for\b|because\b|after\b|over\b|amid\b|recalled\b|recalls\b|"
    r"sold\b|and\s+sold\b|announce\w*|to\s+prevent\b|linked\b)", re.I)
_NOT_A_FIRM = re.compile(
    r"hazard|violat\w*|standards?\b|prompts?\b|risk of|recall|warning|"
    r"consumers?\b|products sold|^the\b.{0,3}$", re.I)
_BOILER_TOKENS = {"cpsc", "announce", "announces", "announced", "and", "recall",
                  "recalls", "expands", "the", "of", "inc", "llc"}
_LEGAL = re.compile(r"\b(?:inc|llc|l\.l\.c|ltd|co|corp|corporation|company|plc|"
                    r"gmbh|sa|srl|lp|llp|pty|limited)\b\.?", re.I)
_GENERIC_TOKENS = {"store", "stores", "shop", "shops", "group", "brands", "brand",
                   "usa", "us", "america", "american", "international", "global",
                   "industries", "enterprises", "imports", "import", "importers",
                   "distribution", "distributors", "sales", "products", "retail",
                   "holdings", "apparel", "clothing", "fashion", "fashions",
                   "kids", "baby", "children", "childrens", "trading", "north",
                   "the", "and", "of", "for", "sports", "design", "designs",
                   "shoes", "shoe", "boots", "footwear", "toys", "jewelry",
                   "textiles", "home", "outfitters", "wear", "importer",
                   "collection", "collections", "creations", "originals"}


def extract_firm_v2(title: str) -> tuple[str | None, str]:
    t = re.sub(r"\s+", " ", str(title or "")).strip()
    for rx, tag in ([(r, "cpsc_joint") for r in _CPSC_JOINT] +
                    [(_ACTIVE, "active"), (_BY, "recalled_by"),
                     (_IMPORTED, "imported_by"), (_SOLD_BY, "sold_by"),
                     (_MANUF, "manufactured_by")]):
        m = rx.search(t)
        if m:
            name = _TRAILING_JUNK.sub(" ", m.group(1))
            name = _NAME_TAIL.split(name)[0]
            name = re.sub(r"\s+", " ", name).strip(" .,-–—:;&")
            if (2 <= len(name) <= 70 and not re.fullmatch(r"(?i)cpsc", name)
                    and not _NOT_A_FIRM.search(name)):
                return name, tag
    return None, "none"


def norm_key(name: str | None) -> str | None:
    k = D.firm_key(name)
    if not k:
        return None
    toks = [t for t in k.split() if t not in _BOILER_TOKENS]
    return " ".join(toks) or k


def merge_key(name: str | None) -> str | None:
    if not name or (isinstance(name, float) and np.isnan(name)):
        return None
    k = _LEGAL.sub(" ", str(name).lower())
    k = re.sub(r"[^a-z0-9 ]", " ", k)
    toks = [t for t in k.split() if t not in {"cpsc", "announce", "announces"}]
    return " ".join(toks) or None


def is_joint_name(name: str | None) -> bool:
    t = str(name or "").lower()
    return bool(re.search(r"\s+and\s+|&|,", t))


def token_jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def fuzzy_merge(keys: list[str], counts: dict,
                joint: dict | None = None) -> tuple[dict, list[dict]]:
    joint = joint or {}
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if counts.get(ra, 0) < counts.get(rb, 0):
                ra, rb = rb, ra
            parent[rb] = ra

    toks = {k: set(k.split()) for k in keys}
    tok_df: dict[str, int] = {}
    for k in keys:
        if joint.get(k):
            continue
        for w in toks[k]:
            tok_df[w] = tok_df.get(w, 0) + 1
    merges = []
    for i, a in enumerate(keys):
        ta = toks[a]
        for b in keys[i + 1:]:
            tb = toks[b]
            if not (ta & tb):
                continue
            inter = len(ta & tb)
            contain = inter / min(len(ta), len(tb))
            jac = token_jaccard(ta, tb)
            ratio = difflib.SequenceMatcher(None, a, b).ratio()
            rule = None
            informative = {w for w in (ta & tb) - _GENERIC_TOKENS
                           if tok_df.get(w, 99) <= 3}
            if (contain == 1.0 and informative and min(len(a), len(b)) >= 4
                    and not (joint.get(a) or joint.get(b))):
                rule = "containment"
            elif jac >= 0.60 and ratio >= 0.80:
                rule = "jaccard+ratio"
            elif ratio >= 0.92:
                rule = "ratio"
            if rule:
                union(a, b)
                merges.append({"key_a": a, "key_b": b, "rule": rule,
                               "jaccard": round(jac, 3), "ratio": round(ratio, 3),
                               "containment": round(contain, 3),
                               "n_a": counts.get(a, 0), "n_b": counts.get(b, 0)})
    return {k: find(k) for k in keys}, merges


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x)) / (n * np.sum(x)) - (n + 1) / n)


def lorenz(x: np.ndarray) -> pd.DataFrame:
    x = np.sort(np.asarray(x, dtype=float))
    cum = np.concatenate([[0.0], np.cumsum(x) / x.sum()])
    frac = np.arange(len(x) + 1) / len(x)
    return pd.DataFrame({"frac_firms": frac, "frac_recalls": cum})


def gini_bootstrap(x: np.ndarray, n_boot: int, seed=7) -> dict:
    rng = np.random.default_rng(seed)
    g = gini(x)
    draws = np.array([gini(rng.choice(x, size=len(x), replace=True))
                      for _ in range(n_boot)])
    return {"gini": g, "lo95": float(np.percentile(draws, 2.5)),
            "hi95": float(np.percentile(draws, 97.5)),
            "n_firms": int(len(x)), "n_recalls": int(x.sum()),
            "_draws": draws}


def kmeans_bic(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    n, d = X.shape
    k = centers.shape[0]
    resid = np.sum((X - centers[labels]) ** 2)
    denom = max(n - k, 1) * d
    var = max(resid / denom, 1e-12)
    ll = 0.0
    for j in range(k):
        nj = int(np.sum(labels == j))
        if nj == 0:
            continue
        ll += (nj * np.log(nj) - nj * np.log(n)
               - nj * d / 2 * np.log(2 * np.pi * var)
               - (nj - 1) * d / 2)
    p = k * (d + 1)
    return float(-2 * ll + p * np.log(n))


def name_cluster(prof: pd.Series) -> str:
    repeat = prof.n_recalls_mean >= 1.8
    kids = prof.share_childrens >= 0.60
    flam = prof.share_flammability >= 0.55
    if prof.share_online_only >= 0.50:
        return ("Marketplace importer, online only" if prof.mean_last_year >= 2015
                else "Online-only seller")
    if prof.n_recalls_mean >= 3.0 or prof.median_units >= 60000:
        return "National retailer / mass brand"
    if prof.median_price >= 100:
        return "Premium / specialist brand"
    if repeat:
        return ("Children's brand, repeat recalls" if kids
                else "Repeat-recall brand")
    if kids:
        return ("Children's supplier, flammability" if flam
                else "Children's supplier, one recall")
    return ("Adult apparel, flammability" if flam
            else "Adult apparel, one recall")


def fit_count_models(y: np.ndarray, quick: bool):
    n_draws, n_tune, n_chains = (600, 600, 2) if quick else (6000, 6000, 4)
    y = np.asarray(y, dtype=float)

    def ztp_loglik(theta):
        mu = np.exp(np.clip(theta[0], -10, 10))
        return M.log_poisson(y, mu) - np.log1p(-np.exp(-mu))

    def ztnb_loglik(theta):
        mu = np.exp(np.clip(theta[0], -10, 10))
        phi = np.exp(np.clip(theta[1], -10, 10))
        p0 = np.exp(phi * (np.log(phi) - np.log(phi + mu)))
        return M.log_nb(y, mu, phi) - np.log(np.clip(1 - p0, 1e-300, None))

    def lp_ztp(theta):
        v = float(np.sum(ztp_loglik(theta)) + M.log_normal(theta[0], 0.0, 1.5))
        return v if np.isfinite(v) else -np.inf

    def lp_ztnb(theta):
        v = float(np.sum(ztnb_loglik(theta))
                  + M.log_normal(theta[0], 0.0, 1.5)
                  + M.log_normal(theta[1], 0.0, 1.5))
        return v if np.isfinite(v) else -np.inf

    print("  [nb] zero-truncated Poisson")
    p_ztp = M.sample(lp_ztp, [np.log(y.mean())], n_draws=n_draws, n_tune=n_tune,
                     n_chains=n_chains, seed=21, names=["log_mu"], verbose=True)
    print("  [nb] zero-truncated negative binomial")
    p_ztnb = M.sample(lp_ztnb, [np.log(y.mean()), 0.0], n_draws=n_draws,
                      n_tune=n_tune, n_chains=n_chains, seed=22,
                      names=["log_mu", "log_phi"], verbose=True)

    def pointwise(post, fn, n_keep=1500):
        f = post.flat()
        idx = np.linspace(0, len(f) - 1, min(n_keep, len(f))).astype(int)
        return np.array([fn(f[i]) for i in idx])

    ll_ztp = pointwise(p_ztp, ztp_loglik)
    ll_ztnb = pointwise(p_ztnb, ztnb_loglik)
    comparison = M.compare({"zero_truncated_poisson": ll_ztp,
                            "zero_truncated_negbin": ll_ztnb})

    rng = np.random.default_rng(3)
    n_firms = len(y)
    ks = [2, 3, 5, 7]
    ppc = {"observed": {f"ge_{k}": int((y >= k).sum()) for k in ks}}
    for label, post in [("zero_truncated_poisson", p_ztp),
                        ("zero_truncated_negbin", p_ztnb)]:
        f = post.flat()
        sel = rng.choice(len(f), size=min(600, len(f)), replace=False)
        tallies = {f"ge_{k}": [] for k in ks}
        for i in sel:
            mu = float(np.exp(f[i][0]))
            if label.endswith("negbin"):
                phi = float(np.exp(f[i][1]))
                p1 = 1.0 - float(np.exp(phi * (np.log(phi) - np.log(phi + mu))))
            else:
                p1 = 1.0 - float(np.exp(-mu))
            if not np.isfinite(p1) or p1 <= 1e-6:
                continue
            size = int(min(max(n_firms / p1 * 1.6, n_firms), 4e6))
            if label.endswith("negbin"):
                draws = rng.poisson(rng.gamma(phi, mu / phi, size=size))
            else:
                draws = rng.poisson(mu, size=size)
            draws = draws[draws >= 1][:n_firms]
            if len(draws) < n_firms:
                continue
            for k in ks:
                tallies[f"ge_{k}"].append(int((draws >= k).sum()))
        ppc[label] = {k: {"mean": float(np.mean(v)),
                          "lo95": float(np.percentile(v, 2.5)),
                          "hi95": float(np.percentile(v, 97.5)),
                          "n_sims": len(v)}
                      for k, v in tallies.items() if v}
    return p_ztp, p_ztnb, comparison, ppc


_RECALL_TOKEN = re.compile(r"\b(?:recalls?|recalling|recalled)\b", re.I)
_AFTER_STRIP = re.compile(r"^(?:of|to\s+repair|to\s+all|for\s+repair|the)\b[\s:,-]*", re.I)
_PASSIVE_MARK = re.compile(r"^(?:by\b|due\b|because\b|for\b|after\b|over\b|from\b|"
                           r"and\b|on\b|in\b|at\b|;|:|$)", re.I)
_CPSC_ANN = re.compile(r"^cpsc[^a-z]{0,3}\s*(?:and\s+)?.*?\bannounces?\b\s*", re.I)
_TAIL = re.compile(
    r"\s*(?:;|:|\bdue\s+to\b|\bbecause\s+of\b|\bfor\s+\w+\s+hazard\b"
    r"|\bsold\s+(?:exclusively\s+)?(?:at|by|on|in|through)\b|\bimported\s+by\b"
    r"|\bmanufactured\s+by\b|\brecalled\b|\brecall\b|\bviolat\w*"
    r"|\(recall\s+alert\))", re.I)


def product_phrase(title: str) -> str:
    t = re.sub(r"\s+", " ", str(title or "")).strip()
    m = _RECALL_TOKEN.search(t)
    if m:
        after = _AFTER_STRIP.sub("", t[m.end():].strip())
        seg = _CPSC_ANN.sub("", t[:m.start()]) if _PASSIVE_MARK.match(after) else after
    else:
        seg = _CPSC_ANN.sub("", t)
    return re.sub(r"\s+", " ", _TAIL.split(seg)[0]).strip(" .,-–—:’'\"")


def main(quick: bool = False):
    n_boot = 300 if quick else 2000
    df = pd.read_csv(D.V2_CSV)
    df["recall_date"] = pd.to_datetime(df.recall_date, errors="coerce")
    df["era"] = np.where(df.year < ERA_SPLIT, f"pre-{ERA_SPLIT}", f"{ERA_SPLIT}+")
    df["product_phrase"] = df.title.map(product_phrase)

    df["firm_baseline"] = df.title.map(D.extract_firm)
    got = df.title.map(extract_firm_v2)
    df["firm_v2"] = [g[0] for g in got]
    df["firm_pattern"] = [g[1] for g in got]
    df["key_baseline"] = df.firm_baseline.map(D.firm_key)
    df["key_v2"] = df.firm_v2.map(norm_key)
    df["key_merge"] = df.firm_v2.map(merge_key)

    qa = {
        "n_records": int(len(df)),
        "parse_rate_baseline": float(df.key_baseline.notna().mean()),
        "parse_rate_rebuilt": float(df.key_merge.notna().mean()),
        "n_unparsed_baseline": int(df.key_baseline.isna().sum()),
        "n_unparsed_rebuilt": int(df.key_merge.isna().sum()),
        "n_distinct_keys_baseline": int(df.key_baseline.nunique()),
        "pattern_counts": df.firm_pattern.value_counts().to_dict(),
        "n_changed_by_rebuild": int((df.key_baseline != df.key_v2).sum()),
        "n_baseline_keys_containing_cpsc":
            int(df.key_baseline.fillna("").str.contains("cpsc").sum()),
        "unparsed_titles": df.loc[df.key_v2.isna(), "title"].head(20).tolist(),
    }

    parsed = df[df.key_merge.notna()].copy()
    counts = parsed.key_merge.value_counts().to_dict()
    keys = sorted(counts)
    joint = (parsed.groupby("key_merge").firm_v2
             .agg(lambda s: bool(is_joint_name(s.iloc[0]))).to_dict())
    merge_map, merges = fuzzy_merge(keys, counts, joint)
    parsed["firm_id"] = parsed.key_merge.map(merge_map)
    df["firm_id"] = df.key_merge.map(merge_map)

    canon = (parsed.groupby("firm_id").firm_v2
             .agg(lambda s: s.value_counts().idxmax()).to_dict())
    parsed["firm_name"] = parsed.firm_id.map(canon)
    df["firm_name"] = df.firm_id.map(canon)

    merge_df = pd.DataFrame(merges)
    if len(merge_df):
        merge_df["merged_into"] = merge_df.key_a.map(merge_map)
        merge_df["canonical_name"] = merge_df.merged_into.map(canon)
    merge_df.to_csv(RES / "firm_merges.csv", index=False)

    merge_qa = {
        "n_keys_before_merge": len(keys),
        "n_firms_after_merge": int(parsed.firm_id.nunique()),
        "n_merge_pairs": int(len(merge_df)),
        "n_keys_absorbed": len(keys) - int(parsed.firm_id.nunique()),
        "merges_by_rule": (merge_df.rule.value_counts().to_dict()
                           if len(merge_df) else {}),
        "largest_merges": (
            merge_df.sort_values(["n_a", "n_b"], ascending=False)
            .head(12)[["key_a", "key_b", "rule", "ratio", "canonical_name"]]
            .to_dict(orient="records") if len(merge_df) else []),
    }

    g = parsed.groupby("firm_id")
    firm = pd.DataFrame({
        "firm_name": g.firm_name.first(),
        "n_recalls": g.size(),
        "first_year": g.year.min(),
        "last_year": g.year.max(),
        "median_units": g.units.median(),
        "median_price": g.price_usd.median(),
        "share_online_only": g.sales_channel.apply(lambda s: (s == "online_only").mean()),
        "share_childrens": g.is_childrens.apply(lambda s: s.fillna(False).mean()),
        "share_flammability": g.hazard_category.apply(
            lambda s: (s == "flammability_burn").mean()),
        "share_violation": g.is_violation.apply(lambda s: s.fillna(False).mean()),
        "primary_country": g.primary_country.agg(
            lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None),
    }).reset_index()
    firm["span_years"] = firm.last_year - firm.first_year
    firm["mean_year"] = g.year.mean().values

    rep = firm[firm.n_recalls >= 2].sort_values("n_recalls", ascending=False)
    rep.to_csv(RES / "repeat_offenders.csv", index=False)

    concentration = {
        "n_firms": int(len(firm)),
        "n_recalls_attributed": int(firm.n_recalls.sum()),
        "n_firms_ge2": int((firm.n_recalls >= 2).sum()),
        "n_firms_ge3": int((firm.n_recalls >= 3).sum()),
        "n_firms_ge5": int((firm.n_recalls >= 5).sum()),
        "share_recalls_from_repeat_firms":
            float(firm.loc[firm.n_recalls >= 2, "n_recalls"].sum() /
                  firm.n_recalls.sum()),
        "max_recalls_one_firm": int(firm.n_recalls.max()),
        "top10_firms": firm.nlargest(10, "n_recalls")[
            ["firm_name", "n_recalls", "first_year", "last_year"]
        ].to_dict(orient="records"),
        "count_distribution": firm.n_recalls.value_counts().sort_index().to_dict(),
    }

    lor = [lorenz(firm.n_recalls.to_numpy()).assign(era="all")]
    gini_all = gini_bootstrap(firm.n_recalls.to_numpy(), n_boot)
    gini_era, era_draws = {}, {}
    for era in [f"pre-{ERA_SPLIT}", f"{ERA_SPLIT}+"]:
        sub = parsed[parsed.era == era]
        cnt = sub.groupby("firm_id").size().to_numpy()
        if len(cnt) < 5:
            continue
        lor.append(lorenz(cnt).assign(era=era))
        b = gini_bootstrap(cnt, n_boot, seed=13)
        era_draws[era] = b.pop("_draws")
        gini_era[era] = b
        gini_era[era]["share_from_repeat_firms"] = float(
            cnt[cnt >= 2].sum() / cnt.sum())
        gini_era[era]["top5pct_share"] = float(
            np.sort(cnt)[::-1][:max(1, int(round(0.05 * len(cnt))))].sum() / cnt.sum())
    pd.concat(lor, ignore_index=True).to_csv(RES / "lorenz_curves.csv", index=False)

    if len(era_draws) == 2:
        d = era_draws[f"{ERA_SPLIT}+"] - era_draws[f"pre-{ERA_SPLIT}"]
        gini_diff = {
            "difference": float(gini_era[f"{ERA_SPLIT}+"]["gini"] -
                                gini_era[f"pre-{ERA_SPLIT}"]["gini"]),
            "lo95": float(np.percentile(d, 2.5)),
            "hi95": float(np.percentile(d, 97.5)),
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
            "n_boot": n_boot,
        }
    else:
        gini_diff = {"skipped": True}
    gini_all.pop("_draws")

    feat_cols = ["n_recalls", "span_years", "log_units", "log_price",
                 "share_online_only", "share_childrens", "share_flammability"]
    F = firm.copy()
    F["log_units"] = np.log10(F.median_units.replace(0, np.nan))
    F["log_price"] = np.log10(F.median_price.replace(0, np.nan))
    imputed = {c: int(F[c].isna().sum()) for c in ["log_units", "log_price"]}
    F["log_units"] = F.log_units.fillna(F.log_units.median())
    F["log_price"] = F.log_price.fillna(F.log_price.median())
    X = F[feat_cols].to_numpy(float)
    X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)

    ks = range(2, 13)
    sel_rows = []
    fits = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        kb = kmeans_bic(X, km.labels_, km.cluster_centers_)
        row = {"k": k, "kmeans_bic": kb, "kmeans_inertia": float(km.inertia_)}
        for cov in ["full", "diag"]:
            gm = GaussianMixture(n_components=k, covariance_type=cov,
                                 random_state=0, n_init=3,
                                 reg_covar=1e-4).fit(X)
            row[f"gmm_{cov}_bic"] = float(gm.bic(X))
            fits[(k, cov)] = gm
        fits[(k, "kmeans")] = km
        sel_rows.append(row)
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(RES / "cluster_model_selection.csv", index=False)

    best_km_k = int(sel.loc[sel.kmeans_bic.idxmin(), "k"])
    gmm_long = sel.melt(id_vars="k", value_vars=["gmm_full_bic", "gmm_diag_bic"],
                        var_name="cov_type", value_name="bic")
    best_gmm = gmm_long.loc[gmm_long.bic.idxmin()]
    best_gmm_k = int(best_gmm.k)
    best_gmm_cov = "full" if "full" in best_gmm["cov_type"] else "diag"

    bic_curve = sel[f"gmm_{best_gmm_cov}_bic"].to_numpy()
    kk = sel.k.to_numpy()
    gains = -np.diff(bic_curve)
    elbow_k = int(kk[-1])
    if len(gains):
        thresh = 0.10 * gains.max()
        for i in range(len(gains)):
            if gains[i] < thresh:
                elbow_k = int(kk[i])
                break
        else:
            elbow_k = int(kk[-1])
    report_k = elbow_k
    if (report_k, best_gmm_cov) not in fits:
        report_k = best_gmm_k

    F["cluster_kmeans"] = fits[(report_k, "kmeans")].labels_
    F["cluster_gmm"] = fits[(report_k, best_gmm_cov)].predict(X)
    F["cluster"] = F.cluster_gmm
    agreement = float(
        pd.crosstab(F.cluster_kmeans, F.cluster_gmm).to_numpy().max(axis=1).sum()
        / len(F))

    prof = (F.groupby("cluster")
             .agg(n_firms=("firm_name", "size"),
                  n_recalls_total=("n_recalls", "sum"),
                  n_recalls_mean=("n_recalls", "mean"),
                  median_units=("median_units", "median"),
                  median_price=("median_price", "median"),
                  share_online_only=("share_online_only", "mean"),
                  share_childrens=("share_childrens", "mean"),
                  share_flammability=("share_flammability", "mean"),
                  mean_span=("span_years", "mean"),
                  mean_last_year=("last_year", "mean"))
             .reset_index())
    prof["label"] = [name_cluster(r) for _, r in prof.iterrows()]
    seen = {}
    labels = []
    for _, r in prof.sort_values("n_firms", ascending=False).iterrows():
        base = r.label
        seen[base] = seen.get(base, 0) + 1
        labels.append((r.cluster, base if seen[base] == 1 else f"{base} ({seen[base]})"))
    lab_map = dict(labels)
    prof["label"] = prof.cluster.map(lab_map)
    prof["example_firms"] = [
        ", ".join(F[F.cluster == c].nlargest(3, "n_recalls").firm_name.tolist())
        for c in prof.cluster]
    prof.to_csv(RES / "cluster_profiles.csv", index=False)
    F["cluster_label"] = F.cluster.map(lab_map)
    F.drop(columns=["log_units", "log_price"]).to_csv(RES / "firm_table.csv",
                                                      index=False)

    typology = {
        "features": feat_cols,
        "imputed_missing": imputed,
        "best_k_kmeans_bic": best_km_k,
        "best_k_gmm_bic": best_gmm_k,
        "reported_k": report_k,
        "k_selection_rule": ("BIC minimum over k = 2..12 for both families; the "
                             "reported typology uses the elbow k, the smallest k "
                             "after which the marginal BIC gain falls below 10% "
                             "of the largest gain"),
        "bic_gain_per_k": {int(kk[i + 1]): float(gains[i]) for i in range(len(gains))},
        "best_gmm_covariance": best_gmm_cov,
        "gmm_bic_min": float(best_gmm.bic),
        "kmeans_gmm_agreement": agreement,
        "profiles": prof.to_dict(orient="records"),
    }

    hop_df = df[df.product_phrase.str.len() > 3].copy()
    if quick:
        hop_df = hop_df[hop_df.year >= 2010]
    hop_df = hop_df.reset_index(drop=True)
    texts = hop_df.product_phrase.str.lower().tolist()

    vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                         min_df=1)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                         min_df=2)
    Xw = vw.fit_transform(texts)
    Xc = vc.fit_transform(texts)
    Sw = (Xw @ Xw.T).toarray()
    Sc = (Xc @ Xc.T).toarray()
    Sim = 0.5 * (Sw + Sc)
    np.fill_diagonal(Sim, 0.0)

    tok = [set(re.findall(r"[a-z0-9']{3,}", t)) for t in texts]
    dfreq: dict[str, int] = {}
    for s in tok:
        for w in s:
            dfreq[w] = dfreq.get(w, 0) + 1
    rare_max = max(RARE_DF_MIN, int(round(RARE_DF_FRAC * len(texts))))
    rare = [{w for w in s if dfreq[w] <= rare_max and w not in PRODUCT_STOP}
            for s in tok]

    dates = hop_df.recall_date.to_numpy()
    fid = hop_df.firm_id.fillna("__unparsed__").to_numpy()
    ctry = hop_df.primary_country.fillna("__unknown__").to_numpy()
    iu = np.triu_indices(len(hop_df), k=1)
    sim_flat = Sim[iu]

    hist_counts, hist_edges = np.histogram(sim_flat, bins=np.arange(0, 1.02, 0.02))
    pd.DataFrame({"bin_left": hist_edges[:-1], "bin_right": hist_edges[1:],
                  "n_pairs": hist_counts}).to_csv(
        RES / "similarity_distribution.csv", index=False)

    cand = []
    for a, b in zip(*iu):
        s = Sim[a, b]
        if s < SIM_THRESHOLD:
            continue
        if fid[a] == fid[b] or fid[a] == "__unparsed__" or fid[b] == "__unparsed__":
            continue
        days = abs((dates[a] - dates[b]) / np.timedelta64(1, "D"))
        if not np.isfinite(days) or days > MAX_DAYS_APART:
            continue
        same_ctry = ctry[a] == ctry[b] and ctry[a] != "__unknown__"
        shared_rare = sorted(rare[a] & rare[b])
        cand.append({
            "similarity": round(float(s), 4),
            "days_apart": int(days),
            "same_country": bool(same_ctry),
            "country": ctry[a] if same_ctry else f"{ctry[a]} / {ctry[b]}",
            "shared_rare_tokens": ";".join(shared_rare),
            "n_shared_rare": len(shared_rare),
            "firm_a": hop_df.firm_name.iloc[a], "firm_b": hop_df.firm_name.iloc[b],
            "year_a": int(hop_df.year.iloc[a]), "year_b": int(hop_df.year.iloc[b]),
            "phrase_a": hop_df.product_phrase.iloc[a],
            "phrase_b": hop_df.product_phrase.iloc[b],
            "channel_a": hop_df.sales_channel.iloc[a],
            "channel_b": hop_df.sales_channel.iloc[b],
            "recall_id_a": int(hop_df.recall_id.iloc[a]),
            "recall_id_b": int(hop_df.recall_id.iloc[b]),
        })
    cand = pd.DataFrame(cand)
    if len(cand):
        cand["flagged"] = cand.same_country & (cand.n_shared_rare >= 1)
        cand["tier"] = np.where(
            cand.flagged, "brand_match",
            np.where(cand.same_country & (cand.similarity >= 0.80)
                     & (cand.channel_a == "online_only")
                     & (cand.channel_b == "online_only"),
                     "category_match_online", "other_high_similarity"))
        cand = cand.sort_values(["flagged", "similarity"], ascending=False)
    else:
        cand["flagged"] = []
        cand["tier"] = []
    cand.to_csv(RES / "entity_hopping_pairs.csv", index=False)

    flagged = cand[cand.flagged] if len(cand) else cand
    hop_firms = sorted(set(flagged.firm_a) | set(flagged.firm_b)) if len(flagged) else []
    hopping = {
        "n_records_compared": int(len(hop_df)),
        "n_pairs_compared": int(len(sim_flat)),
        "similarity_threshold": SIM_THRESHOLD,
        "max_days_apart": MAX_DAYS_APART,
        "rare_token_df_max": rare_max,
        "mean_similarity": float(sim_flat.mean()),
        "p99_similarity": float(np.percentile(sim_flat, 99)),
        "n_pairs_above_threshold": int((sim_flat >= SIM_THRESHOLD).sum()),
        "n_candidate_pairs_after_filters": int(len(cand)),
        "n_flagged_pairs": int(len(flagged)),
        "tier_counts": (cand.tier.value_counts().to_dict() if len(cand) else {}),
        "tier2_examples": (
            cand[cand.tier == "category_match_online"]
            .head(6)[["similarity", "days_apart", "country", "firm_a", "firm_b",
                      "phrase_a", "phrase_b"]].to_dict(orient="records")
            if len(cand) else []),
        "n_firms_involved": len(hop_firms),
        "firms_involved": hop_firms[:40],
        "top_pairs": (flagged.head(15).drop(columns=["flagged"])
                      .to_dict(orient="records") if len(flagged) else []),
        "unflagged_high_similarity_examples": (
            cand[~cand.flagged].head(5)[["similarity", "phrase_a", "phrase_b"]]
            .to_dict(orient="records") if len(cand) else []),
    }

    y = firm.n_recalls.to_numpy(float)
    p_ztp, p_ztnb, comparison, ppc = fit_count_models(y, quick)
    draws = pd.DataFrame(p_ztnb.flat(), columns=p_ztnb.names)
    draws["mu"] = np.exp(draws.log_mu)
    draws["phi"] = np.exp(draws.log_phi)
    draws.sample(min(4000, len(draws)), random_state=0).to_csv(
        RES / "nb_posterior_draws.csv", index=False)

    mu = np.exp(p_ztnb.get("log_mu"))
    phi = np.exp(p_ztnb.get("log_phi"))
    vmr = 1 + mu / phi
    nb_model = {
        "n_firms": int(len(y)), "observed_mean": float(y.mean()),
        "observed_var": float(y.var(ddof=1)),
        "observed_var_mean_ratio": float(y.var(ddof=1) / y.mean()),
        "ztnb_summary": p_ztnb.summary(),
        "ztp_summary": p_ztp.summary(),
        "mu_mean": float(mu.mean()),
        "phi_mean": float(phi.mean()),
        "phi_hdi": [float(np.percentile(phi, 3)), float(np.percentile(phi, 97))],
        "variance_mean_ratio_posterior_mean": float(vmr.mean()),
        "variance_mean_ratio_hdi": [float(np.percentile(vmr, 3)),
                                    float(np.percentile(vmr, 97))],
        "P_overdispersed_vmr_gt_1.2": float((vmr > 1.2).mean()),
        "model_comparison": comparison,
        "posterior_predictive_tails": ppc,
        "converged_ztnb": p_ztnb.converged(rhat_max=1.01, ess_min=400),
        "max_rhat_ztnb": float(np.nanmax(p_ztnb.rhat())),
        "min_ess_ztnb": float(np.nanmin(p_ztnb.ess())),
        "seconds": p_ztp.seconds + p_ztnb.seconds,
    }

    df[["recall_id", "year", "title", "product_phrase", "firm_baseline",
        "firm_v2", "firm_pattern", "key_baseline", "key_v2", "key_merge",
        "firm_id", "firm_name"]].to_csv(RES / "firm_extraction_qa.csv",
                                        index=False)

    summary = {
        "analysis": "13_firm_concentration_repeat",
        "quick_mode": quick,
        "source": str(D.V2_CSV.relative_to(D.ROOT)),
        "extraction_qa": qa,
        "merge_qa": merge_qa,
        "concentration": concentration,
        "gini_all": gini_all,
        "gini_by_era": gini_era,
        "gini_era_difference": gini_diff,
        "typology": typology,
        "entity_hopping": hopping,
        "count_model": nb_model,
    }
    D.write_json(summary, RES / "firm_summary.json")

    print(f"[13] firm parse rate {qa['parse_rate_baseline']:.3f} -> "
          f"{qa['parse_rate_rebuilt']:.3f}; "
          f"{merge_qa['n_keys_before_merge']} keys -> "
          f"{merge_qa['n_firms_after_merge']} firms "
          f"({merge_qa['n_merge_pairs']} merge pairs)")
    print(f"[13] repeat offenders: {concentration['n_firms_ge2']} firms with >=2, "
          f"{concentration['n_firms_ge3']} with >=3, max "
          f"{concentration['max_recalls_one_firm']}")
    print(f"[13] Gini {gini_all['gini']:.3f} "
          f"[{gini_all['lo95']:.3f}, {gini_all['hi95']:.3f}]; by era " +
          "; ".join(f"{k} {v['gini']:.3f}" for k, v in gini_era.items()) +
          f"; diff p={gini_diff.get('p_two_sided', float('nan')):.3f}")
    print(f"[13] typology: BIC min GMM k={best_gmm_k} ({best_gmm_cov}), "
          f"KMeans k={best_km_k}; reported elbow k={report_k}, "
          f"KMeans/GMM agreement {agreement:.2f}")
    print(f"[13] entity hopping: {hopping['n_flagged_pairs']} flagged pairs "
          f"over {hopping['n_firms_involved']} firms")
    print(f"[13] overdispersion: var/mean {nb_model['observed_var_mean_ratio']:.2f}; "
          f"posterior VMR {nb_model['variance_mean_ratio_posterior_mean']:.2f} "
          f"{nb_model['variance_mean_ratio_hdi']}; "
          f"best model {comparison[0]['model']}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fast settings for a smoke run (<60s)")
    main(**vars(ap.parse_args()))
