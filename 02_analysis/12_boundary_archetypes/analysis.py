from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import data as D
from _common import mcmc as M

RES = D.results_dir(__file__)

PERIOD_BINS = [1973, 1989, 1999, 2009, 2019, 2026]
PERIOD_LABELS = ["1974-89", "1990-99", "2000-09", "2010-19", "2020-26"]

_VERB = re.compile(
    r"\b(?:re-?announces?|announces?|issues?|expands?|recalls?|recalling|"
    r"recalled|corrected|retrofits|warning)\b", re.I)
_RECALL_TOKEN = re.compile(r"\b(?:recalls?|recalling|recalled)\b", re.I)
_AFTER_STRIP = re.compile(
    r"^(?:of|to\s+repair|to\s+all|for\s+repair|the|voluntary|expanded|"
    r"consumer|its|two|certain)\b[\s:,-]*", re.I)
_PASSIVE_MARK = re.compile(
    r"^(?:by\b|due\b|because\b|for\b|after\b|over\b|from\b|following\b|"
    r"and\b|linked\b|amid\b|on\b|in\b|at\b|as\b|;|:|$)", re.I)
_CPSC_ANN = re.compile(r"^cpsc[^a-z]{0,3}\s*(?:and\s+)?.*?\bannounces?\b\s*", re.I)
_WARNS = re.compile(r"^cpsc\s+warns?\s+consumers?:?\s*(?:stop\s+using\s*)?"
                    r"(?:certain\s*)?", re.I)
_TAIL = re.compile(
    r"\s*(?:;|:|\bdue\s+to\b|\bbecause\s+of\b|\bfor\s+risk\b|\bposes?\b"
    r"|\bfor\s+\w+\s+hazard\b|\bfor\s+(?:strangulation|choking|fall|burn|fire|"
    r"laceration|entrapment|lead|impalement|suffocation|injury|toxic|leaking)\b"
    r"|\bsold\s+(?:exclusively\s+)?(?:at|by|on|in|through|with)\b"
    r"|\bimported\s+by\b|\bmanufactured\s+by\b|\bdistributed\s+by\b"
    r"|\brecalled\b|\brecall\b|\bviolat\w*|\(recall\s+alert\)|\bafter\b|\bover\b"
    r"|\bmay\s+not\b|\bwarning\b)", re.I)
_QUANT = re.compile(r"^(?:two|three|four|five|six|eight|nine|ten|\d+)\s+"
                    r"(?:styles?|models?|types?|kinds?)\s+of\s+", re.I)
_CITATION_CUT = re.compile(
    r"\b(?:violat\w*|mandatory|federal|standards?|regulations?|rule)\b", re.I)


def product_phrase(title: str) -> str:
    t = re.sub(r"\s+", " ", str(title or "")).strip()
    t = _WARNS.sub("", t)
    m = _RECALL_TOKEN.search(t)
    if m:
        after = _AFTER_STRIP.sub("", t[m.end():].strip())
        if _PASSIVE_MARK.match(after) or len(after.split()) < 1:
            seg = _CPSC_ANN.sub("", t[:m.start()])
        else:
            seg = after
    else:
        seg = _CPSC_ANN.sub("", t)
    seg = _TAIL.split(seg)[0]
    seg = _QUANT.sub("", seg)
    return re.sub(r"\s+", " ", seg).strip(" .,-–—:’'\"")


def reduced_title(title: str) -> str:
    t = re.sub(r"\s+", " ", str(title or "")).strip()
    return _CITATION_CUT.split(t)[0]


LEXICON: list[tuple[str, str, bool | None]] = [
    ("nightgown", r"night\s?gowns?|night\s?dress(?:es)?|night\s?shirts?|"
                  r"sleep\s?gowns?|nighti(?:e|es)|nightys?", False),
    ("robe", r"bath\s?robes?|\brobes?\b|kimonos?|dressing gowns?|housecoats?", False),
    ("loungewear", r"lounge\s?wear|lounge\s?(?:sets?|pants?|shorts?|tops?|"
                   r"bottoms?|suits?)|loungers?\b", False),
    ("wearable_blanket", r"wearable blankets?|blanket sleepers?|sleep\s?sacks?|"
                         r"sleeping bags?|swaddles?|sleep bags?|"
                         r"sleeping sacks?", False),
    ("pajama_set", r"pajamas?|pyjamas?|\bp\.?j\.?s?\b|jammies|jammers|"
                   r"footed sleepers?|one[- ]piece sleepers?|sleep\s?sets?|"
                   r"pajama sets?", True),
    ("sleepwear_generic", r"sleep\s?wear|night\s?wear|\bsleepers?\b", True),
    ("underwear_base", r"underwear|undershirts?|thermal(?:s\b|\s+(?:sets?|"
                       r"underwear|tops?|bottoms?))|long johns|base layers?|"
                       r"\bboxers?\b|\bbriefs?\b|camisoles?|\bslips?\b|bras?\b", True),
    ("swimwear", r"swim\w*|bikinis?|bathing suits?|rash ?guards?|wet ?suits?", None),
    ("costume", r"costumes?|cosplay|dress[- ]up (?:sets?|outfits?)", None),
    ("footwear", r"shoes?|boots?|booties|sandals?|sneakers?|slippers?|footwear|"
                 r"clogs?|flip[- ]?flops?|moccasins?|loafers?|cleats?|"
                 r"slipper socks?", None),
    ("outerwear", r"jackets?|coats?|parkas?|anoraks?|hoodies?|hooded sweat\w*|"
                  r"sweat\s?shirts?|sweat\s?suits?|sweaters?|cardigans?|"
                  r"\bvests?\b|wind\s?suits?|wind\s?breakers?|snow\s?suits?|"
                  r"ski suits?|jogging suits?|track suits?|rain\s?coats?|"
                  r"ponchos?|outerwear|snow ?pants?|bunting bags?|"
                  r"pullovers?|fleece (?:jackets?|tops?|pullovers?)", None),
    ("daywear", r"t[- ]?shirts?|\bshirts?\b|blouses?|\btops?\b|\bpants?\b|"
                r"trousers?|jeans|shorts\b|skirts?|dress(?:es)?\b|rompers?|"
                r"onesies?|bodysuits?|jumpsuits?|overalls?|coveralls?|"
                r"leggings?|jumpers?|short sets?|pant sets?|jerseys?|"
                r"(?:clothing|apparel|play|outfit) sets?|playwear|uniforms?|"
                r"tunics?|tutus?|\bsuits?\b|shortalls?|koveralls?|dungarees?", None),
    ("accessory", r"scar(?:f|ves)|gloves?|mittens?|\bhats?\b|beanies?|caps?\b|"
                  r"socks?|tights|hosiery|\bbelts?\b|jewel(?:ry|lery)|"
                  r"necklaces?|bracelets?|earrings?|charms?|key\s?chains?|"
                  r"head\s?bands?|hair (?:clips?|bands?|accessories)|\bbibs?\b|"
                  r"backpacks?|purses?|hand\s?bags?|sunglasses|wrist\s?bands?|"
                  r"suspenders?|neck\s?ties?|bandanas?|ear\s?muffs?|"
                  r"leg warmers?|arm\s?bands?", None),
    ("home_textile", r"blankets?|comforters?|quilts?|bedding|bed sheets?|"
                     r"sheet sets?|pillows?|curtains?|drapes?|towels?|"
                     r"mattress pads?|throws?|rugs?|shams?|duvets?|"
                     r"crib bumpers?|nap mats?", None),
    ("toy_juvenile", r"toys?|dolls?|pacifiers?|teddy bears?|plush\w*|"
                     r"stuffed \w+|rattles?|bears?\b|bunnies|bunny|animals?\b|"
                     r"figures?|puzzles?|play\s?mats?|strollers?|cribs?|"
                     r"high chairs?|bunk beds?|carriers?|walkers?|swings?", None),
    ("non_textile_other",
     r"bindings?|boilers?|water heaters?|heaters?|washing machines?|"
     r"network cables?|cables?|roller skates?|inline skates?|"
     r"crampons?|harness(?:es)?|tea\s?lights?|candle\s?holders?|"
     r"candlesticks?|\blamps?\b|night\s?lights?|popcorn makers?|"
     r"drinking glasses|tumblers?|water bottles?|coolers?|sippy cups?|"
     r"athletic cups?|\byarns?\b|\bbooks?\b|\bslides?\b|swing sets?|"
     r"exercisers?|copters?|wagons?|jump\s?start|power supply|\bpumps?\b|"
     r"grills?|humidifiers?|\bkits?\b|decorations?|ornaments?|wall hooks?|"
     r"gel fuel|snow\s?shoes?|tents?|canopies|dryers?|deodorizers?|sanders?|"
     r"batteries|chargers?|scooters?|helmets?|climbers?|\bropes?\b|"
     r"foot warmers?|party favors?|glue|\bbottles?\b|\bmugs?\b|steamers?|"
     r"step stools?|\bstools?\b|\bpotty\b", None),
    ("generic_apparel", r"garments?|apparel|clothing|clothes|outfits?|"
                        r"\bwear\b|\bsets?\b|\bwardrobes?\b", None),
]
_COMPILED = [(n, re.compile(p, re.I), t) for n, p, t in LEXICON]

NAMED_SLEEPWEAR = re.compile(
    r"pajamas?|pyjamas?|\bp\.?j\.?s?\b|jammies|jammers|sleep\s?wear|"
    r"night\s?wear|night\s?gowns?|night\s?dress(?:es)?|night\s?shirts?|"
    r"sleep\s?gowns?|nighti(?:e|es)|sleep\s?sets?|footed sleepers?", re.I)

SLEEP_STD_STRICT = re.compile(
    r"1615|1616|standards?\s+for\s+children.?s?\s+sleepwear|"
    r"children.?s?\s+sleepwear\s+(?:standard|flammability|regulation)|"
    r"flammability\s+(?:standard|regulation)s?\s+for\s+(?:children.?s?\s+)?sleepwear|"
    r"sleepwear\s+(?:standard|flammability)", re.I)
GENERAL_APPAREL_STD = re.compile(
    r"1610|standards?\s+for\s+(?:clothing|wearing apparel)|"
    r"regulations?\s+for\s+clothing|general wearing apparel", re.I)
VIOLATION_RE = re.compile(r"violat\w*", re.I)
FLAM_RE = re.compile(r"flammab\w*|burn hazard|flame", re.I)

INELIGIBLE = "exemption_ineligible"
ELIGIBLE = "exemption_eligible"
NOT_SLEEP = "not_sleepwear"

GARMENT_ARCHETYPES = [
    "nightgown", "robe", "loungewear", "wearable_blanket", "pajama_set",
    "sleepwear_generic", "underwear_base", "outerwear", "daywear", "swimwear",
    "costume", "footwear", "generic_apparel",
]
BIN_EDGES = [2004, 2009, 2014, 2019, 2024, 2026]
BIN_LABELS = ["2005-09", "2010-14", "2015-19", "2020-24", "2025-26"]


def classify_phrase(phrase: str):
    hits = []
    for idx, (name, rx, tight) in enumerate(_COMPILED):
        m = rx.search(phrase)
        if m:
            hits.append((m.start(), -(m.end() - m.start()), idx, name, tight))
    if not hits:
        return "unclassified", None, []
    hits.sort()
    labels = list(dict.fromkeys(h[3] for h in sorted(hits, key=lambda h: h[0])))
    return hits[0][3], hits[0][4], labels


def cochran_armitage(y: np.ndarray, score: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    x = np.asarray(score, dtype=float)
    n = len(y)
    p = y.mean()
    T = float(np.sum(x * (y - p)))
    var = p * (1 - p) * float(np.sum(x ** 2) - np.sum(x) ** 2 / n)
    z = T / np.sqrt(var) if var > 0 else np.nan
    return {"z": float(z), "p_two_sided": float(2 * stats.norm.sf(abs(z))),
            "n": int(n), "n_events": int(y.sum()), "mean_outcome": float(p)}


def irls_logistic(X: np.ndarray, y: np.ndarray, max_iter=100, tol=1e-10):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        w = np.maximum(mu * (1 - mu), 1e-9)
        z = eta + (y - mu) / w
        XtW = X.T * w
        try:
            new = np.linalg.solve(XtW @ X, XtW @ z)
        except np.linalg.LinAlgError:
            new = np.linalg.lstsq(XtW @ X, XtW @ z, rcond=None)[0]
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
    w = np.maximum(mu * (1 - mu), 1e-9)
    cov = np.linalg.pinv((X.T * w) @ X)
    ll = float(np.sum(y * np.log(np.clip(mu, 1e-12, 1)) +
                      (1 - y) * np.log(np.clip(1 - mu, 1e-12, 1))))
    return beta, cov, ll


def logistic_trend(year: np.ndarray, y: np.ndarray, centre: float,
                   grid: np.ndarray) -> dict:
    x = (np.asarray(year, dtype=float) - centre) / 10.0
    X = np.column_stack([np.ones_like(x), x])
    beta, cov, ll1 = irls_logistic(X, y)
    _, _, ll0 = irls_logistic(np.ones((len(y), 1)), y)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    lrt = 2 * (ll1 - ll0)
    xg = (np.asarray(grid, dtype=float) - centre) / 10.0
    Xg = np.column_stack([np.ones_like(xg), xg])
    eta = Xg @ beta
    se_eta = np.sqrt(np.einsum("ij,jk,ik->i", Xg, cov, Xg))
    inv = lambda a: 1.0 / (1.0 + np.exp(-a))
    return {
        "beta": beta.tolist(), "se": se.tolist(), "z": z.tolist(),
        "p": (2 * stats.norm.sf(np.abs(z))).tolist(),
        "or_per_decade": float(np.exp(beta[1])),
        "or_ci95": [float(np.exp(beta[1] - 1.96 * se[1])),
                    float(np.exp(beta[1] + 1.96 * se[1]))],
        "lrt_chi2": float(lrt), "lrt_p": float(stats.chi2.sf(lrt, 1)),
        "loglik": ll1, "loglik_null": ll0,
        "fit": pd.DataFrame({"year": grid, "fit": inv(eta),
                             "lo": inv(eta - 1.96 * se_eta),
                             "hi": inv(eta + 1.96 * se_eta)}),
    }


def crosstab_test(a: pd.Series, b: pd.Series, n_perm: int, seed=11) -> dict:
    tab = pd.crosstab(a, b)
    chi2, p, dof, exp = stats.chi2_contingency(tab)
    n = int(tab.to_numpy().sum())
    k = min(tab.shape) - 1
    v = float(np.sqrt(chi2 / (n * k))) if n and k else np.nan
    rng = np.random.default_rng(seed)
    av, bv = a.to_numpy(), b.to_numpy()
    ge = 0
    for _ in range(n_perm):
        t = pd.crosstab(av, rng.permutation(bv))
        ge += stats.chi2_contingency(t)[0] >= chi2
    return {"chi2": float(chi2), "dof": int(dof), "p_asymptotic": float(p),
            "p_permutation": float((ge + 1) / (n_perm + 1)),
            "cramers_v": v, "n": n,
            "min_expected": float(exp.min()), "n_perm": int(n_perm),
            "table": tab}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (max(c - h, 0.0), min(c + h, 1.0))


def iqr_row(s: pd.Series) -> dict:
    v = pd.to_numeric(s, errors="coerce").dropna().to_numpy()
    if len(v) == 0:
        return {"n": 0, "median": np.nan, "q1": np.nan, "q3": np.nan}
    return {"n": int(len(v)), "median": float(np.median(v)),
            "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75))}


def bayes_boundary(df: pd.DataFrame, quick: bool) -> tuple[M.Posterior, dict]:
    d = df.copy()
    y = (d.boundary_class_v3 == INELIGIBLE).to_numpy(float)
    year_c = (d.year.to_numpy(float) - 2010.0) / 10.0
    online = (d.sales_channel == "online_only").to_numpy(float)
    kids = d.is_childrens.fillna(False).to_numpy(bool).astype(float)
    X = np.column_stack([np.ones_like(y), year_c, online, kids])
    names = ["alpha", "b_year_decade", "b_online_only", "b_childrens"]

    def log_post(theta):
        eta = X @ theta
        ll = np.sum(y * eta - np.logaddexp(0.0, eta))
        lp = (M.log_normal(theta[0], 0.0, 2.5)
              + np.sum(M.log_normal(theta[1:], 0.0, 1.5)))
        return float(ll + lp)

    n_draws, n_tune, n_chains = (600, 600, 2) if quick else (6000, 6000, 4)
    beta0, _, _ = irls_logistic(X, y)
    post = M.sample(log_post, beta0, n_draws=n_draws, n_tune=n_tune,
                    n_chains=n_chains, seed=12, names=names, verbose=True)
    rows = post.summary()
    for r in rows:
        r["odds_ratio"] = float(np.exp(r["mean"]))
        r["or_hdi_3%"] = float(np.exp(r["hdi_3%"]))
        r["or_hdi_97%"] = float(np.exp(r["hdi_97%"]))
        r["p_positive"] = float((post.get(r["param"]) > 0).mean())
    info = {
        "n": int(len(y)), "n_ineligible": int(y.sum()),
        "n_draws": n_draws, "n_tune": n_tune, "n_chains": n_chains,
        "accept_rate": post.accept_rate.tolist(),
        "seconds": post.seconds,
        "converged": post.converged(rhat_max=1.01, ess_min=400),
        "max_rhat": float(np.nanmax(post.rhat())),
        "min_ess": float(np.nanmin(post.ess())),
        "params": rows,
        "P_online_effect_positive": float((post.get("b_online_only") > 0).mean()),
        "P_year_effect_positive": float((post.get("b_year_decade") > 0).mean()),
        "mle_check": beta0.tolist(),
    }
    return post, info


def main(quick: bool = False):
    df = pd.read_csv(D.V2_CSV)
    df["period"] = pd.cut(df.year, bins=PERIOD_BINS, labels=PERIOD_LABELS)
    df["bin5"] = pd.cut(df.year, bins=BIN_EDGES, labels=BIN_LABELS)
    n_perm = 200 if quick else 5000

    df["product_phrase"] = df.title.map(product_phrase)
    parsed = df.product_phrase.map(classify_phrase)
    df["archetype_v3"] = [a for a, _, _ in parsed]
    df["can_be_tight_v3"] = [t for _, t, _ in parsed]
    df["archetype_all"] = [";".join(labs) for _, _, labs in parsed]
    df["n_archetypes"] = [len(labs) for _, _, labs in parsed]

    still = df.archetype_v3 == "unclassified"
    fb = df.loc[still, "title"].map(reduced_title).map(classify_phrase)
    df.loc[still, "archetype_v3"] = [a for a, _, _ in fb]
    df.loc[still, "can_be_tight_v3"] = [t for _, t, _ in fb]
    df.loc[still, "archetype_all"] = [";".join(l) for _, _, l in fb]
    df["archetype_from_title_fallback"] = False
    df.loc[still, "archetype_from_title_fallback"] = (
        df.loc[still, "archetype_v3"] != "unclassified")

    df["boundary_class_v3"] = np.select(
        [df.can_be_tight_v3.eq(True), df.can_be_tight_v3.eq(False)],
        [ELIGIBLE, INELIGIBLE], default=NOT_SLEEP)

    base_unc = int((df.archetype == "unclassified").sum())
    new_unc = int((df.archetype_v3 == "unclassified").sum())
    parse_quality = {
        "n_records": int(len(df)),
        "unclassified_v2_baseline": base_unc,
        "unclassified_v2_baseline_pct": round(100 * base_unc / len(df), 1),
        "unclassified_refined": new_unc,
        "unclassified_refined_pct": round(100 * new_unc / len(df), 1),
        "reduction_absolute": base_unc - new_unc,
        "reduction_pct_of_baseline": round(100 * (base_unc - new_unc) / base_unc, 1),
        "n_recovered_from_baseline_unclassified":
            int(((df.archetype == "unclassified") &
                 (df.archetype_v3 != "unclassified")).sum()),
        "n_reclassified_between_named_archetypes":
            int(((df.archetype != "unclassified") &
                 (df.archetype_v3 != "unclassified") &
                 (df.archetype != df.archetype_v3)).sum()),
        "n_multi_archetype_titles": int((df.n_archetypes > 1).sum()),
        "n_recovered_via_title_fallback":
            int(df.archetype_from_title_fallback.sum()),
        "product_phrase_empty": int((df.product_phrase.str.len() == 0).sum()),
        "recovered_into": (df.loc[df.archetype == "unclassified", "archetype_v3"]
                           .value_counts().to_dict()),
        "boundary_class_v2_counts": df.boundary_class.value_counts().to_dict(),
        "boundary_class_v3_counts": df.boundary_class_v3.value_counts().to_dict(),
    }

    t = df.title.fillna("")
    strict = t.str.contains(SLEEP_STD_STRICT)
    general = t.str.contains(GENERAL_APPAREL_STD)
    kids = df.is_childrens.fillna(False).astype(bool)
    inferred = kids & t.str.contains(VIOLATION_RE) & t.str.contains(FLAM_RE) & ~general
    df["sleep_std_strict"] = strict
    df["sleep_std_broad"] = strict | inferred
    df["named_sleepwear"] = df.product_phrase.str.contains(NAMED_SLEEPWEAR)
    df["is_garment"] = df.archetype_v3.isin(GARMENT_ARCHETYPES)
    df["arbitrage_broad"] = df.sleep_std_broad & ~df.named_sleepwear & df.is_garment
    df["arbitrage_strict"] = df.sleep_std_strict & ~df.named_sleepwear & df.is_garment
    df["arbitrage_conservative"] = df.arbitrage_broad & df.archetype_v3.isin(
        ["loungewear", "robe", "wearable_blanket", "underwear_base",
         "generic_apparel"])

    flag_qa = {
        "sleepwear_standard_v2_flag": int(df.sleepwear_standard.sum()),
        "sleep_std_strict": int(strict.sum()),
        "sleep_std_broad": int(df.sleep_std_broad.sum()),
        "v2_flag_not_in_broad": int((df.sleepwear_standard & ~df.sleep_std_broad).sum()),
        "v2_flag_not_in_broad_examples":
            df.loc[df.sleepwear_standard & ~df.sleep_std_broad, "title"].head(6).tolist(),
        "in_broad_not_v2_flag": int((~df.sleepwear_standard & df.sleep_std_broad).sum()),
        "in_broad_not_v2_flag_examples":
            df.loc[~df.sleepwear_standard & df.sleep_std_broad, "title"].head(6).tolist(),
        "category_arbitrage_v2_flag": int(df.category_arbitrage.sum()),
        "arbitrage_broad": int(df.arbitrage_broad.sum()),
        "arbitrage_strict": int(df.arbitrage_strict.sum()),
        "arbitrage_conservative": int(df.arbitrage_conservative.sum()),
    }

    ct = pd.crosstab(df.period, df.archetype_v3)
    shares = ct.div(ct.sum(axis=1), axis=0)
    arche_period = (ct.stack().rename("n").reset_index()
                    .merge(shares.stack().rename("share").reset_index(),
                           on=["period", "archetype_v3"]))
    arche_period["boundary_class"] = arche_period.archetype_v3.map(
        df.drop_duplicates("archetype_v3").set_index("archetype_v3")
        .boundary_class_v3)
    arche_period.to_csv(RES / "archetype_by_period.csv", index=False)

    sw = df[df.sleep_std_broad].copy()
    bt = pd.crosstab(sw.period, sw.boundary_class_v3)
    for c in [ELIGIBLE, INELIGIBLE, NOT_SLEEP]:
        if c not in bt:
            bt[c] = 0
    bt = bt[[ELIGIBLE, INELIGIBLE, NOT_SLEEP]]
    bt_share = bt.div(bt.sum(axis=1).replace(0, np.nan), axis=0)
    boundary_period = bt.add_prefix("n_").join(bt_share.add_prefix("share_"))
    boundary_period["n_total"] = bt.sum(axis=1)
    boundary_period = boundary_period.reset_index()
    boundary_period.to_csv(RES / "boundary_by_period.csv", index=False)

    sws = df[df.sleep_std_strict]
    bts = pd.crosstab(sws.period, sws.boundary_class_v3)
    bts_share = bts.div(bts.sum(axis=1).replace(0, np.nan), axis=0)

    arche_sw = pd.crosstab(sw.period, sw.archetype_v3)
    arche_sw.reset_index().to_csv(RES / "sleepwear_archetype_by_period.csv",
                                  index=False)
    arche_bin = pd.crosstab(sw.bin5, sw.archetype_v3)
    arche_bin.reset_index().to_csv(RES / "sleepwear_archetype_by_bin.csv",
                                   index=False)
    bin_boundary = pd.crosstab(sw.bin5, sw.boundary_class_v3)
    bin_boundary.reset_index().to_csv(RES / "sleepwear_boundary_by_bin.csv",
                                      index=False)

    by_year = (sw.groupby("year")
                 .agg(n_sleepwear_std=("arbitrage_broad", "size"),
                      n_arbitrage=("arbitrage_broad", "sum"))
                 .reset_index())
    by_year["share"] = by_year.n_arbitrage / by_year.n_sleepwear_std
    ci = [wilson(int(k), int(n)) for k, n in
          zip(by_year.n_arbitrage, by_year.n_sleepwear_std)]
    by_year["lo95"] = [c[0] for c in ci]
    by_year["hi95"] = [c[1] for c in ci]
    by_year.to_csv(RES / "arbitrage_by_year.csv", index=False)

    def agg_shares(frame, key):
        out = (frame.groupby(key, observed=True)
                    .agg(n_sleepwear_std=("arbitrage_broad", "size"),
                         n_arbitrage=("arbitrage_broad", "sum"))
                    .reset_index())
        out["share"] = out.n_arbitrage / out.n_sleepwear_std
        c = [wilson(int(k), int(n)) for k, n in
             zip(out.n_arbitrage, out.n_sleepwear_std)]
        out["lo95"] = [x[0] for x in c]
        out["hi95"] = [x[1] for x in c]
        return out

    by_period = agg_shares(sw, "period")
    by_period.to_csv(RES / "arbitrage_by_period.csv", index=False)
    by_bin = agg_shares(sw, "bin5")
    by_bin.to_csv(RES / "arbitrage_by_bin.csv", index=False)

    grid = np.arange(int(sw.year.min()), int(sw.year.max()) + 1)
    trend = {}
    for label, col, frame in [("broad", "arbitrage_broad", sw),
                              ("strict", "arbitrage_strict", df[df.sleep_std_strict]),
                              ("conservative", "arbitrage_conservative", sw)]:
        yv = frame[col].to_numpy(float)
        if yv.sum() in (0, len(yv)) or len(yv) < 10:
            trend[label] = {"skipped": True, "n": int(len(yv)),
                            "n_events": int(yv.sum())}
            continue
        ca = cochran_armitage(yv, frame.year.to_numpy(float))
        lt = logistic_trend(frame.year.to_numpy(float), yv, 2010.0, grid)
        fit = lt.pop("fit")
        if label == "broad":
            fit.to_csv(RES / "arbitrage_trend_fit.csv", index=False)
        trend[label] = {"cochran_armitage": ca, "logistic": lt}

    yrs = np.arange(int(sw.year.min()), int(sw.year.max()) + 1)
    cnt = (by_year.set_index("year").reindex(yrs).fillna(0))
    pois = {}
    for label, col in [("arbitrage_count", "n_arbitrage"),
                       ("sleepwear_standard_count", "n_sleepwear_std")]:
        yv = cnt[col].to_numpy(float)
        xv = (yrs - 2010.0) / 10.0
        Xp = np.column_stack([np.ones_like(xv), xv])
        b = np.zeros(2)
        for _ in range(200):
            mu = np.exp(np.clip(Xp @ b, -30, 30))
            W = np.maximum(mu, 1e-9)
            z = Xp @ b + (yv - mu) / W
            b_new = np.linalg.solve((Xp.T * W) @ Xp, (Xp.T * W) @ z)
            if np.max(np.abs(b_new - b)) < 1e-10:
                b = b_new
                break
            b = b_new
        mu = np.exp(np.clip(Xp @ b, -30, 30))
        cov = np.linalg.pinv((Xp.T * np.maximum(mu, 1e-9)) @ Xp)
        se = np.sqrt(np.diag(cov))
        disp = float(np.sum((yv - mu) ** 2 / np.maximum(mu, 1e-9)) / (len(yv) - 2))
        se_q = se * np.sqrt(max(disp, 1.0))
        pois[label] = {
            "beta": b.tolist(), "se_quasi": se_q.tolist(),
            "rate_ratio_per_decade": float(np.exp(b[1])),
            "rr_ci95": [float(np.exp(b[1] - 1.96 * se_q[1])),
                        float(np.exp(b[1] + 1.96 * se_q[1]))],
            "z_quasi": float(b[1] / se_q[1]),
            "p_quasi": float(2 * stats.norm.sf(abs(b[1] / se_q[1]))),
            "dispersion": disp,
            "note": "2026 is a partial year (data through 2026-07-16)",
        }

    def share(mask):
        s = sw[mask]
        k, n = int(s.arbitrage_broad.sum()), int(len(s))
        lo, hi = wilson(k, n)
        return {"n": n, "k": k, "share": (k / n if n else np.nan),
                "lo95": lo, "hi95": hi}

    era_split = {"pre_2020": share(sw.year < 2020), "from_2020": share(sw.year >= 2020)}
    a = sw[sw.year < 2020].arbitrage_broad
    b = sw[sw.year >= 2020].arbitrage_broad
    tab2 = np.array([[int(b.sum()), int(len(b) - b.sum())],
                     [int(a.sum()), int(len(a) - a.sum())]])
    era_split["fisher_p"] = float(stats.fisher_exact(tab2)[1])
    era_split["odds_ratio"] = float(stats.fisher_exact(tab2)[0])

    GARMENT = [a for a in df.archetype_v3.unique()
               if a not in ("toy_juvenile", "unclassified")]
    g = df[df.archetype_v3.isin(GARMENT)].copy()
    ch = g[g.sales_channel.ne("unknown")]
    x_channel = crosstab_test(ch.archetype_v3, ch.sales_channel, n_perm)
    tab_ch = x_channel.pop("table")
    tab_ch.to_csv(RES / "crosstab_archetype_channel.csv")

    TOP_C = ["China", "Vietnam", "India", "Pakistan", "Bangladesh", "Indonesia",
             "United States"]
    co = g[g.primary_country.notna()].copy()
    co["country_group"] = np.where(co.primary_country.isin(TOP_C),
                                   co.primary_country, "Other")
    x_country = crosstab_test(co.archetype_v3, co.country_group, n_perm)
    tab_co = x_country.pop("table")
    tab_co.to_csv(RES / "crosstab_archetype_country.csv")

    bc = df[df.sales_channel.ne("unknown")]
    x_boundary_channel = crosstab_test(bc.boundary_class_v3, bc.sales_channel, n_perm)
    tab_bc = x_boundary_channel.pop("table")
    tab_bc.to_csv(RES / "crosstab_boundary_channel.csv")

    online_by_arche = (g.assign(online=g.sales_channel.eq("online_only"))
                        .groupby("archetype_v3")
                        .agg(n=("online", "size"), n_online=("online", "sum")))
    online_by_arche["share_online_only"] = (online_by_arche.n_online /
                                            online_by_arche.n)
    online_by_arche.reset_index().to_csv(RES / "online_share_by_archetype.csv",
                                         index=False)

    rows = []
    for arche, sub in df.groupby("archetype_v3"):
        u, p = iqr_row(sub.units), iqr_row(sub.price_usd)
        rows.append({
            "archetype": arche,
            "boundary_class": sub.boundary_class_v3.iloc[0],
            "n_recalls": int(len(sub)),
            "units_n": u["n"], "units_median": u["median"],
            "units_q1": u["q1"], "units_q3": u["q3"],
            "price_n": p["n"], "price_median": p["median"],
            "price_q1": p["q1"], "price_q3": p["q3"],
            "share_online_only": float(sub.sales_channel.eq("online_only").mean()),
            "share_childrens": float(sub.is_childrens.fillna(False).mean()),
            "share_2020plus": float((sub.year >= 2020).mean()),
        })
    pu = pd.DataFrame(rows).sort_values("n_recalls", ascending=False)
    pu.to_csv(RES / "price_units_by_archetype.csv", index=False)

    def kw(col):
        groups = [pd.to_numeric(df.loc[df.boundary_class_v3 == c, col],
                                errors="coerce").dropna().to_numpy()
                  for c in [ELIGIBLE, INELIGIBLE, NOT_SLEEP]]
        groups = [x for x in groups if len(x) > 1]
        if len(groups) < 2:
            return {"skipped": True}
        h, p = stats.kruskal(*groups)
        return {"H": float(h), "p": float(p),
                "medians": {c: float(np.nanmedian(pd.to_numeric(
                    df.loc[df.boundary_class_v3 == c, col], errors="coerce")))
                    for c in [ELIGIBLE, INELIGIBLE, NOT_SLEEP]}}

    kruskal = {"units": kw("units"), "price_usd": kw("price_usd")}

    post, bayes = bayes_boundary(df, quick)
    draws = pd.DataFrame(post.flat(), columns=post.names)
    draws.sample(min(4000, len(draws)), random_state=0).to_csv(
        RES / "boundary_posterior_draws.csv", index=False)

    keep = ["recall_id", "year", "period", "title", "product_phrase",
            "archetype", "archetype_v3", "archetype_all", "can_be_tight_v3",
            "boundary_class", "boundary_class_v3", "named_sleepwear",
            "sleepwear_standard", "sleep_std_strict", "sleep_std_broad",
            "category_arbitrage", "arbitrage_broad", "arbitrage_strict",
            "arbitrage_conservative", "sales_channel", "primary_country",
            "is_childrens", "units", "price_usd", "hazard_category", "firm"]
    df[keep].to_csv(RES / "archetype_records.csv", index=False)
    df.loc[df.archetype_v3 == "unclassified",
           ["recall_id", "year", "title", "product_phrase"]].to_csv(
        RES / "unclassified_residual.csv", index=False)
    df.loc[df.archetype_v3 == "non_textile_other",
           ["recall_id", "year", "title", "product_phrase", "hazard_category"]] \
      .to_csv(RES / "nontextile_candidates.csv", index=False)
    df.loc[df.arbitrage_broad,
           ["recall_id", "year", "title", "product_phrase", "archetype_v3",
            "sales_channel", "primary_country", "units", "price_usd"]].sort_values(
        "year").to_csv(RES / "arbitrage_cases.csv", index=False)

    summary = {
        "analysis": "12_boundary_archetypes",
        "quick_mode": quick,
        "source": str(D.V2_CSV.relative_to(D.ROOT)),
        "n_records": int(len(df)),
        "year_range": [int(df.year.min()), int(df.year.max())],
        "period_bins": PERIOD_LABELS,
        "parse_quality": parse_quality,
        "archetype_counts_refined": df.archetype_v3.value_counts().to_dict(),
        "archetype_counts_baseline": df.archetype.value_counts().to_dict(),
        "sleepwear_standard_flags": flag_qa,
        "boundary_by_period": boundary_period.to_dict(orient="records"),
        "boundary_by_period_strict": (
            bts.add_prefix("n_").join(bts_share.add_prefix("share_"))
            .reset_index().to_dict(orient="records")),
        "sleepwear_archetype_by_period":
            arche_sw.reset_index().to_dict(orient="records"),
        "arbitrage_by_year": by_year.to_dict(orient="records"),
        "arbitrage_by_period": by_period.to_dict(orient="records"),
        "arbitrage_by_bin": by_bin.to_dict(orient="records"),
        "arbitrage_trend": trend,
        "poisson_count_trend": pois,
        "arbitrage_era_split": era_split,
        "arbitrage_case_titles_2020plus":
            df.loc[df.arbitrage_broad & (df.year >= 2020), "title"].tolist(),
        "crosstab_archetype_channel": x_channel,
        "crosstab_archetype_country": x_country,
        "crosstab_boundary_channel": x_boundary_channel,
        "boundary_channel_table": tab_bc.to_dict(),
        "price_units_by_archetype": pu.to_dict(orient="records"),
        "kruskal_by_boundary_class": kruskal,
        "bayes_boundary_model": bayes,
    }
    D.write_json(summary, RES / "boundary_summary.json")

    print(f"[12] unclassified {base_unc} -> {new_unc} "
          f"({parse_quality['reduction_pct_of_baseline']}% of baseline resolved)")
    print(f"[12] sleepwear-standard recalls: v2 flag {flag_qa['sleepwear_standard_v2_flag']}"
          f" -> broad {flag_qa['sleep_std_broad']} (strict {flag_qa['sleep_std_strict']})")
    print(f"[12] category arbitrage: {flag_qa['arbitrage_broad']} of "
          f"{flag_qa['sleep_std_broad']} sleepwear-standard recalls")
    if "cochran_armitage" in trend.get("broad", {}):
        ca = trend["broad"]["cochran_armitage"]
        lg = trend["broad"]["logistic"]
        print(f"[12] arbitrage trend: CA z={ca['z']:.2f} p={ca['p_two_sided']:.4f}; "
              f"OR/decade={lg['or_per_decade']:.2f} "
              f"[{lg['or_ci95'][0]:.2f}, {lg['or_ci95'][1]:.2f}]")
    pc = pois["arbitrage_count"]
    print(f"[12] arbitrage COUNT trend: rate ratio/decade {pc['rate_ratio_per_decade']:.2f} "
          f"[{pc['rr_ci95'][0]:.2f}, {pc['rr_ci95'][1]:.2f}] p={pc['p_quasi']:.4f}")
    print(f"[12] arbitrage share pre-2020 {era_split['pre_2020']['share']:.3f} -> "
          f"2020+ {era_split['from_2020']['share']:.3f} "
          f"(Fisher p={era_split['fisher_p']:.4f})")
    print(f"[12] P(online-only effect > 0) = {bayes['P_online_effect_positive']:.3f}; "
          f"max Rhat {bayes['max_rhat']:.3f}, min ESS {bayes['min_ess']:.0f}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fast settings for a smoke run (<60s)")
    main(**vars(ap.parse_args()))
