from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "02_analysis"
DATA = ROOT / "01_data"

RAW_JSON = DATA / "cpsc_recalls_all.json"
V1_CSV = ANALYSIS / "01_dataset_construction" / "apparel_recalls_clean.csv"
V2_CSV = ANALYSIS / "09_dataset_hardening" / "results" / "apparel_recalls_v2.csv"
OTEXA_SME = DATA / "otexa_m2_total_apparel_imports_1989_2025_sme.csv"
OTEXA_USD = DATA / "otexa_m2_total_apparel_imports_1989_2025_dollars.csv"

YEAR_MIN, YEAR_MAX = 1974, 2025

PERIOD_BINS = [1973, 1989, 1999, 2009, 2019, 2025]
PERIOD_LABELS = ["1974-89", "1990-99", "2000-09", "2010-19", "2020-25"]

FALSE_POSITIVE_PATTERNS = [
    r"side-by-side (?:off-road )?vehicle",
    r"aerosol (?:foam|waterproofing)",
    r"motion lamps?", r"torchiere floor lamps?",
    r"press and go toy vehicles?",
    r"\bgas grills?\b", r"\bpower washers?\b",
]
FALSE_POSITIVE_RE = re.compile("|".join(FALSE_POSITIVE_PATTERNS), re.I)

ELECTRIC_TEXTILE_RE = re.compile(
    r"heated (?:blanket|throw|mattress pad|vest|jacket|socks?)|electric blanket", re.I)

COUNTRY_FIX = {
    "China (Mini Lunn, Beau Kid)": "China",
    "India (Fabric Flavours)": "India",
    "Korea": "South Korea",
    "TUR": "Turkey",
    "RUS": "Russia",
    "Hong Kong SAR": "Hong Kong",
    "Viet Nam": "Vietnam",
    "USA": "United States",
    "U.S.": "United States",
}
COUNTRY_NO_DENOMINATOR = {
    "United States", "European Union (My Little Pie, Joha)",
}

ARCHETYPES = [
    ("nightgown",        r"night ?gown|nightdress|night ?shirt|sleep ?gown", False),
    ("robe",             r"\brobes?\b|bathrobe|kimono|dressing gown", False),
    ("loungewear",       r"lounge ?wear|lounge ?set|lounge ?pants?", False),
    ("wearable_blanket", r"wearable blanket|sleep ?sack|sleeping bag|swaddle", False),
    ("pajama_set",       r"pajama|pyjama|\bpj\b|pj ?s\b|sleep ?set", True),
    ("sleepwear_generic", r"sleep ?wear", True),
    ("underwear_base",   r"underwear|thermal(s| set| underwear)|long johns", True),
    ("outerwear",        r"jacket|coat|parka|hoodie|sweatshirt|sweater|vest", None),
    ("daywear",          r"\bshirt|\bpants?\b|jeans|shorts|skirt|dress\b|romper|"
                         r"onesie|bodysuit|jumpsuit|overalls|legging", None),
    ("footwear",         r"shoes?|boots?|sandals?|sneakers?|slippers?|footwear", None),
    ("accessory",        r"scarf|scarves|gloves?|mittens?|\bhats?\b|beanie|"
                         r"socks?|tights|hosiery|\bbelt", None),
    ("swimwear",         r"swim|bikini|bathing suit", None),
    ("costume",          r"costume|cosplay", None),
    ("home_textile",     r"blanket|comforter|quilt|bedding|sheet|pillow|curtain|"
                         r"drape|towel|mattress pad|throw", None),
]
SLEEPWEAR_STANDARD_RE = re.compile(
    r"1615|1616|children'?s? sleepwear (?:standard|flammability)|"
    r"federal flammability standard", re.I)


def _first_archetype(text: str):
    t = (text or "").lower()
    for name, pat, tight in ARCHETYPES:
        if re.search(pat, t):
            return name, tight
    return "unclassified", None


_FIRM_PATTERNS = [
    r"^(.{2,70}?)\s+Recalls?\b",
    r"\bRecalled by\s+(.{2,70}?)(?:\s+Due to|\s*[;,]|$)",
    r"\bImported by\s+(.{2,70}?)(?:\s+Due to|\s*[;,]|$)",
    r"^CPSC,?\s+(.{2,70}?)\s+Announce",
]
_FIRM_STRIP = re.compile(
    r"\b(inc|llc|ltd|co|corp|company|group|usa|us|brands?|international|"
    r"industries|enterprises?|trading|import(s|ers?)?|apparel|clothing)\b\.?",
    re.I)


def extract_firm(title: str) -> str | None:
    t = str(title or "")
    for pat in _FIRM_PATTERNS:
        m = re.search(pat, t, re.I)
        if m:
            name = m.group(1).strip(" .,-–—")
            if 2 <= len(name) <= 70:
                return name
    return None


def firm_key(name: str | None) -> str | None:
    if not name or (isinstance(name, float) and np.isnan(name)):
        return None
    k = _FIRM_STRIP.sub(" ", str(name).lower())
    k = re.sub(r"[^a-z0-9 ]", " ", k)
    k = re.sub(r"\s+", " ", k).strip()
    return k or None


WARNING_RE = re.compile(
    r"^CPSC Warns|Product Safety Warning|"
    r"urges consumers to (?:stop|immediately)|"
    r"\bwarns? consumers?\b", re.I)
ACTIVE_RECALL_RE = re.compile(r"\bRecalls\b|\bAnnounce (?:the )?Recall\b", re.I)
PASSIVE_RECALL_RE = re.compile(r"\bRecalled\b", re.I)


def enforcement_mode(title: str, remedy: str | None) -> str:
    t = str(title or "")
    if WARNING_RE.search(t):
        return "unilateral_warning"
    if ACTIVE_RECALL_RE.search(t):
        return "firm_led"
    if PASSIVE_RECALL_RE.search(t):
        return "passive_announced"
    return "other_action"


def load_raw_json() -> pd.DataFrame | None:
    if not RAW_JSON.exists():
        return None
    with open(RAW_JSON, "r", encoding="utf-8", errors="replace") as fh:
        recs = json.load(fh)
    rows = []
    for r in recs:
        rows.append({
            "recall_id": r.get("RecallID"),
            "recall_number": r.get("RecallNumber"),
            "recall_date": r.get("RecallDate"),
            "title": r.get("Title") or "",
            "description": r.get("Description") or "",
            "url": r.get("URL") or "",
            "hazard_text": " ".join(h.get("Name", "")
                                    for h in (r.get("Hazards") or [])),
            "remedy_text": " ".join(m.get("Name", "")
                                    for m in (r.get("Remedies") or [])),
            "remedy_options": ";".join(
                sorted({m.get("Option", "") for m in (r.get("RemedyOptions") or [])
                        if m.get("Option")})),
            "products": " | ".join(p.get("Name", "")
                                   for p in (r.get("Products") or [])),
            "retailers": " | ".join(x.get("Name", "")
                                    for x in (r.get("Retailers") or [])),
            "importers": " | ".join(x.get("Name", "")
                                    for x in (r.get("Importers") or [])),
            "manufacturers": " | ".join(x.get("Name", "")
                                        for x in (r.get("Manufacturers") or [])),
            "distributors": " | ".join(x.get("Name", "")
                                       for x in (r.get("Distributors") or [])),
            "countries_raw": ";".join(
                sorted({c.get("Country", "")
                        for c in (r.get("ManufacturerCountries") or [])
                        if c.get("Country")})),
        })
    df = pd.DataFrame(rows)
    df["recall_id"] = pd.to_numeric(df.recall_id, errors="coerce")
    return df


def load_recalls(prefer_v2: bool = True) -> pd.DataFrame:
    if prefer_v2 and V2_CSV.exists():
        df = pd.read_csv(V2_CSV)
        df.attrs["source_version"] = "v2_hardened"
    else:
        df = pd.read_csv(V1_CSV)
        df.attrs["source_version"] = "v1_clean"
    return add_derived(df)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["recall_date"] = pd.to_datetime(df.recall_date, errors="coerce")
    if "year" not in df:
        df["year"] = df.recall_date.dt.year
    df["period"] = pd.cut(df.year, bins=PERIOD_BINS, labels=PERIOD_LABELS)

    text_cols = [c for c in ["title", "products", "description"] if c in df]
    df["_text"] = df[text_cols].fillna("").agg(" ".join, axis=1)

    if "archetype" not in df:
        arche = df["_text"].map(_first_archetype)
        df["archetype"] = [a for a, _ in arche]
        df["archetype_can_be_tight"] = [t for _, t in arche]

    if "is_false_positive" not in df:
        df["is_false_positive"] = df["_text"].str.contains(FALSE_POSITIVE_RE)
    if "is_electric_textile" not in df:
        df["is_electric_textile"] = df["_text"].str.contains(ELECTRIC_TEXTILE_RE)

    if "sleepwear_standard" not in df:
        df["sleepwear_standard"] = df["_text"].str.contains(SLEEPWEAR_STANDARD_RE)

    if "firm" not in df:
        df["firm"] = df.title.map(extract_firm)
    df["firm_key"] = df.firm.map(firm_key)

    if "enforcement_mode" not in df:
        df["enforcement_mode"] = [
            enforcement_mode(t, r) for t, r in
            zip(df.title, df.get("remedy_options", pd.Series([None] * len(df))))
        ]

    ro = df.get("remedy_options", pd.Series([""] * len(df))).fillna("").str.lower()
    df["remedy_refund"] = ro.str.contains("refund")
    df["remedy_repair"] = ro.str.contains("repair")
    df["remedy_replace"] = ro.str.contains("replace")
    df["remedy_any"] = ro.str.len() > 0

    df["log_units"] = np.log10(pd.to_numeric(df.get("units"), errors="coerce")
                               .replace(0, np.nan))
    return df


def explode_countries(df: pd.DataFrame) -> pd.DataFrame:
    col = "countries" if "countries" in df else "countries_raw"
    d = df[df[col].notna()].copy()
    d["country"] = d[col].astype(str).str.split(";")
    d = d.explode("country")
    d["country"] = d.country.str.strip().replace(COUNTRY_FIX)
    d = d[d.country.str.len() > 0]
    return d


def load_imports() -> pd.DataFrame:
    sme = pd.read_csv(OTEXA_SME).rename(columns={"DATA_VALUE": "sme"})
    usd = pd.read_csv(OTEXA_USD).rename(columns={"DATA_VALUE": "usd"})
    keys = ["Country", "Year"]
    imp = (sme[keys + ["sme"]].merge(usd[keys + ["usd"]], on=keys, how="outer")
           .rename(columns={"Country": "country", "Year": "year"}))
    imp = imp[imp.country.notna()]
    imp = imp[~imp.country.str.startswith("_")]
    return imp.groupby(["country", "year"], as_index=False)[["sme", "usd"]].sum()


def suspect_import_years(imp: pd.DataFrame) -> list[int]:
    w = imp[imp.country == "World"].set_index("year").sme.sort_index()
    out = []
    for y in w.index[1:]:
        prev = w.get(y - 1)
        if prev and abs(w[y] - prev) / prev < 1e-3:
            out.append(int(y))
    return out


def results_dir(analysis_file) -> Path:
    d = Path(analysis_file).resolve().parent / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(obj, path: Path):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        if isinstance(o, pd.Series):
            return o.to_dict()
        raise TypeError(f"not serialisable: {type(o)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=default, ensure_ascii=False)
    return path


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
