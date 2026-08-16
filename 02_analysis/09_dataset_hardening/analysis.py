from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import data as D

RES = D.results_dir(__file__)


def main():
    v1 = pd.read_csv(D.V1_CSV)
    n0 = len(v1)

    raw = D.load_raw_json()
    enriched = False
    if raw is not None:
        keep = ["recall_id", "description", "hazard_text", "remedy_text",
                "products", "retailers", "importers", "manufacturers",
                "distributors"]
        v1 = v1.merge(raw[keep], on="recall_id", how="left")
        enriched = True

    df = D.add_derived(v1)

    removed = df[df.is_false_positive].copy()
    removed["removal_reason"] = "non_textile_false_positive"
    df = df[~df.is_false_positive].copy()

    def fix_countries(s):
        if pd.isna(s):
            return s
        parts = [D.COUNTRY_FIX.get(p.strip(), p.strip()) for p in str(s).split(";")]
        parts = [p for p in dict.fromkeys(parts) if p]
        return ";".join(parts) if parts else np.nan

    before_ctry = df.countries.fillna("").str.split(";").explode().str.strip()
    df["countries"] = df.countries.map(fix_countries)
    df["primary_country"] = df.primary_country.map(
        lambda s: D.COUNTRY_FIX.get(str(s).strip(), s) if pd.notna(s) else s)
    after_ctry = df.countries.fillna("").str.split(";").explode().str.strip()
    country_delta = {
        "distinct_before": int(before_ctry[before_ctry != ""].nunique()),
        "distinct_after": int(after_ctry[after_ctry != ""].nunique()),
        "merged": sorted(set(before_ctry) - set(after_ctry) - {""}),
    }

    df["boundary_class"] = np.select(
        [df.archetype_can_be_tight.eq(True),
         df.archetype_can_be_tight.eq(False)],
        ["exemption_eligible", "exemption_ineligible"],
        default="not_sleepwear",
    )
    df["category_arbitrage"] = (
        df.sleepwear_standard
        & df.archetype.isin(["loungewear", "robe", "wearable_blanket",
                             "daywear", "outerwear"])
    )

    queue_src = (D.ANALYSIS / "12_boundary_archetypes" / "results"
                 / "nontextile_candidates.csv")
    if queue_src.exists():
        q = pd.read_csv(queue_src)
        df["is_likely_nontextile"] = df.recall_id.isin(q.recall_id)
        q.to_csv(RES / "nontextile_review_queue.csv", index=False)
    else:
        df["is_likely_nontextile"] = False

    df["has_remedy"] = df.remedy_any
    df["no_remedy_offered"] = (~df.remedy_any) & (df.year >= 2010)

    def flam_share(frame, lo, hi):
        w = frame[frame.year.between(lo, hi)]
        return float((w.hazard_category == "flammability_burn").mean()) if len(w) else np.nan

    v1d = D.add_derived(pd.read_csv(D.V1_CSV))
    headline = {
        "flammability_share_2020_25_v1": flam_share(v1d, 2020, 2025),
        "flammability_share_2020_25_v2": flam_share(df, 2020, 2025),
        "flammability_share_2000_09_v1": flam_share(v1d, 2000, 2009),
        "flammability_share_2000_09_v2": flam_share(df, 2000, 2009),
        "n_v1": n0, "n_v2": int(len(df)),
    }

    fields = {
        "units": df.units.notna(),
        "price_usd": df.price_usd.notna(),
        "country of manufacture": df.countries.notna(),
        "sales channel": df.sales_channel.ne("unknown"),
        "remedy options": df.remedy_any,
        "fiber content": df.main_fiber.notna() if "main_fiber" in df else pd.Series(False, index=df.index),
        "description text": (df.description.notna() if "description" in df
                             else pd.Series(False, index=df.index)),
    }
    cov = pd.DataFrame({k: v for k, v in fields.items()})
    cov["period"] = df.period.values
    coverage = cov.groupby("period", observed=True).mean().reset_index()
    counts = df.groupby("period", observed=True).size().rename("n").reset_index()
    coverage = coverage.merge(counts, on="period")

    out_cols = [c for c in df.columns if not c.startswith("_")]
    df[out_cols].to_csv(RES / "apparel_recalls_v2.csv", index=False)
    removed[["recall_id", "year", "title", "hazard_category", "removal_reason"]] \
        .to_csv(RES / "removed_records.csv", index=False)
    coverage.to_csv(RES / "field_coverage.csv", index=False)

    summary = {
        "analysis": "09_dataset_hardening",
        "source_enriched_with_raw_json": enriched,
        "n_input": n0,
        "n_removed_false_positive": int(len(removed)),
        "removed_titles": removed.title.tolist(),
        "n_output": int(len(df)),
        "n_electric_textile_flagged": int(df.is_electric_textile.sum()),
        "country_normalisation": country_delta,
        "headline_sensitivity": headline,
        "archetype_counts": df.archetype.value_counts().to_dict(),
        "boundary_class_counts": df.boundary_class.value_counts().to_dict(),
        "category_arbitrage_total": int(df.category_arbitrage.sum()),
        "category_arbitrage_by_period":
            df.groupby("period", observed=True).category_arbitrage.sum().to_dict(),
        "enforcement_mode_counts": df.enforcement_mode.value_counts().to_dict(),
        "enforcement_mode_2020_25":
            df[df.year >= 2020].enforcement_mode.value_counts().to_dict(),
        "firm_parse_rate": float(df.firm_key.notna().mean()),
        "n_likely_nontextile_flagged": int(df.is_likely_nontextile.sum()),
        "flammability_share_2020_25_if_queue_dropped": float(
            (df[(~df.is_likely_nontextile) & df.year.between(2020, 2025)]
             .hazard_category == "flammability_burn").mean()),
        "coverage_by_period": coverage.to_dict(orient="records"),
    }
    D.write_json(summary, RES / "hardening_summary.json")

    print(f"[09] {n0} -> {len(df)} records "
          f"({len(removed)} false positives removed)")
    print(f"[09] flammability share 2020-25: "
          f"{headline['flammability_share_2020_25_v1']:.3f} -> "
          f"{headline['flammability_share_2020_25_v2']:.3f}")
    print(f"[09] category arbitrage records: {summary['category_arbitrage_total']}")
    print(f"[09] enforcement modes: {summary['enforcement_mode_counts']}")
    return summary


if __name__ == "__main__":
    main()
