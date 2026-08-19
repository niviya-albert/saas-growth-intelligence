"""End-to-End SaaS Growth Intelligence Pipeline CLI Orchestrator.

Executes data cleaning, feature engineering, ML model training, customer risk scoring,
and AI executive briefing generation in a single reproducible command.
"""

import argparse
import pathlib
import sys
import time

import pandas as pd

from .data.cleaner import clean_customer_data
from .data.loader import (
    CLEAN_DATA_DIR,
    RAW_DATA_DIR,
)
from .features.engineering import FeatureEngineer
from .llm.briefing import (
    REPORTS_DIR,
    build_metrics_summary,
    generate_executive_briefing,
)
from .models.predict import score_customer_dataframe
from .models.train import MODELS_DIR, train_all_models


def run_pipeline(
    raw_path: pathlib.Path | None = None,
    clean_dir: pathlib.Path | None = None,
    models_dir: pathlib.Path | None = None,
    reports_dir: pathlib.Path | None = None,
    skip_llm: bool = False,
) -> None:
    """Execute the full end-to-end analytics and ML pipeline."""
    start_time = time.time()
    raw_file = raw_path or (RAW_DATA_DIR / "telco_churn_raw.csv")
    out_clean_dir = clean_dir or CLEAN_DATA_DIR
    out_models_dir = models_dir or MODELS_DIR
    out_reports_dir = reports_dir or REPORTS_DIR

    out_clean_dir.mkdir(parents=True, exist_ok=True)
    out_models_dir.mkdir(parents=True, exist_ok=True)
    out_reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(">> SAAS GROWTH INTELLIGENCE -- END-TO-END PIPELINE")
    print("=" * 70)

    # ── STEP 1: LOAD & CLEAN RAW DATA ─────────────────────────
    print(f"\n[1/5] Loading and cleaning raw data from: {raw_file}")
    if not raw_file.exists():
        print(f"[!] Error: Raw data file not found at {raw_file}")
        sys.exit(1)

    raw_df = pd.read_csv(raw_file)
    clean_df, errors, dropped = clean_customer_data(raw_df, min_rows=100)
    if errors:
        print(f"[!] Cleaning validation errors: {errors}")
        sys.exit(1)

    clean_file = out_clean_dir / "telco_churn_clean.csv"
    clean_df.to_csv(clean_file, index=False)
    print(f"  [+] Cleaned dataset saved: {clean_file}")
    print(f"  [+] Records: {len(clean_df):,} (dropped {dropped} invalid rows)")

    # ── STEP 2: FEATURE ENGINEERING ───────────────────────────
    print("\n[2/5] Running feature engineering pipeline...")
    fe = FeatureEngineer()
    X, customer_ids, y = fe.fit_transform(clean_df, include_target=True)

    # Save features matrix
    fm_df = X.copy()
    fm_df["customerID"] = customer_ids
    fm_df["Churn"] = y
    fm_file = out_clean_dir / "features_matrix.csv"
    fm_df.to_csv(fm_file, index=False)
    print(f"  [+] Feature matrix saved: {fm_file} (shape: {fm_df.shape})")

    # ── STEP 3: TRAIN & EVALUATE ML MODELS ────────────────────
    print("\n[3/5] Training machine learning models (LR, RF, GB) & calibrating thresholds...")
    train_results = train_all_models(
        clean_df, save_artifacts=True, models_output_dir=out_models_dir
    )

    evals = train_results["evaluations"]
    for m_eval in evals.values():
        print(
            f"  [+] {m_eval['name']:<22} | Threshold: {m_eval['threshold']} | "
            f"Recall: {m_eval['recall']}% | AUC: {m_eval['auc_roc']}% | "
            f"Caught: {m_eval['tp']}/{m_eval['tp'] + m_eval['fn']}"
        )

    # ── STEP 4: ENSEMBLE RISK SCORING & PROFILES ──────────────
    print("\n[4/5] Scoring customer base and generating risk profiles...")
    risk_df = score_customer_dataframe(clean_df, models_dir=out_models_dir)

    risk_file = out_clean_dir / "customer_risk_profiles.csv"
    risk_df.to_csv(risk_file, index=False)

    # Scored customers test split format
    test_idx = train_results["test_data"]["y_test"].index
    scored_test_df = risk_df.loc[risk_df["customerID"].isin(clean_df.loc[test_idx, "customerID"])].copy()
    scored_file = out_clean_dir / "scored_customers.csv"
    scored_test_df.to_csv(scored_file, index=False)

    crit_count = (risk_df["risk_level"] == "Critical").sum()
    crit_rev = risk_df[risk_df["risk_level"] == "Critical"]["annual_revenue_risk"].sum()
    print(f"  [+] Customer risk profiles saved: {risk_file}")
    print(f"  [+] Critical risk accounts: {crit_count:,} (${crit_rev:,.0f} ARR at risk)")

    # ── STEP 5: AI EXECUTIVE BRIEFING GENERATION ──────────────
    print("\n[5/5] Generating AI Executive Briefing & Segment Recommendations...")
    metrics = build_metrics_summary(clean_df, risk_df)

    if skip_llm:
        print("  [*] Skipping live LLM call (--skip-llm requested), using template.")
        from .llm.briefing import _generate_fallback_briefing
        briefing_text, recs_dict = _generate_fallback_briefing(metrics)
    else:
        briefing_text, recs_dict = generate_executive_briefing(metrics)

    briefing_file = out_reports_dir / "executive_briefing.txt"
    recs_file = out_reports_dir / "segment_recommendations.json"

    with open(briefing_file, "w", encoding="utf-8") as f:
        f.write(briefing_text)

    import json
    with open(recs_file, "w", encoding="utf-8") as f:
        json.dump(recs_dict, f, indent=2)

    print(f"  [+] Executive briefing saved: {briefing_file}")
    print(f"  [+] Segment recommendations saved: {recs_file}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f">> PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
    print("=" * 70)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end SaaS Growth Intelligence Pipeline"
    )
    parser.add_argument(
        "--raw",
        type=pathlib.Path,
        default=None,
        help="Path to raw customer CSV file",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip live LLM generation and use template fallback",
    )
    args = parser.parse_args()
    run_pipeline(raw_path=args.raw, skip_llm=args.skip_llm)


if __name__ == "__main__":
    main()
