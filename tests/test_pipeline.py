"""Integration tests for end-to-end pipeline execution."""

import pathlib

import pandas as pd

from src.data.loader import RAW_DATA_DIR
from src.pipeline import run_pipeline


def test_end_to_end_pipeline_execution(tmp_path: pathlib.Path):
    """Test full pipeline execution producing all data and report artifacts."""
    out_clean_dir = tmp_path / "data" / "cleaned"
    out_models_dir = tmp_path / "models"
    out_reports_dir = tmp_path / "reports"

    raw_path = RAW_DATA_DIR / "telco_churn_raw.csv"

    # Run pipeline in isolated temporary directory
    run_pipeline(
        raw_path=raw_path,
        clean_dir=out_clean_dir,
        models_dir=out_models_dir,
        reports_dir=out_reports_dir,
        skip_llm=True,
    )

    # Assert all artifacts exist
    assert (out_clean_dir / "telco_churn_clean.csv").exists()
    assert (out_clean_dir / "features_matrix.csv").exists()
    assert (out_clean_dir / "customer_risk_profiles.csv").exists()
    assert (out_clean_dir / "scored_customers.csv").exists()

    assert (out_models_dir / "gradient_boosting.pkl").exists()
    assert (out_models_dir / "random_forest.pkl").exists()
    assert (out_models_dir / "logistic_regression.pkl").exists()
    assert (out_models_dir / "scaler.pkl").exists()
    assert (out_models_dir / "thresholds.json").exists()

    assert (out_reports_dir / "executive_briefing.txt").exists()
    assert (out_reports_dir / "segment_recommendations.json").exists()

    # Assert data consistency
    clean_df = pd.read_csv(out_clean_dir / "telco_churn_clean.csv")
    assert len(clean_df) == 7032

    risk_df = pd.read_csv(out_clean_dir / "customer_risk_profiles.csv")
    assert len(risk_df) == 7032
    assert risk_df["risk_score"].min() >= 0
    assert risk_df["risk_score"].max() <= 100
