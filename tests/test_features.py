"""Unit tests for feature engineering transformer."""

import pandas as pd
import pytest

from src.features.engineering import (
    FEATURE_COLUMNS_ORDER,
    FeatureEngineer,
    engineer_features,
)


@pytest.fixture
def clean_sample_df() -> pd.DataFrame:
    """Fixture providing clean customer records."""
    return pd.DataFrame(
        {
            "customerID": ["CUST-1", "CUST-2"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [6, 48],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["Yes", "No"],
            "InternetService": ["Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["Yes", "No"],
            "StreamingMovies": ["Yes", "No"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Credit card (automatic)"],
            "MonthlyCharges": [90.0, 45.0],
            "TotalCharges": [540.0, 2160.0],
            "Churn": [1, 0],
        }
    )


def test_feature_engineer_shape_and_columns(clean_sample_df):
    """Test that FeatureEngineer produces the exact 30 ordered feature columns."""
    fe = FeatureEngineer()
    X, customer_ids, y = fe.fit_transform(clean_sample_df, include_target=True)

    assert X.shape == (2, 30)
    assert list(X.columns) == FEATURE_COLUMNS_ORDER
    assert len(customer_ids) == 2
    assert y.tolist() == [1, 0]
    assert X.isnull().sum().sum() == 0


def test_feature_engineer_engagement_score(clean_sample_df):
    """Test product stickiness engagement score calculation."""
    fe = FeatureEngineer()
    X, _, _ = fe.fit_transform(clean_sample_df)

    # Customer 1 has: MultipleLines(1) + OnlineBackup(1) + StreamingTV(1) + StreamingMovies(1) = 4
    # Customer 2 has: OnlineSecurity(1) + DeviceProtection(1) + TechSupport(1) = 3
    assert X["engagement_score"].iloc[0] == 4
    assert X["engagement_score"].iloc[1] == 3


def test_feature_engineer_high_risk_profile(clean_sample_df):
    """Test business high risk combination profile detection."""
    fe = FeatureEngineer()
    X, _, _ = fe.fit_transform(clean_sample_df)

    # Customer 1: tenure=6 (tenure_group=0), Month-to-month (Contract=0), Electronic check (is_auto_payment=0) -> 1
    # Customer 2: tenure=48 (tenure_group=2), Two year (Contract=2), Credit card (is_auto_payment=1) -> 0
    assert X["is_high_risk_profile"].iloc[0] == 1
    assert X["is_high_risk_profile"].iloc[1] == 0


def test_feature_engineer_handles_missing_optional_columns():
    """Test transformer robustness when optional sub-service columns are omitted."""
    minimal_df = pd.DataFrame(
        {
            "customerID": ["CUST-A"],
            "tenure": [12],
            "MonthlyCharges": [75.0],
            "TotalCharges": [900.0],
            "Contract": ["One year"],
            "PaymentMethod": ["Bank transfer (automatic)"],
            "Churn": [0],
        }
    )
    X, _customer_ids, _y = engineer_features(minimal_df)
    assert X.shape == (1, 30)
    assert X["Contract"].iloc[0] == 1
    assert X["is_auto_payment"].iloc[0] == 1
    assert X["engagement_score"].iloc[0] == 0
