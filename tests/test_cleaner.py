"""Unit tests for data validation and cleaning pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import clean_customer_data, validate_schema


@pytest.fixture
def sample_valid_raw_df() -> pd.DataFrame:
    """Fixture providing valid customer records."""
    return pd.DataFrame(
        {
            "customerID": ["CUST-001", "CUST-002", "CUST-003", "CUST-004", "CUST-005"],
            "gender": ["Female", "Male", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0, 1],
            "Partner": ["Yes", "No", "Yes", "No", "No"],
            "Dependents": ["No", "No", "Yes", "No", "No"],
            "tenure": [12, 1, 48, 24, 6],
            "PhoneService": ["Yes", "No", "Yes", "Yes", "Yes"],
            "MultipleLines": ["Yes", "No phone service", "No", "Yes", "No"],
            "InternetService": ["Fiber optic", "DSL", "No", "Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "Yes", "No internet service", "Yes", "No"],
            "OnlineBackup": ["Yes", "No", "No internet service", "Yes", "No"],
            "DeviceProtection": ["No", "Yes", "No internet service", "No", "No"],
            "TechSupport": ["No", "No", "No internet service", "Yes", "No"],
            "StreamingTV": ["Yes", "No", "No internet service", "Yes", "No"],
            "StreamingMovies": ["Yes", "No", "No internet service", "No", "No"],
            "Contract": [
                "Month-to-month",
                "Month-to-month",
                "Two year",
                "One year",
                "Month-to-month",
            ],
            "PaperlessBilling": ["Yes", "No", "No", "Yes", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
            ],
            "MonthlyCharges": [85.50, 45.00, 25.00, 70.00, 55.00],
            "TotalCharges": [1026.00, 45.00, 1200.00, 1680.00, 330.00],
            "Churn": ["Yes", "No", "No", "No", "Yes"],
        }
    )


def test_validate_schema_success(sample_valid_raw_df):
    """Test schema validation on complete valid dataset."""
    errors = validate_schema(sample_valid_raw_df)
    assert errors == []


def test_validate_schema_missing_columns():
    """Test schema validation with missing mandatory columns."""
    incomplete_df = pd.DataFrame({"customerID": ["CUST-1"], "tenure": [10]})
    errors = validate_schema(incomplete_df)
    assert len(errors) == 1
    assert "Missing required columns" in errors[0]


def test_clean_customer_data_success(sample_valid_raw_df):
    """Test successful data cleaning, type coercion, and Churn mapping."""
    clean_df, errors, dropped = clean_customer_data(sample_valid_raw_df, min_rows=1)
    assert errors == []
    assert dropped == 0
    assert len(clean_df) == 5
    assert clean_df["Churn"].dtype == np.int64 or clean_df["Churn"].dtype == int
    assert clean_df["Churn"].tolist() == [1, 0, 0, 0, 1]
    assert clean_df["MonthlyCharges"].dtype == np.float64 or clean_df["MonthlyCharges"].dtype == float


def test_clean_customer_data_handles_blank_total_charges(sample_valid_raw_df):
    """Test handling of blank string TotalCharges and dropping invalid records."""
    df_with_blank = sample_valid_raw_df.copy()
    df_with_blank["TotalCharges"] = df_with_blank["TotalCharges"].astype(object)
    df_with_blank.loc[0, "TotalCharges"] = "   "
    df_with_blank.loc[0, "tenure"] = 0

    clean_df, errors, dropped = clean_customer_data(df_with_blank, min_rows=1)
    assert errors == []
    assert dropped == 1
    assert len(clean_df) == 4
    assert "CUST-001" not in clean_df["customerID"].values


def test_clean_customer_data_various_churn_formats():
    """Test various string representations of Churn (Yes, No, TRUE, 1, 0)."""
    raw_df = pd.DataFrame(
        {
            "customerID": ["C1", "C2", "C3", "C4"],
            "tenure": [5, 10, 15, 20],
            "MonthlyCharges": [50.0, 60.0, 70.0, 80.0],
            "TotalCharges": [250.0, 600.0, 1050.0, 1600.0],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            "Churn": ["true", "FALSE", "1", "0"],
        }
    )
    clean_df, errors, _dropped = clean_customer_data(raw_df, min_rows=1)
    assert errors == []
    assert clean_df["Churn"].tolist() == [1, 0, 1, 0]
