"""Unit tests for ML model training, prediction, scoring, and dynamic what-if simulation."""

import pandas as pd
import pytest

from src.models.predict import (
    CustomerRiskScorer,
    assign_risk_level,
    extract_churn_drivers,
    generate_recommended_action,
)


@pytest.fixture(scope="module")
def scorer() -> CustomerRiskScorer:
    """Fixture providing initialized CustomerRiskScorer instance."""
    return CustomerRiskScorer()


def test_assign_risk_level():
    """Test risk score to tier mapping."""
    assert assign_risk_level(85.0) == "Critical"
    assert assign_risk_level(55.0) == "High"
    assert assign_risk_level(35.0) == "Medium"
    assert assign_risk_level(15.0) == "Low"


def test_extract_churn_drivers():
    """Test driver attribution rule engine."""
    high_risk_row = {
        "Contract": 0,
        "is_auto_payment": 0,
        "tenure_group": 0,
        "MonthlyCharges": 95.0,
    }
    drivers = extract_churn_drivers(high_risk_row)
    assert "Month-to-month contract" in drivers
    assert "Manual payment method" in drivers
    assert "New customer (0-12 months)" in drivers


def test_generate_recommended_action():
    """Test prescriptive playbook generation."""
    action = generate_recommended_action(
        "Critical", {"Contract": 0, "MonthlyCharges": 85.0}
    )
    assert "Personal call within 24 hours" in action
    assert "annual plan" in action


def test_scorer_predict_and_score_dataframe(scorer):
    """Test scoring a batch dataframe produces valid probabilities and complete profiles."""
    test_df = pd.DataFrame(
        {
            "customerID": ["TEST-1", "TEST-2"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["No", "Yes"],
            "Dependents": ["No", "No"],
            "tenure": [2, 60],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["No", "Yes"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Credit card (automatic)"],
            "MonthlyCharges": [95.0, 45.0],
            "TotalCharges": [190.0, 2700.0],
            "Churn": [1, 0],
        }
    )

    scored_df = scorer.score_dataframe(test_df)
    assert len(scored_df) == 2
    assert "risk_score" in scored_df.columns
    assert "ensemble_probability" in scored_df.columns
    assert "churn_drivers" in scored_df.columns
    assert "recommended_action" in scored_df.columns

    # High-risk customer 1 should score higher than loyal customer 2
    cust1 = scored_df[scored_df["customerID"] == "TEST-1"].iloc[0]
    cust2 = scored_df[scored_df["customerID"] == "TEST-2"].iloc[0]
    assert cust1["risk_score"] > cust2["risk_score"]
    assert cust1["risk_level"] in ["Critical", "High"]
    assert cust2["risk_level"] in ["Low", "Medium"]


def test_simulate_what_if_dynamic(scorer):
    """Test dynamic ML model inference simulation on customer parameter modifications."""
    cust_dict = {
        "customerID": "SIM-01",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 90.0,
        "TotalCharges": 270.0,
    }

    # Simulate upgrading from Month-to-month to Two year and auto-payment
    res = scorer.simulate_what_if(
        cust_dict, new_contract="Two year", new_payment="Bank transfer (automatic)"
    )

    assert res["base_score"] > 0
    assert res["new_score"] < res["base_score"]
    assert res["score_delta"] < 0
    assert res["new_probability"] < res["base_probability"]
    assert len(res["reasons"]) == 2
