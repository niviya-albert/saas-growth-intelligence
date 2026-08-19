"""Unit tests for SaaS KPI engine, health score, and ROI financial calculations."""

import pandas as pd
import pytest

from src.metrics.saas_kpis import (
    calculate_reduction_scenarios,
    calculate_retention_roi,
    compute_business_health_score,
    compute_saas_kpis,
)


@pytest.fixture
def kpi_test_df() -> pd.DataFrame:
    """Fixture providing known dataset for KPI assertions."""
    return pd.DataFrame(
        {
            "customerID": ["1", "2", "3", "4"],
            "MonthlyCharges": [100.0, 50.0, 80.0, 70.0],
            "tenure": [10, 20, 5, 15],
            "Churn": [0, 0, 1, 1],  # 2 active ($150 MRR), 2 churned ($150 Churn MRR)
        }
    )


def test_compute_saas_kpis(kpi_test_df):
    """Test core SaaS KPI math: MRR, ARR, Churn Rate, LTV, and Retention."""
    kpis = compute_saas_kpis(kpi_test_df, default_cac=300.0)

    assert kpis["total_customers"] == 4
    assert kpis["total_churned"] == 2
    assert kpis["churn_rate"] == 50.0
    assert kpis["mrr"] == 150.0
    assert kpis["arr"] == 1800.0
    assert kpis["churn_mrr"] == 150.0
    assert kpis["churn_arr"] == 1800.0
    assert kpis["avg_monthly"] == 75.0
    assert kpis["rev_retention"] == 50.0
    # Average tenure of churned (5 + 15)/2 = 10.0
    # LTV = 75.0 * 10.0 = 750.0
    assert kpis["ltv"] == 750.0
    # LTV:CAC = 750 / 300 = 2.5
    assert kpis["ltv_cac"] == 2.5


def test_compute_business_health_score():
    """Test health score gauge calculation and thresholds."""
    # Scenario 1: Healthy business
    healthy = compute_business_health_score(churn_rate=5.0, rev_retention=92.0, ltv_cac=4.5)
    assert healthy["health_score"] >= 70.0
    assert healthy["status"] == "Healthy"

    # Scenario 2: Critical business
    critical = compute_business_health_score(churn_rate=30.0, rev_retention=60.0, ltv_cac=1.5)
    assert critical["health_score"] < 50.0
    assert critical["status"] == "Critical"


def test_calculate_reduction_scenarios():
    """Test scenario calculation table generation."""
    scen = calculate_reduction_scenarios(
        total_churned=100, avg_monthly=50.0, churn_rate=20.0, mrr=10000.0, targets=[10, 20, 30]
    )
    assert len(scen) == 3
    # 20% target on 100 churners = 20 saved
    row_20 = scen[scen["Reduction %"] == 20].iloc[0]
    assert row_20["Customers Saved"] == 20
    # 20 saved * $50/mo * 12 = $12,000
    assert row_20["Revenue Saved"] == 12000.0
    assert row_20["New Churn Rate"] == 16.0


def test_calculate_retention_roi():
    """Test retention ROI and payback period calculator."""
    roi = calculate_retention_roi(revenue_saved=120000.0, programme_cost=40000.0)
    assert roi["net_gain"] == 80000.0
    assert roi["roi_pct"] == 200.0
    # Monthly savings = $10,000. Payback = $40,000 / $10,000 = 4.0 months
    assert roi["payback_months"] == 4.0
