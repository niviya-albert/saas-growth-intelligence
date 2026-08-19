"""SaaS KPI computation, business health scoring, and retention financial modeling."""

from typing import Any

import pandas as pd


def compute_saas_kpis(
    df: pd.DataFrame, default_cac: float = 300.0
) -> dict[str, Any]:
    """Compute standard SaaS revenue, retention, and unit economic KPIs.

    Args:
        df: Cleaned customer DataFrame containing Churn, MonthlyCharges, tenure.
        default_cac: Customer Acquisition Cost benchmark in dollars.

    Returns:
        Dictionary of formatted and raw SaaS metric values.
    """
    total_customers = len(df)
    if total_customers == 0:
        return {
            "total_customers": 0,
            "total_churned": 0,
            "churn_rate": 0.0,
            "mrr": 0.0,
            "arr": 0.0,
            "churn_mrr": 0.0,
            "churn_arr": 0.0,
            "avg_monthly": 0.0,
            "avg_tenure_churned": 0.0,
            "ltv": 0.0,
            "ltv_cac": 0.0,
            "rev_retention": 0.0,
        }

    total_churned = int(df["Churn"].sum())
    churn_rate = round(float(df["Churn"].mean() * 100), 2)

    active_df = df[df["Churn"] == 0]
    churned_df = df[df["Churn"] == 1]

    mrr = round(float(active_df["MonthlyCharges"].sum()), 2)
    arr = round(float(mrr * 12), 2)

    churn_mrr = round(float(churned_df["MonthlyCharges"].sum()), 2)
    churn_arr = round(float(churn_mrr * 12), 2)

    avg_monthly = round(float(df["MonthlyCharges"].mean()), 2)

    # Average tenure of churned customers
    avg_tenure_churned = (
        round(float(churned_df["tenure"].mean()), 1)
        if len(churned_df) > 0
        else round(float(df["tenure"].mean()), 1)
    )

    # LTV = average monthly charges * average lifespan (tenure before churn)
    ltv = round(float(avg_monthly * avg_tenure_churned), 2)
    ltv_cac = round(float(ltv / default_cac), 2) if default_cac > 0 else 0.0

    # Revenue Retention %
    total_revenue_base = mrr + churn_mrr
    rev_retention = (
        round(float((mrr / total_revenue_base) * 100), 2)
        if total_revenue_base > 0
        else 0.0
    )

    return {
        "total_customers": total_customers,
        "total_churned": total_churned,
        "churn_rate": churn_rate,
        "mrr": mrr,
        "arr": arr,
        "churn_mrr": churn_mrr,
        "churn_arr": churn_arr,
        "avg_monthly": avg_monthly,
        "avg_tenure_churned": avg_tenure_churned,
        "ltv": ltv,
        "cac": default_cac,
        "ltv_cac": ltv_cac,
        "rev_retention": rev_retention,
    }


def compute_business_health_score(
    churn_rate: float, rev_retention: float, ltv_cac: float
) -> dict[str, Any]:
    """Calculate the 0-100 composite Business Health Score.

    Weighting:
    - 40% Churn Rate Score (lower churn = higher score)
    - 35% Revenue Retention Score (higher retention = higher score)
    - 25% LTV:CAC Score (higher ratio = higher score, capped at 100)
    """
    churn_score = max(0.0, 100.0 - (churn_rate * 2.5))
    retention_score = min(100.0, max(0.0, rev_retention))
    ltv_score = min(100.0, max(0.0, ltv_cac * 20.0))

    health_score = round(
        churn_score * 0.4 + retention_score * 0.35 + ltv_score * 0.25, 1
    )

    if health_score < 50.0:
        status = "Critical"
        colour = "#e74c3c"
    elif health_score < 70.0:
        status = "Needs Improvement"
        colour = "#e67e22"
    else:
        status = "Healthy"
        colour = "#27ae60"

    breakdown = [
        {
            "Metric": "Churn Rate Score (40%)",
            "Score": round(churn_score, 1),
            "Status": "Critical" if churn_score < 50 else ("Needs Work" if churn_score < 70 else "Good"),
        },
        {
            "Metric": "Revenue Retention (35%)",
            "Score": round(retention_score, 1),
            "Status": "Critical" if retention_score < 70 else ("Needs Work" if retention_score < 85 else "Good"),
        },
        {
            "Metric": "LTV:CAC Score (25%)",
            "Score": round(ltv_score, 1),
            "Status": "Marginal" if ltv_score < 60 else "Good",
        },
    ]

    return {
        "health_score": health_score,
        "status": status,
        "colour": colour,
        "churn_score": churn_score,
        "retention_score": retention_score,
        "ltv_score": ltv_score,
        "breakdown": breakdown,
    }


def calculate_reduction_scenarios(
    total_churned: int,
    avg_monthly: float,
    churn_rate: float,
    mrr: float,
    targets: list[int] | None = None,
) -> pd.DataFrame:
    """Generate financial projections across different churn reduction percentages."""
    target_list = targets or list(range(5, 80, 5))
    records = []
    for pct in target_list:
        saved = round(total_churned * (pct / 100.0))
        rev_saved = round(saved * avg_monthly * 12.0, 0)
        new_churn_rate = round(churn_rate * (1.0 - pct / 100.0), 2)
        new_mrr = round(mrr + (saved * avg_monthly), 0)
        records.append(
            {
                "Reduction %": pct,
                "Customers Saved": saved,
                "Revenue Saved": rev_saved,
                "New Churn Rate": new_churn_rate,
                "New MRR": new_mrr,
            }
        )
    return pd.DataFrame(records)


def calculate_retention_roi(
    revenue_saved: float, programme_cost: float
) -> dict[str, Any]:
    """Calculate Net Revenue Gain, ROI percentage, and Payback Period in months."""
    if programme_cost <= 0:
        return {
            "net_gain": revenue_saved,
            "roi_pct": 0.0,
            "payback_months": 0.0,
        }

    net_gain = round(revenue_saved - programme_cost, 2)
    roi_pct = round((net_gain / programme_cost) * 100.0, 1)
    monthly_savings = revenue_saved / 12.0
    payback_months = (
        round(programme_cost / monthly_savings, 1)
        if monthly_savings > 0
        else 0.0
    )

    return {
        "net_gain": net_gain,
        "roi_pct": roi_pct,
        "payback_months": payback_months,
    }
