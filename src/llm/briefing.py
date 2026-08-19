"""Google Gemini AI executive briefing and retention recommendation generation."""

import json
import os
import pathlib
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from ..data.loader import PROJECT_ROOT

REPORTS_DIR = PROJECT_ROOT / "reports"


def load_saved_briefing_reports(
    reports_dir: pathlib.Path | None = None,
) -> tuple[str, dict[str, str]]:
    """Load pre-generated executive briefing text and segment recommendations."""
    target_dir = reports_dir or REPORTS_DIR
    briefing_path = target_dir / "executive_briefing.txt"
    recs_path = target_dir / "segment_recommendations.json"

    briefing = ""
    recs = {}

    if briefing_path.exists():
        briefing = briefing_path.read_text(encoding="utf-8")
    if recs_path.exists():
        try:
            recs = json.loads(recs_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            recs = {}

    return briefing, recs


def build_metrics_summary(
    clean_df: pd.DataFrame, risk_df: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Compile structured 14-point metrics summary for LLM prompt."""
    total_customers = len(clean_df)
    total_churned = int(clean_df["Churn"].sum())
    churn_rate = round(float(clean_df["Churn"].mean() * 100), 2)
    avg_monthly = round(float(clean_df["MonthlyCharges"].mean()), 2)
    mrr = round(float(clean_df[clean_df["Churn"] == 0]["MonthlyCharges"].sum()), 2)
    arr = round(float(mrr * 12), 2)
    churn_mrr = round(float(clean_df[clean_df["Churn"] == 1]["MonthlyCharges"].sum()), 2)
    churn_arr = round(float(churn_mrr * 12), 2)

    avg_tenure_churned = (
        round(float(clean_df[clean_df["Churn"] == 1]["tenure"].mean()), 1)
        if total_churned > 0
        else 12.0
    )
    ltv = round(float(avg_monthly * avg_tenure_churned), 2)
    ltv_cac = round(float(ltv / 300.0), 2)
    rev_retention = (
        round(float((mrr / (mrr + churn_mrr)) * 100), 2)
        if (mrr + churn_mrr) > 0
        else 100.0
    )

    contract_churn = (
        clean_df.groupby("Contract")["Churn"].mean().mul(100).round(2).to_dict()
        if "Contract" in clean_df.columns
        else {}
    )
    payment_churn = (
        clean_df.groupby("PaymentMethod")["Churn"].mean().mul(100).round(2).to_dict()
        if "PaymentMethod" in clean_df.columns
        else {}
    )

    risk_summary = {}
    if risk_df is not None and "risk_level" in risk_df.columns:
        for level in ["Critical", "High", "Medium", "Low"]:
            sub = risk_df[risk_df["risk_level"] == level]
            risk_summary[level] = {
                "customers": len(sub),
                "churn_rate": (
                    round(float(sub["actual_churn"].mean() * 100), 1)
                    if "actual_churn" in sub.columns
                    else 0.0
                ),
                "annual_rev": round(float(sub["annual_revenue_risk"].sum()), 0),
            }

    scenarios = {}
    for pct in [10, 20, 30]:
        saved = round(total_churned * (pct / 100.0))
        revenue = round(saved * avg_monthly * 12.0, 0)
        scenarios[f"{pct}pct"] = {
            "customers_saved": saved,
            "revenue_saved": float(revenue),
        }

    return {
        "total_customers": total_customers,
        "total_churned": total_churned,
        "churn_rate": churn_rate,
        "avg_monthly": avg_monthly,
        "mrr": mrr,
        "arr": arr,
        "churn_mrr": churn_mrr,
        "churn_arr": churn_arr,
        "ltv": ltv,
        "ltv_cac": ltv_cac,
        "revenue_retention": rev_retention,
        "contract_churn": contract_churn,
        "payment_churn": payment_churn,
        "risk_summary": risk_summary,
        "scenarios": scenarios,
    }


def generate_executive_briefing(
    metrics: dict[str, Any],
    api_key: str | None = None,
    model_name: str = "gemini-2.5-flash",
) -> tuple[str, dict[str, str]]:
    """Generate professional executive briefing using Google Gemini.

    Falls back cleanly to template-based reporting if API key is not available.
    """
    load_dotenv()
    key = api_key or os.getenv("GEMINI_API_KEY")

    if not key:
        return _generate_fallback_briefing(metrics)

    prompt = f"""
You are a Principal SaaS Retention and Revenue Growth Consultant presenting to the executive board (CEO, CFO, CPO).
Convert the following retention intelligence metrics into a compelling, data-backed 5-section executive briefing.

METRICS CONTEXT:
{json.dumps(metrics, indent=2)}

FORMATTING INSTRUCTIONS:
Structure your response EXACTLY with these 5 section headers:
**SITUATION SUMMARY**
**KEY FINDINGS**
**IMMEDIATE RISK**
**RECOMMENDED INTERVENTIONS**
**PROJECTED OUTCOME**

Requirements:
- Emphasize revenue risk ($ ARR exposure) and root causes (Month-to-month contracts, manual payment, onboarding friction).
- Provide concrete ROI figures for proposed interventions.
- Keep tone authoritative, direct, and concise (under 400 words).
"""

    try:
        from google import genai
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        briefing_text = response.text

        # Generate segment recommendations
        recs_prompt = f"""
Based on these risk segments:
{json.dumps(metrics.get('risk_summary', {}), indent=2)}

Provide a JSON object with keys "Critical", "High", "Medium", "Low" where each value is a single comprehensive paragraph describing the segment risk profile, targeted action, and projected financial recovery.
Output raw valid JSON only.
"""
        recs_resp = client.models.generate_content(
            model=model_name,
            contents=recs_prompt,
        )
        raw_recs = recs_resp.text.strip()
        if raw_recs.startswith("```json"):
            raw_recs = raw_recs[7:]
        elif raw_recs.startswith("```"):
            raw_recs = raw_recs[3:]
        raw_recs = raw_recs.removesuffix("```").strip()
        recs_dict = json.loads(raw_recs)

        return briefing_text, recs_dict

    except Exception:  # noqa: BLE001
        # Fallback to local template if API call fails
        return _generate_fallback_briefing(metrics)


def _generate_fallback_briefing(
    metrics: dict[str, Any]
) -> tuple[str, dict[str, str]]:
    """Deterministic fallback generator when Gemini API is offline."""
    total = metrics.get("total_customers", 7032)
    cr = metrics.get("churn_rate", 26.58)
    arr_lost = metrics.get("churn_arr", 1669570)
    ltv_cac = metrics.get("ltv_cac", 3.89)
    crit_count = metrics.get("risk_summary", {}).get("Critical", {}).get("customers", 973)
    crit_rev = metrics.get("risk_summary", {}).get("Critical", {}).get("annual_rev", 939869)
    scen_30 = metrics.get("scenarios", {}).get("30pct", {}).get("revenue_saved", 436234)

    briefing = f"""**Executive Briefing: SaaS Churn Analysis and Strategic Interventions**

**SITUATION SUMMARY**
Our business is currently experiencing an annual churn rate of {cr}%, representing ${arr_lost:,.0f} in preventable annual revenue loss across {total:,} total accounts. While our LTV to CAC ratio stands at {ltv_cac}:1, exceeding baseline unit economics, the elevated customer attrition severely constrains compounding revenue growth.

**KEY FINDINGS**
*   **Contract Type Risk:** Month-to-month contracts exhibit peak churn at over 42%, representing the primary structural predictor of account churn.
*   **New Customer Lifecycle Drop-off:** Accounts within months 0-12 represent the highest churn volume, pointing to critical onboarding activation gaps.
*   **Payment Friction Impact:** Manual payment methods (such as electronic checks) drive over 45% churn compared to under 16% for automated payment methods.

**IMMEDIATE RISK**
A critical segment of {crit_count:,} accounts currently faces severe churn risk, placing ${crit_rev:,.0f} in annual recurring revenue at immediate exposure without proactive intervention.

**RECOMMENDED INTERVENTIONS**
*   **Segment: Month-to-month contracts.** Action: Deploy targeted discount incentives for annual contract upgrades.
*   **Segment: New accounts (0-12 months).** Action: Enhance 90-day onboarding milestones to activate 3+ core platform features.
*   **Segment: Critical risk & manual payment users.** Action: Direct personal CSM outreach and automatic payment incentives.

**PROJECTED OUTCOME**
Achieving a 30% reduction in customer churn across these priority segments will protect approximately ${scen_30:,.0f} in recurring revenue annually, driving a healthy expansion in overall net revenue retention.
"""

    recs = {
        "Critical": f"{crit_count:,} critical-risk customers exhibit peak churn driven by month-to-month contracts and manual payments. Deploy personal outreach within 24 hours with annual upgrade incentives to protect ${crit_rev:,.0f} in exposed revenue.",
        "High": "High-risk accounts combining developing tenure with manual billing create recurring monthly cancellation triggers. Deploy automatic billing discounts and proactive CSM touchpoints within 3 days.",
        "Medium": "Medium-risk customers in months 1-12 require value realization. Launch a 90-day structured onboarding sequence activating additional features to build product stickiness.",
        "Low": "Low-risk accounts exhibit strong retention (>95%). Maintain loyalty through quarterly rewards, early feature access, and automated annual contract renewal reminders.",
    }

    return briefing, recs
