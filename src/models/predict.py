"""Inference, ensemble risk scoring, churn driver attribution, and dynamic what-if simulation."""

import json
import pathlib
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..data.loader import PROJECT_ROOT
from ..features.engineering import FeatureEngineer

MODELS_DIR = PROJECT_ROOT / "models"


def assign_risk_level(score: float) -> str:
    """Map numeric risk score (0-100) to standard categorical risk tier."""
    if score >= 70.0:
        return "Critical"
    elif score >= 50.0:
        return "High"
    elif score >= 30.0:
        return "Medium"
    else:
        return "Low"


def extract_churn_drivers(row_dict: dict[str, Any]) -> str:
    """Identify top 3 primary business drivers explaining churn risk for a customer."""
    drivers = []
    rules = [
        ("Contract", 0, "Month-to-month contract"),
        ("is_auto_payment", 0, "Manual payment method"),
        ("tenure_group", 0, "New customer (0-12 months)"),
        ("Internet_Fiber optic", 1, "Fiber optic (premium price risk)"),
        ("is_high_risk_profile", 1, "Combined high-risk profile"),
        ("Payment_Electronic check", 1, "Electronic check payment"),
        ("SeniorCitizen", 1, "Senior customer"),
        ("PaperlessBilling", 1, "Paperless billing"),
    ]

    for feature, risk_val, explanation in rules:
        if feature in row_dict and row_dict[feature] == risk_val:
            drivers.append(explanation)
        if len(drivers) == 3:
            break

    # Continuous feature checks
    if len(drivers) < 3:
        eng = row_dict.get("engagement_score", 5)
        if eng <= 2:
            drivers.append(f"Low engagement ({int(eng)} services)")

    if len(drivers) < 3:
        chg = row_dict.get("MonthlyCharges", 0.0)
        if chg > 80.0:
            drivers.append(f"High charge (${chg:.0f}/mo)")

    if not drivers:
        drivers = ["Standard risk profile"]

    return " | ".join(drivers[:3])


def generate_recommended_action(risk_level: str, row_dict: dict[str, Any]) -> str:
    """Generate prescriptive retention action tailored to the customer's risk profile."""
    contract = row_dict.get("Contract", 1)
    auto_pay = row_dict.get("is_auto_payment", 1)
    eng = row_dict.get("engagement_score", 3)
    monthly = row_dict.get("MonthlyCharges", 64.0)

    if risk_level == "Critical":
        if contract == 0:
            return (
                "Personal call within 24 hours. "
                "Offer 40% discount on annual plan upgrade. "
                "Escalate to senior retention specialist."
            )
        elif monthly > 80.0:
            return (
                "Personal call within 24 hours. "
                "Offer premium feature walkthrough & discount. "
                "Assign dedicated success manager."
            )
        else:
            return (
                "Personal call within 24 hours. "
                "Conduct onboarding check-in. "
                "Offer loyalty incentive."
            )

    elif risk_level == "High":
        if auto_pay == 0:
            return (
                "Send targeted email offering 10% discount for switching to auto-payment. "
                "Personal follow-up call within 3 days if unread."
            )
        elif eng <= 2:
            return (
                "Send product walkthrough sequence. "
                "Offer free 60-day trial of TechSupport or OnlineSecurity. "
                "Schedule check-in call within 5 days."
            )
        else:
            return (
                "Send satisfaction survey with direct CSM escalation. "
                "Offer annual contract renewal incentive."
            )

    elif risk_level == "Medium":
        if eng <= 2:
            return (
                "Trigger 90-day automated onboarding sequence. "
                "Highlight unused platform features. "
                "Send weekly value-focused tips."
            )
        else:
            return (
                "Send quarterly value summary report. "
                "Invite to customer advisory panel. "
                "Standard retention monitoring."
            )

    else:  # Low
        return (
            "Maintain standard lifecycle communications. "
            "Quarterly loyalty rewards & early access to new features. "
            "Renewal reminder 60 days before contract expiry."
        )


class CustomerRiskScorer:
    """Production inference engine for scoring customer churn risk."""

    def __init__(
        self,
        models_dir: pathlib.Path | None = None,
        models: dict[str, Any] | None = None,
        scaler: Any | None = None,
        thresholds: dict[str, float] | None = None,
    ):
        self.models_dir = models_dir or MODELS_DIR
        self.models = models or {}
        self.scaler = scaler
        self.thresholds = thresholds or {}
        self.feature_engineer = FeatureEngineer()

        if not self.models:
            self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load pickled models, scaler, and threshold configs from disk."""
        gb_path = self.models_dir / "gradient_boosting.pkl"
        rf_path = self.models_dir / "random_forest.pkl"
        lr_path = self.models_dir / "logistic_regression.pkl"
        scaler_path = self.models_dir / "scaler.pkl"
        thresh_path = self.models_dir / "thresholds.json"

        if not gb_path.exists():
            raise FileNotFoundError(
                f"Trained models not found at {self.models_dir}. "
                "Run `python -m src.pipeline` or `train_all_models()` first."
            )

        self.models["gradient_boosting"] = joblib.load(gb_path)
        self.models["random_forest"] = joblib.load(rf_path)
        self.models["logistic_regression"] = joblib.load(lr_path)
        self.scaler = joblib.load(scaler_path)

        if thresh_path.exists():
            with open(thresh_path, "r", encoding="utf-8") as f:
                self.thresholds = json.load(f)
        else:
            self.thresholds = {
                "gradient_boosting": 0.29,
                "random_forest": 0.50,
                "logistic_regression": 0.62,
            }

    def predict_probabilities(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute individual and ensemble probabilities for feature matrix X."""
        gb_prob = self.models["gradient_boosting"].predict_proba(X)[:, 1]
        rf_prob = self.models["random_forest"].predict_proba(X)[:, 1]
        X_scaled = self.scaler.transform(X)
        lr_prob = self.models["logistic_regression"].predict_proba(X_scaled)[:, 1]

        ensemble_prob = 0.50 * gb_prob + 0.30 * rf_prob + 0.20 * lr_prob
        return ensemble_prob, gb_prob, rf_prob, lr_prob

    def score_dataframe(
        self, raw_df: pd.DataFrame, include_original_columns: bool = True
    ) -> pd.DataFrame:
        """Score an entire customer DataFrame and generate full risk profiles."""
        X, customer_ids, y = self.feature_engineer.transform(
            raw_df, include_target="Churn" in raw_df.columns
        )

        ensemble_prob, gb_prob, rf_prob, lr_prob = self.predict_probabilities(X)
        risk_score = np.round(ensemble_prob * 100.0, 1)

        result_df = pd.DataFrame(
            {
                "customerID": customer_ids,
                "risk_score": risk_score,
                "ensemble_probability": np.round(ensemble_prob, 4),
                "gb_probability": np.round(gb_prob, 4),
                "rf_probability": np.round(rf_prob, 4),
                "lr_probability": np.round(lr_prob, 4),
            }
        )

        result_df["risk_level"] = result_df["risk_score"].apply(assign_risk_level)

        # Merge key customer fields
        monthly_rev = pd.to_numeric(
            raw_df.get("MonthlyCharges", 65.0), errors="coerce"
        ).fillna(65.0)
        tenure_mo = pd.to_numeric(
            raw_df.get("tenure", 12), errors="coerce"
        ).fillna(12)
        contract_str = raw_df.get("Contract", "Month-to-month").astype(str)
        payment_str = raw_df.get("PaymentMethod", "Electronic check").astype(str)

        result_df["monthly_revenue"] = monthly_rev.values
        result_df["annual_revenue_risk"] = np.round(monthly_rev.values * 12.0, 2)
        result_df["tenure_months"] = tenure_mo.values
        result_df["contract_type"] = contract_str.values
        result_df["payment_method"] = payment_str.values

        if y is not None:
            result_df["actual_churn"] = y.values

        # Sort and assign priority rank (1 = highest risk)
        result_df.sort_values(by="risk_score", ascending=False, inplace=True)
        result_df["priority_rank"] = range(1, len(result_df) + 1)

        # Generate churn drivers and recommended actions
        feature_dicts = X.to_dict(orient="records")
        # Align with result_df indexing
        orig_indices = result_df.index
        drivers_list = []
        actions_list = []

        for idx in orig_indices:
            row_dict = feature_dicts[idx]
            level = result_df.loc[idx, "risk_level"]
            drivers_list.append(extract_churn_drivers(row_dict))
            actions_list.append(generate_recommended_action(level, row_dict))

        result_df["churn_drivers"] = drivers_list
        result_df["recommended_action"] = actions_list

        return result_df.reset_index(drop=True)

    def simulate_what_if(
        self,
        base_customer_dict: dict[str, Any],
        new_contract: str | None = None,
        new_payment: str | None = None,
        new_tenure: int | None = None,
        new_monthly_charges: float | None = None,
    ) -> dict[str, Any]:
        """Perform dynamic model-driven What-If simulation for a customer."""
        # 1. Base score
        base_df = pd.DataFrame([base_customer_dict])
        base_scored = self.score_dataframe(base_df).iloc[0]
        base_score = float(base_scored["risk_score"])
        base_prob = float(base_scored["ensemble_probability"])

        # 2. Modified row
        mod_dict = dict(base_customer_dict)
        reasons = []

        if new_contract and new_contract != mod_dict.get("Contract"):
            old_c = mod_dict.get("Contract")
            mod_dict["Contract"] = new_contract
            reasons.append(f"Contract: {old_c} → {new_contract}")

        if new_payment and new_payment != mod_dict.get("PaymentMethod"):
            old_p = mod_dict.get("PaymentMethod")
            mod_dict["PaymentMethod"] = new_payment
            reasons.append(f"Payment: {old_p} → {new_payment}")

        if new_tenure is not None and new_tenure != mod_dict.get("tenure"):
            old_t = mod_dict.get("tenure")
            mod_dict["tenure"] = new_tenure
            reasons.append(f"Tenure: {old_t}m → {new_tenure}m")

        if new_monthly_charges is not None and new_monthly_charges != mod_dict.get("MonthlyCharges"):
            old_m = mod_dict.get("MonthlyCharges")
            mod_dict["MonthlyCharges"] = new_monthly_charges
            reasons.append(f"Monthly: ${old_m:.2f} → ${new_monthly_charges:.2f}")

        mod_df = pd.DataFrame([mod_dict])
        new_scored = self.score_dataframe(mod_df).iloc[0]
        new_score = float(new_scored["risk_score"])
        new_prob = float(new_scored["ensemble_probability"])

        delta = round(new_score - base_score, 1)

        return {
            "base_score": base_score,
            "new_score": new_score,
            "base_probability": base_prob,
            "new_probability": new_prob,
            "score_delta": delta,
            "new_risk_level": new_scored["risk_level"],
            "new_drivers": new_scored["churn_drivers"],
            "new_action": new_scored["recommended_action"],
            "reasons": reasons,
        }


def score_customer_dataframe(
    raw_df: pd.DataFrame, models_dir: pathlib.Path | None = None
) -> pd.DataFrame:
    """Convenience function to score a DataFrame using the default model artifacts."""
    scorer = CustomerRiskScorer(models_dir=models_dir)
    return scorer.score_dataframe(raw_df)


def simulate_what_if(
    base_customer_dict: dict[str, Any],
    new_contract: str | None = None,
    new_payment: str | None = None,
    scorer: CustomerRiskScorer | None = None,
) -> dict[str, Any]:
    """Convenience function to run a dynamic what-if simulation on a customer."""
    risk_scorer = scorer or CustomerRiskScorer()
    return risk_scorer.simulate_what_if(
        base_customer_dict, new_contract=new_contract, new_payment=new_payment
    )
