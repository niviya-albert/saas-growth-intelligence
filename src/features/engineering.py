"""Leakage-free Feature Engineering Transformer for SaaS Churn Prediction."""

import numpy as np
import pandas as pd

FEATURE_COLUMNS_ORDER: list[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "MonthlyCharges",
    "TotalCharges",
    "engagement_score",
    "Internet_DSL",
    "Internet_Fiber optic",
    "Internet_No",
    "is_auto_payment",
    "Payment_Bank transfer (automatic)",
    "Payment_Credit card (automatic)",
    "Payment_Electronic check",
    "Payment_Mailed check",
    "tenure_group",
    "charges_per_month",
    "is_high_value",
    "is_high_risk_profile",
]

SERVICE_COLUMNS: list[str] = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "MultipleLines",
]


class FeatureEngineer:
    """Scikit-Learn style Feature Engineering transformer.

    Transforms cleaned raw customer records into a standardized 30-feature matrix
    suitable for training and inference without data leakage.
    """

    def __init__(self, p66_threshold: float | None = None):
        self.p66_threshold: float | None = p66_threshold
        self.feature_names_: list[str] = FEATURE_COLUMNS_ORDER

    def fit(self, df: pd.DataFrame, y=None) -> "FeatureEngineer":
        """Fit the transformer by learning dataset statistics (e.g. p66 threshold)."""
        if "MonthlyCharges" in df.columns:
            self.p66_threshold = float(df["MonthlyCharges"].quantile(0.66))
        else:
            self.p66_threshold = 80.35  # Default empirical baseline
        return self

    def transform(
        self, df: pd.DataFrame, include_target: bool = False
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
        """Transform raw clean DataFrame into feature matrix.

        Returns:
            Tuple of:
            - X: DataFrame with the exact 30 ordered numerical features.
            - customer_ids: Series of customer identifiers.
            - y: Series of binary target (if Churn column is present and requested).
        """
        df_feat = df.copy()

        customer_ids = (
            df_feat["customerID"].copy()
            if "customerID" in df_feat.columns
            else pd.Series(range(len(df_feat)), name="customerID")
        )

        y = df_feat["Churn"].astype(int) if ("Churn" in df_feat.columns and include_target) else None

        # 1. Binary encodings
        if "gender" in df_feat.columns:
            df_feat["gender"] = df_feat["gender"].astype(str).map({"Female": 0, "Male": 1}).fillna(0).astype(int)
        else:
            df_feat["gender"] = 0

        for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
            if col in df_feat.columns:
                df_feat[col] = (
                    df_feat[col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}).fillna(0).astype(int)
                )
            else:
                df_feat[col] = 0

        if "SeniorCitizen" in df_feat.columns:
            df_feat["SeniorCitizen"] = pd.to_numeric(df_feat["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
        else:
            df_feat["SeniorCitizen"] = 0

        # 2. Service Columns (Yes=1, other=0)
        for col in SERVICE_COLUMNS:
            if col in df_feat.columns:
                df_feat[col] = (
                    df_feat[col].astype(str).str.strip().str.lower().map({"yes": 1, "1": 1}).fillna(0).astype(int)
                )
            else:
                df_feat[col] = 0

        # 3. Product stickiness engagement score
        df_feat["engagement_score"] = df_feat[SERVICE_COLUMNS].sum(axis=1).astype(int)

        # 4. InternetService one-hot encoding
        internet_col = df_feat.get("InternetService", pd.Series(["No"] * len(df_feat))).astype(str)
        df_feat["Internet_DSL"] = (internet_col == "DSL").astype(int)
        df_feat["Internet_Fiber optic"] = (internet_col == "Fiber optic").astype(int)
        df_feat["Internet_No"] = (internet_col.isin(["No", "None", "No internet service"])).astype(int)

        # 5. Contract ordinal encoding
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        if "Contract" in df_feat.columns:
            df_feat["Contract"] = df_feat["Contract"].map(contract_map).fillna(0).astype(int)
        else:
            df_feat["Contract"] = 0

        # 6. PaymentMethod encodings
        payment_col = df_feat.get("PaymentMethod", pd.Series(["Electronic check"] * len(df_feat))).astype(str)
        df_feat["is_auto_payment"] = payment_col.str.lower().str.contains("automatic").astype(int)
        df_feat["Payment_Bank transfer (automatic)"] = (payment_col == "Bank transfer (automatic)").astype(int)
        df_feat["Payment_Credit card (automatic)"] = (payment_col == "Credit card (automatic)").astype(int)
        df_feat["Payment_Electronic check"] = (payment_col == "Electronic check").astype(int)
        df_feat["Payment_Mailed check"] = (payment_col == "Mailed check").astype(int)

        # 7. Domain engineered features
        tenure_num = pd.to_numeric(df_feat.get("tenure", 0), errors="coerce").fillna(0).astype(int)
        monthly_num = pd.to_numeric(df_feat.get("MonthlyCharges", 0.0), errors="coerce").fillna(0.0).astype(float)
        total_num = pd.to_numeric(df_feat.get("TotalCharges", 0.0), errors="coerce").fillna(monthly_num).astype(float)

        df_feat["tenure"] = tenure_num
        df_feat["MonthlyCharges"] = monthly_num
        df_feat["TotalCharges"] = total_num

        # tenure_group: 0 (0-12m), 1 (13-36m), 2 (>36m)
        df_feat["tenure_group"] = np.where(tenure_num <= 12, 0, np.where(tenure_num <= 36, 1, 2)).astype(int)

        # charges_per_month
        df_feat["charges_per_month"] = np.where(
            tenure_num > 0, (total_num / tenure_num).round(2), monthly_num
        ).astype(float)

        # is_high_value
        threshold = self.p66_threshold if self.p66_threshold is not None else 80.35
        df_feat["is_high_value"] = (monthly_num > threshold).astype(int)

        # is_high_risk_profile (tenure <= 12, month-to-month, manual payment)
        df_feat["is_high_risk_profile"] = (
            (df_feat["tenure_group"] == 0)
            & (df_feat["Contract"] == 0)
            & (df_feat["is_auto_payment"] == 0)
        ).astype(int)

        # Ensure exact column ordering and numeric types
        X = df_feat[FEATURE_COLUMNS_ORDER].copy()
        for col in FEATURE_COLUMNS_ORDER:
            if col in ["MonthlyCharges", "TotalCharges", "charges_per_month"]:
                X[col] = X[col].astype(float)
            else:
                X[col] = X[col].astype(int)

        return X, customer_ids, y

    def fit_transform(
        self, df: pd.DataFrame, include_target: bool = True
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
        """Fit statistics on DataFrame and return transformed features."""
        return self.fit(df).transform(df, include_target=include_target)


def engineer_features(
    df: pd.DataFrame, p66_threshold: float | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """Convenience function to transform a DataFrame into features."""
    fe = FeatureEngineer(p66_threshold=p66_threshold)
    if p66_threshold is None:
        fe.fit(df)
    return fe.transform(df, include_target="Churn" in df.columns)
