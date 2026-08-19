"""Data cleaning, validation, and schema standardization pipeline."""


import pandas as pd

REQUIRED_MINIMAL_COLUMNS: set[str] = {
    "customerID",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "PaymentMethod",
    "Churn",
}

FULL_FEATURE_COLUMNS: set[str] = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}

CHURN_MAPPING = {
    "yes": 1,
    "no": 0,
    "true": 1,
    "false": 0,
    "1": 1,
    "0": 0,
    "1.0": 1,
    "0.0": 0,
}


def validate_schema(
    df: pd.DataFrame, required_columns: set[str] | None = None
) -> list[str]:
    """Validate that the input DataFrame contains all required columns."""
    target_cols = required_columns or REQUIRED_MINIMAL_COLUMNS
    missing = sorted(target_cols - set(df.columns))
    errors = []
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
    return errors


def clean_customer_data(
    raw_df: pd.DataFrame,
    min_rows: int = 1,
    require_full_schema: bool = False,
) -> tuple[pd.DataFrame | None, list[str], int]:
    """Validate, clean, and standardize customer churn data.

    Args:
        raw_df: Raw input pandas DataFrame.
        min_rows: Minimum valid rows required (default 1).
        require_full_schema: If True, validates all 21 standard Telco columns.

    Returns:
        Tuple containing:
        - Cleaned pandas DataFrame (or None if invalid)
        - List of validation/cleaning error messages
        - Count of dropped invalid rows
    """
    if raw_df is None or raw_df.empty:
        return None, ["Input dataset is empty or None."], 0

    target_schema = FULL_FEATURE_COLUMNS if require_full_schema else REQUIRED_MINIMAL_COLUMNS
    errors = validate_schema(raw_df, target_schema)
    if errors:
        return None, errors, 0

    df_clean = raw_df.copy()

    # 1. Standardize string columns (strip whitespace)
    for col in df_clean.select_dtypes(include=["object", "string"]).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # 2. Coerce Churn to integer (0 or 1)
    churn_raw = df_clean["Churn"].astype(str).str.strip().str.lower()
    df_clean["Churn"] = churn_raw.map(CHURN_MAPPING)

    # Fallback to numeric coercion if mapping produced NaNs
    if df_clean["Churn"].isna().any():
        df_clean["Churn"] = df_clean["Churn"].fillna(
            pd.to_numeric(churn_raw, errors="coerce")
        )

    # 3. Coerce numeric columns
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    if "SeniorCitizen" in df_clean.columns:
        numeric_cols.append("SeniorCitizen")

    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # 4. Filter valid rows:
    # - Churn must be 0 or 1
    # - tenure >= 0
    # - MonthlyCharges >= 0
    # - TotalCharges >= 0 (and not NaN)
    valid_mask = (
        df_clean["Churn"].isin([0, 1])
        & df_clean["tenure"].ge(0)
        & df_clean["MonthlyCharges"].ge(0)
        & df_clean["TotalCharges"].notna()
        & df_clean["TotalCharges"].ge(0)
    )

    # If tenure == 0 with NaN TotalCharges, drop them as unbillable brand new signups
    dropped_count = int((~valid_mask).sum())
    df_clean = df_clean.loc[valid_mask].copy()

    if len(df_clean) < min_rows:
        return (
            None,
            [f"At least {min_rows} valid rows are required; found {len(df_clean)}."],
            dropped_count,
        )

    # 5. Type casting
    df_clean["Churn"] = df_clean["Churn"].astype(int)
    df_clean["tenure"] = df_clean["tenure"].astype(int)
    if "SeniorCitizen" in df_clean.columns:
        df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].fillna(0).astype(int)

    df_clean.reset_index(drop=True, inplace=True)
    return df_clean, [], dropped_count
