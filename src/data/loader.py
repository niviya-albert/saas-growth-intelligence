"""Data loading helpers with path resolution."""

import pathlib

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "cleaned"


def get_project_root() -> pathlib.Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT


def load_raw_data(filepath: pathlib.Path | None = None) -> pd.DataFrame:
    """Load the raw customer churn dataset."""
    target_path = filepath or (RAW_DATA_DIR / "telco_churn_raw.csv")
    if not target_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {target_path}")
    return pd.read_csv(target_path)


def load_clean_data(filepath: pathlib.Path | None = None) -> pd.DataFrame:
    """Load the cleaned customer churn dataset."""
    target_path = filepath or (CLEAN_DATA_DIR / "telco_churn_clean.csv")
    if not target_path.exists():
        raise FileNotFoundError(f"Clean data file not found at: {target_path}")
    return pd.read_csv(target_path)


def load_risk_profiles(filepath: pathlib.Path | None = None) -> pd.DataFrame:
    """Load precomputed customer risk profiles."""
    target_path = filepath or (CLEAN_DATA_DIR / "customer_risk_profiles.csv")
    if not target_path.exists():
        raise FileNotFoundError(f"Customer risk profiles not found at: {target_path}")
    return pd.read_csv(target_path)


def load_all_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both clean data and customer risk profiles."""
    return load_clean_data(), load_risk_profiles()
