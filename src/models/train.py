"""Model training, hyperparameter tuning, threshold calibration, and serialization."""

import json
import pathlib
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from ..data.loader import PROJECT_ROOT
from ..features.engineering import FeatureEngineer

MODELS_DIR = PROJECT_ROOT / "models"


def tune_threshold(
    probs: np.ndarray, y_true: np.ndarray, min_recall: float = 0.60
) -> tuple[float, float]:
    """Find decision threshold maximizing F1 while maintaining minimum recall."""
    thresholds = np.arange(0.1, 0.8, 0.01)
    best_thresh = 0.5
    best_f1 = 0.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        if rec >= min_recall and f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return round(float(best_thresh), 2), round(float(best_f1 * 100), 2)


def evaluate_model(
    probs: np.ndarray, y_true: np.ndarray, threshold: float, name: str
) -> dict[str, Any]:
    """Calculate comprehensive classification metrics at a given threshold."""
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    acc = round(float((tp + tn) / (tp + tn + fp + fn) * 100), 2)
    prec = round(float(tp / (tp + fp) * 100), 2) if (tp + fp) > 0 else 0.0
    rec = round(float(tp / (tp + fn) * 100), 2) if (tp + fn) > 0 else 0.0
    f1 = round(float(2 * prec * rec / (prec + rec)), 2) if (prec + rec) > 0 else 0.0
    auc = round(float(roc_auc_score(y_true, probs) * 100), 2)

    return {
        "name": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc_roc": auc,
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "threshold": threshold,
    }


def train_all_models(
    clean_df: pd.DataFrame,
    save_artifacts: bool = True,
    models_output_dir: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Train Logistic Regression, Random Forest, Gradient Boosting, and Scaler.

    Args:
        clean_df: Clean customer DataFrame.
        save_artifacts: Whether to serialize models to disk.
        models_output_dir: Directory path to save model files.

    Returns:
        Dictionary containing trained models, scaler, thresholds, and performance metrics.
    """
    out_dir = models_output_dir or MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature Engineering
    fe = FeatureEngineer()
    X, _customer_ids, y = fe.fit_transform(clean_df, include_target=True)

    # 2. Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Fit Scaler (for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Model 1: Logistic Regression with Cross-Validation for C
    c_values = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for c in c_values:
        m = LogisticRegression(
            C=c, class_weight="balanced", max_iter=1000, random_state=42
        )
        s = cross_val_score(m, X_train_scaled, y_train, cv=skf, scoring="roc_auc")
        cv_scores.append(s.mean())

    best_c = c_values[int(np.argmax(cv_scores))]
    lr_model = LogisticRegression(
        C=best_c, class_weight="balanced", max_iter=1000, random_state=42
    )
    lr_model.fit(X_train_scaled, y_train)
    lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_thresh, _ = tune_threshold(lr_prob, y_test.values)
    lr_eval = evaluate_model(lr_prob, y_test.values, lr_thresh, "Logistic Regression")

    # 5. Train Model 2: Random Forest (Regularized with max_depth to avoid 34MB bloat)
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    rf_model.fit(X_train, y_train)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    rf_thresh, _ = tune_threshold(rf_prob, y_test.values)
    rf_eval = evaluate_model(rf_prob, y_test.values, rf_thresh, "Random Forest")

    # 6. Train Model 3: Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=3,
        subsample=0.85,
        random_state=42,
    )
    gb_model.fit(X_train, y_train)
    gb_prob = gb_model.predict_proba(X_test)[:, 1]
    gb_thresh, _ = tune_threshold(gb_prob, y_test.values, min_recall=0.65)
    gb_eval = evaluate_model(gb_prob, y_test.values, gb_thresh, "Gradient Boosting")

    thresholds_config = {
        "logistic_regression": lr_thresh,
        "random_forest": rf_thresh,
        "gradient_boosting": gb_thresh,
    }

    # 7. Save models and thresholds with compression
    if save_artifacts:
        joblib.dump(lr_model, out_dir / "logistic_regression.pkl", compress=3)
        joblib.dump(rf_model, out_dir / "random_forest.pkl", compress=3)
        joblib.dump(gb_model, out_dir / "gradient_boosting.pkl", compress=3)
        joblib.dump(scaler, out_dir / "scaler.pkl", compress=3)
        with open(out_dir / "thresholds.json", "w", encoding="utf-8") as f:
            json.dump(thresholds_config, f, indent=2)

    return {
        "models": {
            "logistic_regression": lr_model,
            "random_forest": rf_model,
            "gradient_boosting": gb_model,
        },
        "scaler": scaler,
        "feature_engineer": fe,
        "thresholds": thresholds_config,
        "evaluations": {
            "logistic_regression": lr_eval,
            "random_forest": rf_eval,
            "gradient_boosting": gb_eval,
        },
        "test_data": {
            "X_test": X_test,
            "y_test": y_test,
        },
    }
