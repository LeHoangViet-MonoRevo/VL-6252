import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


def _parse_mixed_datetime(x):
    if pd.isna(x):
        return pd.NaT

    s = str(x).strip().strip("'").strip('"')

    if s in {"", "None", "nan", "NaN", "NaT", "NULL", "null"}:
        return pd.NaT

    try:
        # ISO kiểu 2025-05-22T02:10:05Z
        if "T" in s:
            s2 = s.replace("Z", "+00:00")
            dt = pd.Timestamp(s2)
            if dt.tz is not None:
                dt = dt.tz_convert("UTC").tz_localize(None)
            return dt

        # SQL kiểu 2025-04-22 23:00:00
        return pd.Timestamp(s)

    except Exception:
        return pd.NaT


def normalize_datetime_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(_parse_mixed_datetime)
    return df


def build_model(model_type):
    if model_type == "xgboost":
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            # scale_pos_weight=scale_pos_weight,    # Temporarily disabled
        )

    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    elif model_type == "lightgbm":
        if lgb is None:
            return None
        return lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_metrics(y_true, y_proba, th=0.5):
    y_pred = (y_proba >= th).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def find_best_threshold(y_true, y_proba, thresholds):
    best_th = 0.5
    best_f1 = -1
    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th
