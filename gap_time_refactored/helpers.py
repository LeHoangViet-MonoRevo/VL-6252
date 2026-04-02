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


def subtract_weekend_hours(start, end):
    """Return gap in hours, excluding Saturday and Sunday hours."""
    if pd.isna(start) or pd.isna(end) or end <= start:
        return 0.0

    total_hours = (end - start).total_seconds() / 3600

    # Count weekend hours in the interval
    weekend_hours = 0.0
    cursor = start

    while cursor < end:
        next_day = (cursor + pd.Timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        next_day = min(next_day, end)
        if cursor.weekday() >= 5:  # 5=Saturday, 6=Sunday
            weekend_hours += (next_day - cursor).total_seconds() / 3600
        cursor = next_day

    return total_hours - weekend_hours


def filter1(sub: pd.DataFrame):
    mask_duration = sub["start_at_planning"] != sub["end_at_planning"]
    print(f"Filter 1 — duration = 0: removed {(~mask_duration).sum():,} rows")
    sub = sub[mask_duration].copy()
    return sub


def filter2(sub: pd.DataFrame):
    lot_unique_starts = sub.groupby("production_lot_id")["start_at_planning"].nunique()
    valid_lots = lot_unique_starts[lot_unique_starts > 1].index
    removed_lots = sub["production_lot_id"].nunique() - len(valid_lots)
    removed_rows = (~sub["production_lot_id"].isin(valid_lots)).sum()
    print(
        f"Filter 2 — all same start: removed {removed_lots:,} lots / {removed_rows:,} rows"
    )
    sub = sub[sub["production_lot_id"].isin(valid_lots)].copy()

    print()
    print("=== AFTER FILTER 1 & 2 ===")
    print(f"Rows:  {len(sub):,}")
    print(f'Lots:  {sub["production_lot_id"].nunique():,}')
    print()

    # Sort by lot then by sequence (sequence defines the true order within a lot)
    sub = sub.sort_values(["production_lot_id", "sequence_planning"]).reset_index(
        drop=True
    )

    # Shift within each lot — consecutive rows after sort = consecutive sequences
    # Note: sequence may not be contiguous (e.g. 2, 6) but sort order is still correct
    sub["prev_end"] = sub.groupby("production_lot_id")["end_at_planning"].shift(1)
    sub["prev_sequence"] = sub.groupby("production_lot_id")["sequence_planning"].shift(
        1
    )

    # Gap = start of current step - end of previous step (in hours)
    sub["gap_hours"] = (
        sub["start_at_planning"] - sub["prev_end"]
    ).dt.total_seconds() / 3600

    # Drop the first step of each lot (no previous step to compare)
    gaps = sub.dropna(subset=["gap_hours", "prev_end"]).copy()

    print(f"Total gaps (before overlap removal): {len(gaps):,}")
    print(f'  overlap (< 0h):  {(gaps["gap_hours"] < 0).sum():,}')
    print(f'  zero   (= 0h):   {(gaps["gap_hours"] == 0).sum():,}')
    print(f'  positive (> 0h): {(gaps["gap_hours"] > 0).sum():,}')
    print()
    return sub, gaps


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
