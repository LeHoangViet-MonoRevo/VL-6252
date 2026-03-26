"""
Production ML Pipeline
======================
Predicts whether a manufacturing job will complete within its planned duration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from helpers import normalize_datetime_columns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("./data_prod")

PATHS = {
    "production_lots": DATA_DIR / "production/production_lots.csv",
    "planning": DATA_DIR / "production/planning_processes.csv",
    "actual": DATA_DIR / "production/planning_process_results.csv",
    "account": DATA_DIR / "account.csv",
}

TARGET_USERS = [
    "teknia",
    "daiho-gr",
    "kondosk",
    "topura",
    "stertec",
    "goldlink",
    "yagi-kinzoku",
    "kimurass",
]

DATETIME_COLS = {
    "production_lots": [
        "delivery_deadline",
        "deadline",
        "material_deadline",
        "start_at",
        "end_at",
        "original_start_date",
        "start_date",
        "created_at",
        "updated_at",
        "deleted_at",
    ],
    "planning": [
        "deadline",
        "start_at",
        "end_at",
        "created_at",
        "updated_at",
        "deleted_at",
    ],
    "actual": [
        "before_processing_start_at",
        "before_processing_end_at",
        "processing_start_at",
        "processing_end_at",
        "after_processing_start_at",
        "first_start_at",
        "last_end_at",
        "created_at",
        "updated_at",
        "deleted_at",
    ],
}

# Data quality thresholds
MAX_RUNTIME_MIN = 5_000
MAX_DELIVERY_QTY = 200
RATIO_CLIP = (0.001, 1_000)

# Modelling
TIME_COL = "created_at_actual"
SUCCESS_QUANTILE = 0.80
TRAIN_FRAC = 0.70
VALID_FRAC = 0.85  # cumulative

CATEGORICAL_COLS = ["worker_id", "process_id", "process_group_or_class_id"]

FEATURE_COLS = [
    "quantity",
    "planning_duration_min",
    "w_success_rate",
    "w_ratio_median",
    "w_ratio_std",
    "w_count",
    "m_success_rate",
    "m_ratio_median",
    "m_ratio_std",
    "m_count",
    "pk_success_rate",
    "pk_ratio_median",
    "pk_ratio_std",
    "pk_count",
    "wm_success_rate",
    "wm_ratio_median",
    "wm_count",
    *CATEGORICAL_COLS,
]

TARGET_COL = "y_success"


# ---------------------------------------------------------------------------
# 1. Load & filter
# ---------------------------------------------------------------------------


def load_raw_tables() -> dict[str, pd.DataFrame]:
    tables = {name: pd.read_csv(path, low_memory=False) for name, path in PATHS.items()}
    for name, dt_cols in DATETIME_COLS.items():
        tables[name] = normalize_datetime_columns(tables[name], dt_cols)
    return tables


def resolve_org_ids(account: pd.DataFrame, users: list[str]) -> set[int]:
    """Return the set of organization_ids that belong to the target users."""
    org_ids: set[int] = set()
    for user in users:
        rows = account[account["updated_by"].str.contains(user, na=False)]
        if not rows.empty:
            org_ids.add(int(rows["organization_id"].iloc[0]))
    return org_ids


def filter_by_orgs(tables: dict[str, pd.DataFrame], org_ids: set[int]) -> None:
    """Filter production_lots, planning, actual in-place to the target orgs."""
    for name in ("production_lots", "planning", "actual"):
        tables[name] = tables[name][tables[name]["organization_id"].isin(org_ids)]


# ---------------------------------------------------------------------------
# 2. Join
# ---------------------------------------------------------------------------


def build_full_join(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    actual_planning = tables["actual"].merge(
        tables["planning"],
        how="left",
        left_on="planning_process_id",
        right_on="id",
        suffixes=("_actual", "_planning"),
    )
    full = actual_planning.merge(
        tables["production_lots"],
        how="left",
        left_on="production_lot_id",
        right_on="id",
        suffixes=("", "_production_lots"),
    )
    print(f"Full join shape: {full.shape}")
    return full


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["planning_duration_min"] = (
        df["end_at"] - df["start_at"]
    ).dt.total_seconds() / 60
    df["actual_duration_min"] = (
        df["last_end_at"] - df["first_start_at"]
    ).dt.total_seconds() / 60

    qty = pd.to_numeric(df["quantity"], errors="coerce").where(lambda s: s > 0)
    df["planning_duration_per_unit_min"] = df["planning_duration_min"] / qty
    df["actual_duration_per_unit_min"] = df["actual_duration_min"] / qty
    df["duration_gap_per_unit_min"] = (
        df["actual_duration_per_unit_min"] - df["planning_duration_per_unit_min"]
    )
    df["duration_ratio_per_unit"] = (
        df["actual_duration_per_unit_min"] / df["planning_duration_per_unit_min"]
    )

    df["ratio"] = (df["actual_duration_min"] / df["planning_duration_min"]).clip(
        *RATIO_CLIP
    )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    required_numeric = ["actual_duration_min", "planning_duration_min", "ratio"]

    mask = (
        df[TIME_COL].notna()
        & df["worker_id"].notna()
        & df["process_id"].notna()
        & df["process_group_or_class_id"].notna()
        & df["duration_gap_per_unit_min"].notna()
        & df["actual_duration_min"].between(0, MAX_RUNTIME_MIN)
        & df["planning_duration_min"].between(
            1, MAX_RUNTIME_MIN
        )  # > 0 for ratio validity
        & df["quantity"].le(MAX_DELIVERY_QTY)
        & df["ratio"].replace([np.inf, -np.inf], np.nan).notna()
        & df["ratio"].gt(0)
    )
    # Replace infs before the mask check
    for col in required_numeric:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df["process_id"] = df.loc[mask, "process_id"].astype(int)
    cleaned = df[mask].copy()
    cleaned["process_id"] = cleaned["process_id"].astype(int)

    print(
        f"Rows after cleaning: {len(cleaned):,}  (dropped {len(df) - len(cleaned):,})"
    )
    return cleaned


# ---------------------------------------------------------------------------
# 4. Train / valid / test split
# ---------------------------------------------------------------------------


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    n = len(df)
    train = df.iloc[: int(n * TRAIN_FRAC)]
    valid = df.iloc[int(n * TRAIN_FRAC) : int(n * VALID_FRAC)]
    test = df.iloc[int(n * VALID_FRAC) :]
    print(f"Train / Valid / Test: {len(train):,} / {len(valid):,} / {len(test):,}")
    return train, valid, test


def label_split(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign binary success label; threshold derived from train only."""
    threshold = train["ratio"].quantile(SUCCESS_QUANTILE)
    print(f"Success threshold (ratio ≤ {threshold:.4f})")
    for df_ in (train, valid, test):
        df_[TARGET_COL] = (df_["ratio"] <= threshold).astype(int)
    return train, valid, test


# ---------------------------------------------------------------------------
# 5. Historical feature aggregation (leak-free: train stats only)
# ---------------------------------------------------------------------------


def _agg(df: pd.DataFrame, group_cols: list[str], prefix: str) -> pd.DataFrame:
    return (
        df.groupby(group_cols)
        .agg(
            **{f"{prefix}_count": ("ratio", "count")},
            **{f"{prefix}_ratio_median": ("ratio", "median")},
            **{f"{prefix}_ratio_mean": ("ratio", "mean")},
            **{f"{prefix}_ratio_std": ("ratio", "std")},
            **{f"{prefix}_success_rate": (TARGET_COL, "mean")},
        )
        .reset_index()
    )


def build_historical_features(train: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "worker": _agg(train, ["worker_id"], "w"),
        "machine": _agg(train, ["process_id"], "m"),
        "pk": _agg(train, ["process_group_or_class_id"], "pk"),
        "wm": _agg(train, ["worker_id", "process_id"], "wm"),
    }


def attach_historical_features(
    df: pd.DataFrame,
    hist: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    df = (
        df.merge(hist["worker"], on="worker_id", how="left")
        .merge(hist["machine"], on="process_id", how="left")
        .merge(hist["pk"], on="process_group_or_class_id", how="left")
        .merge(hist["wm"], on=["worker_id", "process_id"], how="left")
    )

    # Fill missings with neutral values
    for col in df.columns:
        if "success_rate" in col:
            df[col] = df[col].fillna(0.5)
        elif "ratio" in col and col != "ratio":
            df[col] = df[col].fillna(1.0)
        elif col.endswith("_count"):
            df[col] = df[col].fillna(0)

    return df


# ---------------------------------------------------------------------------
# 6. Encode & align
# ---------------------------------------------------------------------------


def encode_and_align(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for df_ in (X_train, X_valid, X_test):
        for col in CATEGORICAL_COLS:
            if col in df_.columns:
                df_[col] = df_[col].astype(str)

    dummy_kw = dict(
        columns=[c for c in CATEGORICAL_COLS if c in X_train.columns], dummy_na=False
    )
    X_train_enc = pd.get_dummies(X_train, **dummy_kw)
    X_valid_enc = pd.get_dummies(X_valid, **dummy_kw).reindex(
        columns=X_train_enc.columns, fill_value=0
    )
    X_test_enc = pd.get_dummies(X_test, **dummy_kw).reindex(
        columns=X_train_enc.columns, fill_value=0
    )

    return X_train_enc, X_valid_enc, X_test_enc


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------


def main() -> dict:
    # Load
    tables = load_raw_tables()
    org_ids = resolve_org_ids(tables["account"], TARGET_USERS)
    filter_by_orgs(tables, org_ids)

    # Build & clean
    df = build_full_join(tables)
    df = engineer_features(df)
    df = clean_data(df)

    # Split & label
    train, valid, test = temporal_split(df)
    train, valid, test = label_split(train, valid, test)

    # Historical features (fit on train, apply to all)
    hist = build_historical_features(train)
    train = attach_historical_features(train, hist)
    valid = attach_historical_features(valid, hist)
    test = attach_historical_features(test, hist)

    # Prepare X / y
    X_train = train[FEATURE_COLS].copy()
    X_valid = valid[FEATURE_COLS].copy()
    X_test = test[FEATURE_COLS].copy()

    y_train, y_valid, y_test = (
        train[TARGET_COL].astype(int),
        valid[TARGET_COL].astype(int),
        test[TARGET_COL].astype(int),
    )

    X_train_enc, X_valid_enc, X_test_enc = encode_and_align(X_train, X_valid, X_test)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)

    print(f"\nFeatures : {X_train_enc.shape[1]}")
    print(f"neg={neg:,}  pos={pos:,}  scale_pos_weight={scale_pos_weight:.4f}")
    print(f"y_train distribution:\n{y_train.value_counts(normalize=True)}")

    return dict(
        X_train=X_train_enc,
        X_valid=X_valid_enc,
        X_test=X_test_enc,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        scale_pos_weight=scale_pos_weight,
    )


if __name__ == "__main__":
    artifacts = main()
