# features.py
# one function per feature group, a single build() entry point
import numpy as np
import pandas as pd
from config import TARGET_ORGS

RAW_FEATURE_COLS = [
    "id_planning",
    "organization_id_planning",
    "production_lot_id",
    "sequence_planning",
    "process_id",
    "process_group_or_class_id",
    "is_process_group_or_class",
    "is_static_process",
    "is_worker_static",
    "is_work_break_time",
    "is_overtime_expected",
    "is_nightshift_expected",
    "processing_secs",
    "processing_unit",
    "before_processing_secs",
    "before_processing_unit",
    "after_processing_secs",
    "after_processing_unit",
    "quantity_planning",
    "worker_id",
    "worker_group_id",
    "supplier_id",
    "is_fixed",
    "is_included_holiday",
    "delivery_deadline",
    "deadline_production_lots",
    "created_at_planning",
]


def _to_hours(value, unit: str) -> float:
    if pd.isna(value) or value == 0:
        return 0.0
    unit = str(unit).strip("'").lower()
    return {
        "sec": value / 3600,
        "min": value / 60,
        "hour": value,
        "day": value * 24,
    }.get(unit, 0.0)


def _add_process_features(df: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    df["processing_h"] = df.apply(
        lambda r: _to_hours(r["processing_secs"], r["processing_unit"]), axis=1
    )
    df["before_processing_h"] = df.apply(
        lambda r: _to_hours(r["before_processing_secs"], r["before_processing_unit"]),
        axis=1,
    )
    df["after_processing_h"] = df.apply(
        lambda r: _to_hours(r["after_processing_secs"], r["after_processing_unit"]),
        axis=1,
    )
    df["total_step_duration_h"] = (
        df["processing_h"] + df["before_processing_h"] + df["after_processing_h"]
    )

    lot_max_seq = (
        sub.groupby("production_lot_id")["sequence_planning"]
        .max()
        .rename("max_seq_in_lot")
    )
    df = df.merge(lot_max_seq, on="production_lot_id", how="left")
    df["seq_position"] = df["sequence_planning"]
    df["seq_relative"] = df["sequence_planning"] / df["max_seq_in_lot"]
    df["is_last_step"] = (df["sequence_planning"] == df["max_seq_in_lot"]).astype(int)

    bool_cols = [
        "is_static_process",
        "is_worker_static",
        "is_work_break_time",
        "is_overtime_expected",
        "is_nightshift_expected",
        "is_process_group_or_class",
        "is_fixed",
        "is_included_holiday",
    ]
    for col in bool_cols:
        df[col] = df[col].fillna(0).astype(int)

    df["has_worker_assigned"] = df["worker_id"].notna().astype(int)
    df["worker_group_id"] = df["worker_group_id"].fillna(0).astype(int)
    df["log_quantity"] = np.log1p(df["quantity_planning"])
    return df


def _add_lot_features(df: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    lot_n_steps = sub.groupby("production_lot_id").size().rename("n_steps_in_lot")
    df = df.merge(lot_n_steps, on="production_lot_id", how="left")
    df["days_created_to_deadline"] = (
        df["delivery_deadline"] - df["created_at_planning"]
    ).dt.total_seconds() / 86400
    df["deadline_tightness"] = df["n_steps_in_lot"] / df[
        "days_created_to_deadline"
    ].clip(lower=1)
    return df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    prev_end = pd.to_datetime(df["prev_end"])
    df["prev_end_dow"] = prev_end.dt.dayofweek
    df["prev_end_month"] = prev_end.dt.month
    df["prev_end_hour"] = prev_end.dt.hour
    df["prev_end_is_fri"] = (df["prev_end_dow"] == 4).astype(int)
    df["prev_end_is_weekend"] = (df["prev_end_dow"] >= 5).astype(int)
    return df


def _add_transition_features(df: pd.DataFrame) -> pd.DataFrame:
    df["process_group_id"] = df["process_group_or_class_id"].fillna(-1).astype(int)
    grp = df.groupby("production_lot_id")
    df["prev_group_id"] = (
        grp["process_group_or_class_id"].shift(1).fillna(-1).astype(int)
    )
    df["is_group_transition"] = (
        df["process_group_or_class_id"] != df["prev_group_id"]
    ).astype(int)
    df["prev_worker_id"] = grp["worker_id"].shift(1).fillna(-1).astype(int)
    df["prev_process_id"] = grp["process_id"].shift(1).fillna(-1).astype(int)
    df["is_worker_change"] = (
        df["worker_id"].fillna(-1).astype(int) != df["prev_worker_id"]
    ).astype(int)
    df["prev_processing_h"] = grp["processing_h"].shift(1).fillna(0)
    df["cumulative_processing_h"] = grp["processing_h"].cumsum()
    df["remaining_steps"] = df["n_steps_in_lot"] - df["seq_position"]
    return df


def _add_org_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dummies = pd.get_dummies(
        df["organization_id_planning"].astype(str).radd("org_"), drop_first=True
    )
    return pd.concat([df, dummies], axis=1), dummies.columns.tolist()


def build(
    gaps: pd.DataFrame, sub: pd.DataFrame, df_raw: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Merge gaps with raw features, engineer all feature groups, return (model_df, feature_cols)."""
    df_feat = df_raw[RAW_FEATURE_COLS].copy()
    for col in ["delivery_deadline", "deadline_production_lots", "created_at_planning"]:
        df_feat[col] = pd.to_datetime(df_feat[col], errors="coerce")

    merge_keys = [
        "id_planning",
        "organization_id_planning",
        "production_lot_id",
        "sequence_planning",
    ]
    df = gaps[merge_keys + ["prev_end", "gap_working_hours"]].merge(
        df_feat, on=merge_keys, how="left"
    )
    df = df[df["organization_id_planning"].isin(TARGET_ORGS)].copy()
    df = df.sort_values(["production_lot_id", "sequence_planning"]).reset_index(
        drop=True
    )

    df = _add_process_features(df, sub)
    df = _add_lot_features(df, sub)
    df = _add_temporal_features(df)
    df = _add_transition_features(df)
    df, org_cols = _add_org_dummies(df)

    feature_cols = [
        "processing_h",
        "before_processing_h",
        "after_processing_h",
        "total_step_duration_h",
        "seq_position",
        "seq_relative",
        "is_last_step",
        "is_static_process",
        "is_worker_static",
        "is_work_break_time",
        "is_overtime_expected",
        "is_nightshift_expected",
        "is_process_group_or_class",
        "is_fixed",
        "is_included_holiday",
        "has_worker_assigned",
        "worker_group_id",
        "log_quantity",
        "n_steps_in_lot",
        "days_created_to_deadline",
        "deadline_tightness",
        "prev_end_dow",
        "prev_end_month",
        "prev_end_hour",
        "prev_end_is_fri",
        "prev_end_is_weekend",
        "process_group_id",
        "prev_group_id",
        "is_group_transition",
        "prev_worker_id",
        "prev_process_id",
        "is_worker_change",
        "prev_processing_h",
        "cumulative_processing_h",
        "remaining_steps",
    ] + org_cols

    return df, feature_cols
