from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from helpers import _parse_mixed_datetime, normalize_datetime_columns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def load_data(paths):
    return {
        "production_lots": pd.read_csv(paths["production_lots"], low_memory=False),
        "planning": pd.read_csv(paths["planning"], low_memory=False),
        "actual": pd.read_csv(paths["actual"], low_memory=False),
        "account": pd.read_csv(paths["account"], low_memory=False),
    }


def get_target_orgs(account, target_users):
    mapping = {}
    for user in target_users:
        filtered = account[account["updated_by"].str.contains(user, na=False)]
        if not filtered.empty:
            mapping[user] = int(filtered["organization_id"].iloc[0])
    return mapping


def filter_by_org(df_dict, org_ids):
    return {
        k: v[v["organization_id"].isin(org_ids)].copy()
        for k, v in df_dict.items()
        if k != "account"
    }


def normalize_all_datetimes(df_dict):
    """
    Normalise datetime
    """

    config = {
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

    for name, cols in config.items():
        df_dict[name] = normalize_datetime_columns(df_dict[name], cols)

    return df_dict


def build_full_join(production_lots, planning, actual):
    actual_with_planning = actual.merge(
        planning,
        how="left",
        left_on="planning_process_id",
        right_on="id",
        suffixes=("_actual", "_planning"),
    )

    full_join = actual_with_planning.merge(
        production_lots,
        how="left",
        left_on="production_lot_id",
        right_on="id",
        suffixes=("", "_production_lots"),
    )

    return full_join


def clean_dataset(df):
    df = df.copy()

    df["machiningTime_actual"] = (
        df["last_end_at"] - df["first_start_at"]
    ).dt.total_seconds() / 60

    df["planning_duration_min"] = (
        df["end_at"] - df["start_at"]
    ).dt.total_seconds() / 60

    df["actual_duration_min"] = (
        df["last_end_at"] - df["first_start_at"]
    ).dt.total_seconds() / 60

    df["ratio"] = df["actual_duration_min"] / df["planning_duration_min"]

    df = df[
        df["machiningTime_actual"].notna()
        & (df["machiningTime_actual"] > 0)
        & df["worker_id"].notna()
        & df["process_id"].notna()
        & df["process_group_or_class_id"].notna()
        & df["planning_duration_min"].gt(0)
        & df["actual_duration_min"].ge(0)
    ].copy()

    df["ratio"] = df["ratio"].replace([np.inf, -np.inf], np.nan)
    df = df[df["ratio"].notna() & (df["ratio"] > 0)]

    return df


def add_features(df):
    """Feature Engineering"""
    df = df.copy()

    df["qty_safe"] = pd.to_numeric(df["quantity"], errors="coerce")
    df.loc[df["qty_safe"] <= 0, "qty_safe"] = np.nan

    df["planning_per_unit"] = df["planning_duration_min"] / df["qty_safe"]
    df["actual_per_unit"] = df["actual_duration_min"] / df["qty_safe"]

    df["gap_per_unit"] = df["actual_per_unit"] - df["planning_per_unit"]
    df["ratio_per_unit"] = df["actual_per_unit"] / df["planning_per_unit"]

    return df


def time_split(df, time_col, train=0.7, valid=0.15):
    """Split data"""
    df = df.sort_values(time_col).reset_index(drop=True)

    n = len(df)
    t1 = int(n * train)
    t2 = int(n * (train + valid))

    return df[:t1], df[t1:t2], df[t2:]


def add_labels(train, valid, test, quantile=0.8):
    threshold = train["ratio"].quantile(quantile)

    for df in [train, valid, test]:
        df["y_success"] = (df["ratio"] <= threshold).astype(int)

    return train, valid, test, threshold


def build_group_features(train_df):
    def agg(df, keys, prefix):
        return (
            df.groupby(keys)
            .agg(
                count=("ratio", "count"),
                ratio_mean=("ratio", "mean"),
                ratio_std=("ratio", "std"),
                success_rate=("y_success", "mean"),
            )
            .rename(columns=lambda x: f"{prefix}_{x}")
            .reset_index()
        )

    return {
        "worker": agg(train_df, ["worker_id"], "w"),
        "machine": agg(train_df, ["process_id"], "m"),
        "pk": agg(train_df, ["process_group_or_class_id"], "pk"),
        "wm": agg(train_df, ["worker_id", "process_id"], "wm"),
    }


def merge_group_features(df, feats):
    df = df.copy()

    df = df.merge(feats["worker"], on="worker_id", how="left")
    df = df.merge(feats["machine"], on="process_id", how="left")
    df = df.merge(feats["pk"], on="process_group_or_class_id", how="left")
    df = df.merge(feats["wm"], on=["worker_id", "process_id"], how="left")

    return df


def encode_data(X_train, X_valid, X_test, categorical_cols):
    for df in [X_train, X_valid, X_test]:
        for c in categorical_cols:
            df[c] = df[c].astype(str)

    X_train_enc = pd.get_dummies(X_train)
    X_valid_enc = pd.get_dummies(X_valid).reindex(
        columns=X_train_enc.columns, fill_value=0
    )
    X_test_enc = pd.get_dummies(X_test).reindex(
        columns=X_train_enc.columns, fill_value=0
    )

    return X_train_enc, X_valid_enc, X_test_enc


if __name__ == "__main__":
    paths = {
        "production_lots": "data_prod/production/production_lots.csv",
        "planning": "data_prod/production/planning_processes.csv",
        "actual": "data_prod/production/planning_process_results.csv",
        "account": "data_prod/account.csv",
    }
    target_user = [
        "teknia",
        "daiho-gr",
        "kondosk",
        "topura",
        "stertec",
        "goldlink",
        "yagi-kinzoku",
        "kimurass",
    ]
    data = load_data(paths)

    target_map = get_target_orgs(data["account"], target_user)
    data = filter_by_org(data, target_map.values())

    data = normalize_all_datetimes(data)

    df = build_full_join(data["production_lots"], data["planning"], data["actual"])

    df = clean_dataset(df)
    df = add_features(df)

    train_df, valid_df, test_df = time_split(df, "created_at_actual")

    train_df, valid_df, test_df, threshold = add_labels(train_df, valid_df, test_df)

    feats = build_group_features(train_df)

    train_df = merge_group_features(train_df, feats)
    valid_df = merge_group_features(valid_df, feats)
    test_df = merge_group_features(test_df, feats)
