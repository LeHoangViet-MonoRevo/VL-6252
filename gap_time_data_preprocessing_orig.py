import numpy as np
import pandas as pd

from helpers import normalize_datetime_columns, subtract_weekend_hours

production_lots_path = "./data_prod/production/production_lots.csv"
planning_path = "./data_prod/production/planning_processes.csv"
production_lots = pd.read_csv(production_lots_path, low_memory=False)
planning = pd.read_csv(planning_path, low_memory=False)

target_user_map = {
    "teknia": 86,
    "daiho-gr": 90,
    "kondosk": 88,
    "topura": 74,
    "stertec": 116,
    "yagi-kinzoku": 72,
    "kimurass": 128,
}

production_lots = production_lots[
    production_lots["organization_id"].isin(target_user_map.values())
]
planning = planning[planning["organization_id"].isin(target_user_map.values())]
print("\nAfter filtering by target users:")
print("production_lots:", production_lots.shape)
print("planning:", planning.shape)


# Production_lots datetime columns (only convert if they exist)
production_lots_dt_cols = [
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
]
production_lots = normalize_datetime_columns(production_lots, production_lots_dt_cols)

# Planning datetime columns
planning_dt_cols = [
    "deadline",
    "start_at",
    "end_at",
    "created_at",
    "updated_at",
    "deleted_at",
]

planning = normalize_datetime_columns(planning, planning_dt_cols)


planning_with_production_lot = planning.merge(
    production_lots,
    how="left",
    left_on="production_lot_id",
    right_on="id",
    suffixes=("_planning", "_production_lots"),
)

print("\nplanning_with_production_lot shape:", planning_with_production_lot.shape)
planning_with_production_lot.to_csv("planning_with_production_lot.csv", index=False)


FILE_PATH = "planning_with_production_lot.csv"  # update path if needed

df = pd.read_csv(FILE_PATH, low_memory=False)
print(f"Rows: {len(df):,}  |  Columns: {df.shape[1]}")
print(f'Organizations: {sorted(df["organization_id_planning"].unique())}')
print(f'Production lots: {df["production_lot_id"].nunique():,}')

COLS = [
    "id_planning",
    "organization_id_planning",
    "production_lot_id",
    "sequence_planning",
    "start_at_planning",
    "end_at_planning",
]

sub = df[COLS].copy()
sub["start_at_planning"] = pd.to_datetime(sub["start_at_planning"], errors="coerce")
sub["end_at_planning"] = pd.to_datetime(sub["end_at_planning"], errors="coerce")

print("=== BEFORE CLEANING ===")
print(f"Rows:  {len(sub):,}")
print(f'Lots:  {sub["production_lot_id"].nunique():,}')
print()

# Filter 1: remove processes where duration = 0 (start == end, unscheduled)
mask_duration = sub["start_at_planning"] != sub["end_at_planning"]
print(f"Filter 1 — duration = 0: removed {(~mask_duration).sum():,} rows")
sub = sub[mask_duration].copy()

# Filter 2: remove lots where all processes share the same start time
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
sub = sub.sort_values(["production_lot_id", "sequence_planning"]).reset_index(drop=True)

# Shift within each lot — consecutive rows after sort = consecutive sequences
# Note: sequence may not be contiguous (e.g. 2, 6) but sort order is still correct
sub["prev_end"] = sub.groupby("production_lot_id")["end_at_planning"].shift(1)
sub["prev_sequence"] = sub.groupby("production_lot_id")["sequence_planning"].shift(1)

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

# Filter 3: remove entire lots that contain ANY overlap gap (gap < 0)
# Dropping individual transitions is not enough — the whole lot's schedule is unreliable
lots_with_overlap = gaps[gaps["gap_hours"] < 0]["production_lot_id"].unique()
print(
    f"Filter 3 — lots with at least 1 overlap: removed {len(lots_with_overlap):,} lots"
)
gaps = gaps[~gaps["production_lot_id"].isin(lots_with_overlap)].copy()
sub = sub[~sub["production_lot_id"].isin(lots_with_overlap)].copy()

print()
print("=== AFTER CLEANING ===")
print(f"Rows:  {len(sub):,}")
print(f'Lots:  {sub["production_lot_id"].nunique():,}')
print(f"Gaps:  {len(gaps):,}")
print(f'  zero   (= 0h):   {(gaps["gap_hours"] == 0).sum():,}')
print(f'  positive (> 0h): {(gaps["gap_hours"] > 0).sum():,}')


# Apply to gaps
gaps["gap_working_hours"] = gaps.apply(
    lambda r: subtract_weekend_hours(r["prev_end"], r["start_at_planning"]), axis=1
)

# Apply to sub — duration of each planning process excluding weekends
sub["duration_working_hours"] = sub.apply(
    lambda r: subtract_weekend_hours(r["start_at_planning"], r["end_at_planning"]),
    axis=1,
)

# Also apply gap_working_hours to sub for rows that have prev_end
sub["gap_working_hours"] = sub.apply(
    lambda r: (
        subtract_weekend_hours(r["prev_end"], r["start_at_planning"])
        if pd.notna(r["prev_end"])
        else np.nan
    ),
    axis=1,
)

# Filter 4: remove entire lots that contain ANY gap > 30 days (working hours)
GAP_THRESHOLD_HOURS = 30 * 24

lots_with_long_gap = gaps[gaps["gap_working_hours"] > GAP_THRESHOLD_HOURS][
    "production_lot_id"
].unique()
print(f"Filter 4 — lots with gap > 30 days: removed {len(lots_with_long_gap):,} lots")

gaps = gaps[~gaps["production_lot_id"].isin(lots_with_long_gap)].copy()
sub = sub[~sub["production_lot_id"].isin(lots_with_long_gap)].copy()

print(f"Gaps after filter 4: {len(gaps):,}")
print(f'Lots after filter 4: {sub["production_lot_id"].nunique():,}')

print("=== gap_hours vs gap_working_hours ===")
print(
    f'Mean  — calendar: {gaps["gap_hours"].mean():.1f}h  |  working: {gaps["gap_working_hours"].mean():.1f}h'
)
print(
    f'Median— calendar: {gaps["gap_hours"].median():.1f}h  |  working: {gaps["gap_working_hours"].median():.1f}h'
)
print(
    f'Max   — calendar: {gaps["gap_hours"].max():.1f}h  |  working: {gaps["gap_working_hours"].max():.1f}h'
)
print()

# Gaps affected by weekend removal (diff >= 24h = at least 1 weekend day removed)
diff = gaps[gaps["gap_hours"] - gaps["gap_working_hours"] >= 24]
print(f"Gaps affected by weekend (diff >= 24h): {len(diff):,}")

TARGET_ORGS = [72, 74, 86, 88, 116, 128]

# ── Helper functions ──────────────────────────────────────────────────────────


def to_hours(value, unit):
    """Convert value + unit string to hours."""
    if pd.isna(value) or value == 0:
        return 0.0
    unit = str(unit).strip("'").lower()
    if unit == "sec":
        return value / 3600
    elif unit == "min":
        return value / 60
    elif unit == "hour":
        return value
    elif unit == "day":
        return value * 24
    else:
        return 0.0


# ── Select relevant columns from raw df ──────────────────────────────────────

df_feat = df[
    [
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
].copy()

df_feat["delivery_deadline"] = pd.to_datetime(
    df_feat["delivery_deadline"], errors="coerce"
)
df_feat["deadline_production_lots"] = pd.to_datetime(
    df_feat["deadline_production_lots"], errors="coerce"
)
df_feat["created_at_planning"] = pd.to_datetime(
    df_feat["created_at_planning"], errors="coerce"
)

# ── Merge gap target into features ───────────────────────────────────────────

model_df = gaps[
    [
        "id_planning",
        "organization_id_planning",
        "production_lot_id",
        "sequence_planning",
        "prev_end",
        "gap_working_hours",
    ]
].merge(
    df_feat,
    on=[
        "id_planning",
        "organization_id_planning",
        "production_lot_id",
        "sequence_planning",
    ],
    how="left",
)

# Filter to target orgs only
model_df = model_df[model_df["organization_id_planning"].isin(TARGET_ORGS)].copy()
model_df = model_df.sort_values(["production_lot_id", "sequence_planning"]).reset_index(
    drop=True
)

print(f"Samples before feature engineering: {len(model_df):,}")


# ── Feature Group 1: Process-level (current step properties) ─────────────────

# Processing time components — convert using unit column
model_df["processing_h"] = model_df.apply(
    lambda r: to_hours(r["processing_secs"], r["processing_unit"]), axis=1
)
model_df["before_processing_h"] = model_df.apply(
    lambda r: to_hours(r["before_processing_secs"], r["before_processing_unit"]), axis=1
)
model_df["after_processing_h"] = model_df.apply(
    lambda r: to_hours(r["after_processing_secs"], r["after_processing_unit"]), axis=1
)
model_df["total_step_duration_h"] = (
    model_df["processing_h"]
    + model_df["before_processing_h"]
    + model_df["after_processing_h"]
)

# Sequence position within lot (use sub — all steps, not just gaps)
lot_max_seq = (
    sub.groupby("production_lot_id")["sequence_planning"].max().rename("max_seq_in_lot")
)
model_df = model_df.merge(lot_max_seq, on="production_lot_id", how="left")
model_df["seq_position"] = model_df["sequence_planning"]
model_df["seq_relative"] = model_df["sequence_planning"] / model_df["max_seq_in_lot"]
model_df["is_last_step"] = (
    model_df["sequence_planning"] == model_df["max_seq_in_lot"]
).astype(int)

# Binary flags
model_df["is_static_process"] = model_df["is_static_process"].fillna(0).astype(int)
model_df["is_worker_static"] = model_df["is_worker_static"].fillna(0).astype(int)
model_df["is_work_break_time"] = model_df["is_work_break_time"].fillna(0).astype(int)
model_df["is_overtime_expected"] = (
    model_df["is_overtime_expected"].fillna(0).astype(int)
)
model_df["is_nightshift_expected"] = (
    model_df["is_nightshift_expected"].fillna(0).astype(int)
)
model_df["is_process_group_or_class"] = (
    model_df["is_process_group_or_class"].fillna(0).astype(int)
)
model_df["is_fixed"] = model_df["is_fixed"].fillna(0).astype(int)
model_df["is_included_holiday"] = model_df["is_included_holiday"].fillna(0).astype(int)

# Worker & quantity
model_df["has_worker_assigned"] = model_df["worker_id"].notna().astype(int)
model_df["worker_group_id"] = model_df["worker_group_id"].fillna(0).astype(int)
model_df["log_quantity"] = np.log1p(model_df["quantity_planning"])


# ── Feature Group 2: Lot-level ───────────────────────────────────────────────

# Total steps in lot — use sub (all steps) not gaps
lot_n_steps = sub.groupby("production_lot_id").size().rename("n_steps_in_lot")
model_df = model_df.merge(lot_n_steps, on="production_lot_id", how="left")

# Days from lot creation to deadline
model_df["days_created_to_deadline"] = (
    model_df["delivery_deadline"] - model_df["created_at_planning"]
).dt.total_seconds() / 86400

# Deadline tightness: steps per available day
model_df["deadline_tightness"] = model_df["n_steps_in_lot"] / model_df[
    "days_created_to_deadline"
].clip(lower=1)


# ── Feature Group 3: Temporal (from prev_end) ────────────────────────────────
# prev_end is available at inference time (end of previous adjusted step)

model_df["prev_end_dow"] = pd.to_datetime(model_df["prev_end"]).dt.dayofweek  # 0=Mon
model_df["prev_end_month"] = pd.to_datetime(model_df["prev_end"]).dt.month
model_df["prev_end_hour"] = pd.to_datetime(model_df["prev_end"]).dt.hour
model_df["prev_end_is_fri"] = (model_df["prev_end_dow"] == 4).astype(int)
model_df["prev_end_is_weekend"] = (model_df["prev_end_dow"] >= 5).astype(int)


# ── Feature Group 4: Previous step & cross-step transitions ──────────────────

# Process group
model_df["process_group_id"] = (
    model_df["process_group_or_class_id"].fillna(-1).astype(int)
)

model_df["prev_group_id"] = (
    model_df.groupby("production_lot_id")["process_group_or_class_id"]
    .shift(1)
    .fillna(-1)
    .astype(int)
)
model_df["is_group_transition"] = (
    model_df["process_group_or_class_id"] != model_df["prev_group_id"]
).astype(int)

# Previous step identity
model_df["prev_worker_id"] = (
    model_df.groupby("production_lot_id")["worker_id"].shift(1).fillna(-1).astype(int)
)
model_df["prev_process_id"] = (
    model_df.groupby("production_lot_id")["process_id"].shift(1).fillna(-1).astype(int)
)

# Worker handoff flag
model_df["is_worker_change"] = (
    model_df["worker_id"].fillna(-1).astype(int) != model_df["prev_worker_id"]
).astype(int)

# Previous step processing time
model_df["prev_processing_h"] = (
    model_df.groupby("production_lot_id")["processing_h"].shift(1).fillna(0)
)

# Lot progress
model_df["cumulative_processing_h"] = model_df.groupby("production_lot_id")[
    "processing_h"
].cumsum()
model_df["remaining_steps"] = model_df["n_steps_in_lot"] - model_df["seq_position"]

# ── Feature Group 5: Organization encoding ───────────────────────────────────

org_dummies = pd.get_dummies(
    model_df["organization_id_planning"].astype(str).radd("org_"), drop_first=True
)
model_df = pd.concat([model_df, org_dummies], axis=1)
org_dummy_cols = org_dummies.columns.tolist()


# ── Final feature list ────────────────────────────────────────────────────────

FEATURES = [
    # ── Group 1: Process-level (current step) ──
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
    # ── Group 2: Lot-level ──
    "n_steps_in_lot",
    "days_created_to_deadline",
    "deadline_tightness",
    # ── Group 3: Temporal (prev_end) ──
    "prev_end_dow",
    "prev_end_month",
    "prev_end_hour",
    "prev_end_is_fri",
    "prev_end_is_weekend",
    # ── Group 4: Previous step & transitions ──
    "process_group_id",
    "prev_group_id",
    "is_group_transition",
    "prev_worker_id",
    "prev_process_id",
    "is_worker_change",
    "prev_processing_h",
    "cumulative_processing_h",
    "remaining_steps",
    # ── Group 5: Organization ──
] + org_dummy_cols

TARGET = "gap_working_hours"

# ── Sanity check ─────────────────────────────────────────────────────────────

print(f"Total samples in model_df: {len(model_df):,}")
print(f"Total features: {len(FEATURES)}")
print()
print("Null counts per feature:")
null_counts = model_df[FEATURES + [TARGET]].isna().sum()
print(null_counts[null_counts > 0].to_string())
