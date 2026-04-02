# cleaner.py
# All four filters plus gap computation
import numpy as np
import pandas as pd
from config import GAP_THRESHOLD_HOURS

from helpers import subtract_weekend_hours

STEP_COLS = [
    "id_planning",
    "organization_id_planning",
    "production_lot_id",
    "sequence_planning",
    "start_at_planning",
    "end_at_planning",
]


def _filter_zero_duration(sub: pd.DataFrame) -> pd.DataFrame:
    mask = sub["start_at_planning"] != sub["end_at_planning"]
    print(f"Filter 1 — zero duration: removed {(~mask).sum():,} rows")
    return sub[mask].copy()


def _filter_same_start(sub: pd.DataFrame) -> pd.DataFrame:
    lot_unique_starts = sub.groupby("production_lot_id")["start_at_planning"].nunique()
    valid = lot_unique_starts[lot_unique_starts > 1].index
    removed = sub["production_lot_id"].nunique() - len(valid)
    print(f"Filter 2 — all same start: removed {removed:,} lots")
    return sub[sub["production_lot_id"].isin(valid)].copy()


def _compute_gaps(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.sort_values(["production_lot_id", "sequence_planning"]).reset_index(
        drop=True
    )
    grp = sub.groupby("production_lot_id")
    sub["prev_end"] = grp["end_at_planning"].shift(1)
    sub["prev_sequence"] = grp["sequence_planning"].shift(1)
    sub["gap_hours"] = (
        sub["start_at_planning"] - sub["prev_end"]
    ).dt.total_seconds() / 3600
    return sub


def _filter_overlapping_lots(
    sub: pd.DataFrame, gaps: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bad = gaps[gaps["gap_hours"] < 0]["production_lot_id"].unique()
    print(f"Filter 3 — overlapping lots: removed {len(bad):,} lots")
    return (
        sub[~sub["production_lot_id"].isin(bad)].copy(),
        gaps[~gaps["production_lot_id"].isin(bad)].copy(),
    )


def _apply_working_hours(
    sub: pd.DataFrame, gaps: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gaps["gap_working_hours"] = gaps.apply(
        lambda r: subtract_weekend_hours(r["prev_end"], r["start_at_planning"]), axis=1
    )
    sub["duration_working_hours"] = sub.apply(
        lambda r: subtract_weekend_hours(r["start_at_planning"], r["end_at_planning"]),
        axis=1,
    )
    sub["gap_working_hours"] = sub.apply(
        lambda r: (
            subtract_weekend_hours(r["prev_end"], r["start_at_planning"])
            if pd.notna(r["prev_end"])
            else np.nan
        ),
        axis=1,
    )
    return sub, gaps


def _filter_long_gaps(
    sub: pd.DataFrame, gaps: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bad = gaps[gaps["gap_working_hours"] > GAP_THRESHOLD_HOURS][
        "production_lot_id"
    ].unique()
    print(f"Filter 4 — gaps > 30 days: removed {len(bad):,} lots")
    return (
        sub[~sub["production_lot_id"].isin(bad)].copy(),
        gaps[~gaps["production_lot_id"].isin(bad)].copy(),
    )


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sub, gaps) after all four cleaning filters."""
    sub = df[STEP_COLS].copy()
    sub["start_at_planning"] = pd.to_datetime(sub["start_at_planning"], errors="coerce")
    sub["end_at_planning"] = pd.to_datetime(sub["end_at_planning"], errors="coerce")

    sub = _filter_zero_duration(sub)
    sub = _filter_same_start(sub)
    sub = _compute_gaps(sub)

    gaps = sub.dropna(subset=["gap_hours", "prev_end"]).copy()
    sub, gaps = _filter_overlapping_lots(sub, gaps)
    sub, gaps = _apply_working_hours(sub, gaps)
    sub, gaps = _filter_long_gaps(sub, gaps)

    return sub, gaps
