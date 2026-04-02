# data_loader.py
# Loading and merging
import pandas as pd
from config import (PLANNING_DT_COLS, PLANNING_PATH, PRODUCTION_LOTS_DT_COLS,
                    PRODUCTION_LOTS_PATH, TARGET_ORGS)

from helpers import normalize_datetime_columns


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    lots = pd.read_csv(PRODUCTION_LOTS_PATH, low_memory=False)
    planning = pd.read_csv(PLANNING_PATH, low_memory=False)

    lots = lots[lots["organization_id"].isin(TARGET_ORGS)]
    planning = planning[planning["organization_id"].isin(TARGET_ORGS)]

    lots = normalize_datetime_columns(lots, PRODUCTION_LOTS_DT_COLS)
    planning = normalize_datetime_columns(planning, PLANNING_DT_COLS)
    return lots, planning


def merge_planning_with_lots(
    planning: pd.DataFrame, lots: pd.DataFrame
) -> pd.DataFrame:
    return planning.merge(
        lots,
        how="left",
        left_on="production_lot_id",
        right_on="id",
        suffixes=("_planning", "_production_lots"),
    )
