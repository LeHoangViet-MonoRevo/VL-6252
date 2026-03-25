import pandas as pd


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
