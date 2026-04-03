# pipeline.py
import features
from cleaner import clean
from data_loader import load_raw, merge_planning_with_lots


def run():
    lots, planning = load_raw()
    df_raw = merge_planning_with_lots(planning, lots)

    sub, gaps = clean(df_raw)

    model_df, FEATURES = features.build(gaps, sub, df_raw)
    TARGET = "gap_working_hours"

    print(f"Samples: {len(model_df):,}  |  Features: {len(FEATURES)}")
    null_counts = model_df[FEATURES + [TARGET]].isna().sum()
    print(null_counts[null_counts > 0].to_string())

    return model_df, FEATURES, TARGET


if __name__ == "__main__":
    model_df, FEATURES, TARGET = run()
