import numpy as np
import pandas as pd
import xgboost as xgb
from pipeline import run
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model_df, FEATURES, TARGET = run()

# ── Config ────────────────────────────────────────────────────────────────────

TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_LIST = {
    "linear_regression": LinearRegression(),
    "ridge": Ridge(alpha=1.0),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100, random_state=RANDOM_STATE
    ),
    "xgboost": xgb.XGBRegressor(n_estimators=100, random_state=RANDOM_STATE),
}

SELECTED_MODEL = "random_forest"  # change to try different models

# ── Prepare: sort by time, drop nulls ────────────────────────────────────────
data = (
    model_df[FEATURES + [TARGET, "created_at_planning"]]
    .dropna(subset=FEATURES + [TARGET, "created_at_planning"])
    .sort_values("created_at_planning")
    .reset_index(drop=True)
)

print(f"Total samples: {len(data):,}")
print(
    f'Date range: {data["created_at_planning"].min().date()} → {data["created_at_planning"].max().date()}'
)
print()

# ── Temporal split ────────────────────────────────────────────────────────────

split_idx = int(len(data) * (1 - TEST_SIZE))
split_date = data.iloc[split_idx]["created_at_planning"]

train_df = data.iloc[:split_idx].copy()
test_df = data.iloc[split_idx:].copy()

print(
    f'Train: {len(train_df):,}  ({train_df["created_at_planning"].min().date()} → {train_df["created_at_planning"].max().date()})'
)
print(
    f'Test:  {len(test_df):,}   ({test_df["created_at_planning"].min().date()} → {test_df["created_at_planning"].max().date()})'
)
print(f"Split date: {split_date.date()}")
print()

# ── Prepare X / y ─────────────────────────────────────────────────────────────

X_train = train_df[FEATURES]
y_train_log = np.log1p(train_df[TARGET])

X_test = test_df[FEATURES]
y_test_orig = test_df[TARGET]
y_test_log = np.log1p(y_test_orig)

# ── Train ─────────────────────────────────────────────────────────────────────

model = MODEL_LIST[SELECTED_MODEL]
model.fit(X_train, y_train_log)


# ── Evaluate ──────────────────────────────────────────────────────────────────

y_pred_log = model.predict(X_test)
y_pred = np.clip(np.expm1(y_pred_log), 0, None)

mae = mean_absolute_error(y_test_orig, y_pred)
rmse = mean_squared_error(y_test_orig, y_pred) ** 0.5
r2 = r2_score(y_test_orig, y_pred)
r2_log = r2_score(y_test_log, y_pred_log)

print(f"=== {SELECTED_MODEL} ===")
print(f"MAE:       {mae:.2f}h")
print(f"RMSE:      {rmse:.2f}h")
print(f"R2 (orig): {r2:.4f}")
print(f"R2 (log):  {r2_log:.4f}")
print()

# ── Error breakdown by actual bucket ─────────────────────────────────────────

results = pd.DataFrame(
    {
        "actual": y_test_orig.values,
        "predicted": y_pred,
        "error": np.abs(y_test_orig.values - y_pred),
    }
)
results["actual_bucket"] = pd.cut(
    results["actual"],
    bins=[-np.inf, 0, 8, 24, np.inf],
    labels=["zero", "short", "medium", "long"],
)
print("=== MAE by bucket ===")
print(
    results.groupby("actual_bucket", observed=True)["error"]
    .agg(["mean", "median", "count"])
    .round(2)
)
print()

# ── Feature importance (tree-based only) ─────────────────────────────────────
if hasattr(model, "feature_importances_"):
    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(
        ascending=False
    )
    print("=== Top 15 feature importances ===")
    print(fi.head(15).round(4).to_string())
