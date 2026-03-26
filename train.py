import numpy as np
from prepare_data import main
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


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
            # scale_pos_weight=scale_pos_weight,
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


info = main()
X_train_enc = info["X_train"]
y_train = info["y_train"]
X_valid_enc = info["X_valid"]
y_valid = info["y_valid"]
MODEL_TYPES = ["xgboost", "random_forest", "lightgbm"]
THRESHOLDS = np.arange(0.10, 0.96, 0.05)
X_test_enc = info["X_test"]
y_test = info["y_test"]


# =========================================================
# TRAIN + COMPARE
# =========================================================
results = []
pred_store = {}
model_store = {}

for model_type in MODEL_TYPES:
    model = build_model(model_type)
    if model is None:
        continue

    print(f"Training {model_type}...")

    if model_type == "xgboost":
        model.fit(
            X_train_enc,
            y_train,
            eval_set=[(X_valid_enc, y_valid)],
            verbose=False,
        )
    elif model_type == "lightgbm":
        model.fit(
            X_train_enc,
            y_train,
            eval_set=[(X_valid_enc, y_valid)],
        )
    else:
        model.fit(X_train_enc, y_train)

    valid_proba = model.predict_proba(X_valid_enc)[:, 1]
    test_proba = model.predict_proba(X_test_enc)[:, 1]

    best_th = find_best_threshold(y_valid, valid_proba, THRESHOLDS)

    valid_m = get_metrics(y_valid, valid_proba, th=0.5)
    test_m_05 = get_metrics(y_test, test_proba, th=0.5)
    test_m_best = get_metrics(y_test, test_proba, th=best_th)

    results.append(
        {
            "model": model_type,
            "valid_roc_auc": valid_m["roc_auc"],
            "valid_pr_auc": valid_m["pr_auc"],
            "test_roc_auc@0.5": test_m_05["roc_auc"],
            "test_pr_auc@0.5": test_m_05["pr_auc"],
            "test_f1@0.5": test_m_05["f1"],
            "best_threshold": best_th,
            "test_precision@best": test_m_best["precision"],
            "test_recall@best": test_m_best["recall"],
            "test_f1@best": test_m_best["f1"],
        }
    )

    pred_store[model_type] = {
        "valid_proba": valid_proba,
        "test_proba": test_proba,
        "best_th": best_th,
    }
    model_store[model_type] = model
