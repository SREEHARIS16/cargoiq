"""
CargoIQ - Training Pipeline
------------------------------
Trains:
    1. DemandModel  - predicts demand_tons (next-day cargo demand per route)
    2. RateModel    - predicts achieved_rate_per_kg (dynamic pricing)

Uses a time-based split (train on the past, test on the most recent
slice) instead of a random split, since this is time-series data and
a random split would leak future information into training.

Run:
    python src/train.py
Outputs:
    models/demand_model.joblib
    models/rate_model.joblib
    models/feature_columns.json
    models/metrics.json
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

from model import build_demand_model, build_rate_model

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
MODELS_DIR = ROOT / "models"

TARGET_DEMAND = "demand_tons"
TARGET_RATE = "achieved_rate_per_kg"

# Columns that must never be used as model inputs (targets, raw dates, leaky same-day columns)
DROP_COLS = {
    "date", TARGET_DEMAND, TARGET_RATE,
    "utilization",  # same-day utilization is derived from demand -> leakage for demand model
    "origin", "dest",  # redundant with the one-hot route_id columns, and non-numeric
}


def time_split(df: pd.DataFrame, test_frac: float = 0.15):
    df = df.sort_values("date")
    cutoff_idx = int(len(df) * (1 - test_frac))
    cutoff_date = df.iloc[cutoff_idx]["date"]
    train = df[df["date"] < cutoff_date]
    test = df[df["date"] >= cutoff_date]
    return train, test


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "mape": round(float(mean_absolute_percentage_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def main():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    train_df, test_df = time_split(df)

    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    X_train, X_test = train_df[feature_cols], test_df[feature_cols]

    metrics = {}

    # --- Demand model ---
    y_train_d, y_test_d = train_df[TARGET_DEMAND], test_df[TARGET_DEMAND]
    demand_model = build_demand_model()
    demand_model.fit(X_train, y_train_d)
    preds_d = demand_model.predict(X_test)
    metrics["demand_model"] = evaluate(y_test_d, preds_d)

    # --- Rate model (dynamic pricing) ---
    y_train_r, y_test_r = train_df[TARGET_RATE], test_df[TARGET_RATE]
    rate_model = build_rate_model()
    rate_model.fit(X_train, y_train_r)
    preds_r = rate_model.predict(X_test)
    metrics["rate_model"] = evaluate(y_test_r, preds_r)

    # --- Save artifacts ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(demand_model, MODELS_DIR / "demand_model.joblib")
    joblib.dump(rate_model, MODELS_DIR / "rate_model.joblib")
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,} | Features: {len(feature_cols)}")


if __name__ == "__main__":
    main()
