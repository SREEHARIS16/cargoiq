"""
CargoIQ - Prediction Module
------------------------------
Loads the trained artifacts and exposes a single function,
`predict_for_route`, that takes a plain dict of raw-ish inputs
(route, date, cargo type, recent history) and returns:
    - predicted next-day demand (tons)
    - predicted dynamic rate ($/kg)
    - a simple recommended action based on predicted utilization

This module is imported directly by the FastAPI app (api/main.py).
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

_demand_model = None
_rate_model = None
_feature_cols = None


def _load_artifacts():
    global _demand_model, _rate_model, _feature_cols
    if _demand_model is None:
        _demand_model = joblib.load(MODELS_DIR / "demand_model.joblib")
        _rate_model = joblib.load(MODELS_DIR / "rate_model.joblib")
        with open(MODELS_DIR / "feature_columns.json") as f:
            _feature_cols = json.load(f)
    return _demand_model, _rate_model, _feature_cols


def _build_feature_row(payload: dict, feature_cols: list) -> pd.DataFrame:
    """
    payload expected keys:
        route_id (str), cargo_type (str), date (str YYYY-MM-DD),
        distance_km (float), capacity_tons (float), fuel_index (float),
        demand_lag_1, demand_lag_7, demand_roll_mean_7, demand_roll_mean_14,
        rate_lag_1, rate_roll_mean_7   (recent-history numbers the caller supplies)
    """
    date = pd.to_datetime(payload["date"])
    row = {col: 0 for col in feature_cols}

    row["distance_km"] = payload["distance_km"]
    row["day_of_week"] = date.dayofweek
    row["month"] = date.month
    row["fuel_index"] = payload["fuel_index"]
    row["capacity_tons"] = payload["capacity_tons"]
    row["demand_lag_1"] = payload["demand_lag_1"]
    row["demand_lag_7"] = payload["demand_lag_7"]
    row["demand_roll_mean_7"] = payload["demand_roll_mean_7"]
    row["demand_roll_mean_14"] = payload["demand_roll_mean_14"]
    row["rate_lag_1"] = payload["rate_lag_1"]
    row["rate_roll_mean_7"] = payload["rate_roll_mean_7"]

    row["dow_sin"] = np.sin(2 * np.pi * row["day_of_week"] / 7)
    row["dow_cos"] = np.cos(2 * np.pi * row["day_of_week"] / 7)
    row["month_sin"] = np.sin(2 * np.pi * row["month"] / 12)
    row["month_cos"] = np.cos(2 * np.pi * row["month"] / 12)

    route_col = f"route_{payload['route_id']}"
    if route_col in row:
        row[route_col] = 1

    cargo_col = f"cargo_{payload['cargo_type']}"
    if cargo_col in row:
        row[cargo_col] = 1

    return pd.DataFrame([row])[feature_cols]


def predict_for_route(payload: dict) -> dict:
    demand_model, rate_model, feature_cols = _load_artifacts()
    X = _build_feature_row(payload, feature_cols)

    predicted_demand = float(demand_model.predict(X)[0])
    predicted_rate = float(rate_model.predict(X)[0])

    predicted_utilization = predicted_demand / max(payload["capacity_tons"], 1e-6)

    if predicted_utilization >= 0.95:
        action = "SURGE_PRICE — near full capacity, raise rate and prioritize high-value cargo"
    elif predicted_utilization >= 0.75:
        action = "HOLD — healthy demand, keep current pricing"
    else:
        action = "PROMOTE — under-utilized, consider discount to attract bookings"

    return {
        "route_id": payload["route_id"],
        "predicted_demand_tons": round(predicted_demand, 2),
        "predicted_rate_per_kg": round(predicted_rate, 3),
        "predicted_utilization": round(min(predicted_utilization, 1.5), 3),
        "recommended_action": action,
    }
