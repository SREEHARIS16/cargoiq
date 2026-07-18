"""
CargoIQ - Preprocessing & Feature Engineering
------------------------------------------------
Turns raw booking rows into a model-ready feature table:
    - lag features (previous day/week demand per route)
    - rolling averages (7-day, 14-day demand trend)
    - one-hot encoding for categoricals (route_id, cargo_type)
    - cyclical encoding for day_of_week / month

Run:
    python src/preprocessing.py
Input:
    data/raw/cargo_bookings.csv
Output:
    data/processed/features.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "cargo_bookings.csv"
PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "features.csv"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["route_id", "date"])

    # --- Lag & rolling features (per route) ---
    grp = df.groupby("route_id")["demand_tons"]
    df["demand_lag_1"] = grp.shift(1)
    df["demand_lag_7"] = grp.shift(7)
    df["demand_roll_mean_7"] = grp.transform(lambda s: s.shift(1).rolling(7).mean())
    df["demand_roll_mean_14"] = grp.transform(lambda s: s.shift(1).rolling(14).mean())

    rate_grp = df.groupby("route_id")["achieved_rate_per_kg"]
    df["rate_lag_1"] = rate_grp.shift(1)
    df["rate_roll_mean_7"] = rate_grp.transform(lambda s: s.shift(1).rolling(7).mean())

    # --- Cyclical encodings ---
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- One-hot encode categoricals ---
    df = pd.get_dummies(df, columns=["route_id", "cargo_type"], prefix=["route", "cargo"])

    # Drop rows with NaNs introduced by lag/rolling windows (first ~14 days per route)
    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    raw = pd.read_csv(RAW_PATH)
    features = build_features(raw)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(PROCESSED_PATH, index=False)
    print(f"Built {features.shape[1]} features across {len(features):,} rows -> {PROCESSED_PATH}")
