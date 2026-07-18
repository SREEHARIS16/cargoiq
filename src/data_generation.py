"""
CargoIQ - Synthetic Air Cargo Booking Data Generator
------------------------------------------------------
Generates a realistic (but synthetic) air-cargo booking dataset,
modeled on the kind of route/booking/capacity data an air-cargo
management platform (e.g. IBS Software's iCargo) would work with.

No real IBS or airline data is used anywhere in this project —
this is purely a synthetic dataset so the project is safe to
build, run, and show in an interview without any data licensing
or confidentiality concerns.

Run:
    python src/data_generation.py
Output:
    data/raw/cargo_bookings.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
N_DAYS = 730  # 2 years of daily data
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "cargo_bookings.csv"

# A handful of realistic long-haul / regional cargo routes
ROUTES = [
    {"route_id": "BLR-DXB", "origin": "BLR", "dest": "DXB", "distance_km": 2800, "base_capacity_tons": 45},
    {"route_id": "BLR-LHR", "origin": "BLR", "dest": "LHR", "distance_km": 7900, "base_capacity_tons": 60},
    {"route_id": "DEL-JFK", "origin": "DEL", "dest": "JFK", "distance_km": 11800, "base_capacity_tons": 70},
    {"route_id": "BOM-SIN", "origin": "BOM", "dest": "SIN", "distance_km": 3900, "base_capacity_tons": 50},
    {"route_id": "MAA-HKG", "origin": "MAA", "dest": "HKG", "distance_km": 3300, "base_capacity_tons": 40},
    {"route_id": "DEL-FRA", "origin": "DEL", "dest": "FRA", "distance_km": 6100, "base_capacity_tons": 55},
]

CARGO_TYPES = ["general", "perishable", "pharma", "e-commerce", "mail"]
CARGO_TYPE_WEIGHTS = [0.40, 0.15, 0.10, 0.30, 0.05]


def generate(seed: int = RNG_SEED, n_days: int = N_DAYS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")

    rows = []
    for route in ROUTES:
        # Route-level baseline demand and seasonality/trend components
        base_demand = route["base_capacity_tons"] * rng.uniform(0.55, 0.75)
        trend_per_day = rng.uniform(0.002, 0.01)  # slow organic growth

        for i, date in enumerate(dates):
            day_of_week = date.dayofweek  # 0=Mon
            month = date.month

            # Weekly seasonality: weekends lighter for B2B cargo
            weekly_factor = 0.8 if day_of_week >= 5 else 1.0

            # Yearly seasonality: Oct-Dec peak (e-commerce/holiday), Feb dip
            yearly_factor = 1.0 + 0.25 * np.sin((month - 10) / 12 * 2 * np.pi)

            # Fuel price proxy (affects pricing), random-walk-ish
            fuel_index = 100 + 15 * np.sin(i / 60) + rng.normal(0, 3)

            # Random demand shocks (e.g. disruption, promo)
            shock = rng.normal(0, 3)

            trend = trend_per_day * i
            demand_tons = max(
                1.0,
                base_demand * weekly_factor * yearly_factor + trend * base_demand + shock,
            )

            capacity_tons = route["base_capacity_tons"] * rng.uniform(0.95, 1.05)
            utilization = min(demand_tons / capacity_tons, 1.15)  # can slightly overbook/waitlist

            cargo_type = rng.choice(CARGO_TYPES, p=CARGO_TYPE_WEIGHTS)

            # Base rate per kg depends on distance, cargo type, fuel
            type_premium = {
                "general": 1.0, "perishable": 1.35, "pharma": 1.8,
                "e-commerce": 1.1, "mail": 0.9,
            }[cargo_type]
            base_rate_per_kg = (
                0.15 + route["distance_km"] * 0.00025
            ) * type_premium * (fuel_index / 100)

            # Actual achieved rate reacts to utilization (dynamic pricing signal)
            demand_pressure = max(0.0, utilization - 0.6) * 1.8
            achieved_rate_per_kg = base_rate_per_kg * (1 + demand_pressure) * rng.uniform(0.95, 1.05)

            rows.append({
                "date": date,
                "route_id": route["route_id"],
                "origin": route["origin"],
                "dest": route["dest"],
                "distance_km": route["distance_km"],
                "day_of_week": day_of_week,
                "month": month,
                "cargo_type": cargo_type,
                "fuel_index": round(fuel_index, 2),
                "capacity_tons": round(capacity_tons, 2),
                "demand_tons": round(demand_tons, 2),
                "utilization": round(utilization, 3),
                "achieved_rate_per_kg": round(achieved_rate_per_kg, 3),
            })

    df = pd.DataFrame(rows).sort_values(["date", "route_id"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df):,} rows -> {OUTPUT_PATH}")
    print(df.head())
