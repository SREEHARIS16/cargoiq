import sys
from pathlib import Path
import datetime

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from bootstrap import ensure_pipeline_ready
ensure_pipeline_ready()
from predict import predict_for_route

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "cargo_bookings.csv"

st.set_page_config(page_title="CargoIQ", page_icon="✈️", layout="centered")
st.title("✈️ CargoIQ — Air Cargo Demand & Dynamic Pricing")
st.caption("Portfolio project inspired by IBS Software's iCargo platform. All data is synthetic.")

df = pd.read_csv(RAW_PATH, parse_dates=["date"])
routes = sorted(df["route_id"].unique())
cargo_types = sorted(df["cargo_type"].unique())

st.subheader("1. Choose a route & date to forecast")
col1, col2 = st.columns(2)
with col1:
    route_id = st.selectbox("Route", routes)
with col2:
    cargo_type = st.selectbox("Cargo type", cargo_types)

target_date = st.date_input("Forecast date", datetime.date.today() + datetime.timedelta(days=1))

hist = df[df["route_id"] == route_id].sort_values("date")
recent = hist.tail(14)

if len(recent) < 7:
    st.warning("Not enough history for this route yet.")
else:
    demand_lag_1 = float(recent["demand_tons"].iloc[-1])
    demand_lag_7 = float(recent["demand_tons"].iloc[-7])
    demand_roll_mean_7 = float(recent["demand_tons"].tail(7).mean())
    demand_roll_mean_14 = float(recent["demand_tons"].mean())
    rate_lag_1 = float(recent["achieved_rate_per_kg"].iloc[-1])
    rate_roll_mean_7 = float(recent["achieved_rate_per_kg"].tail(7).mean())
    distance_km = float(recent["distance_km"].iloc[-1])
    capacity_tons = float(recent["capacity_tons"].mean())
    fuel_index = float(recent["fuel_index"].iloc[-1])

    with st.expander("Auto-filled recent history (editable)"):
        demand_lag_1 = st.number_input("Demand 1 day ago (tons)", value=demand_lag_1)
        demand_roll_mean_7 = st.number_input("7-day rolling avg demand (tons)", value=demand_roll_mean_7)
        capacity_tons = st.number_input("Route capacity (tons)", value=capacity_tons)
        fuel_index = st.number_input("Fuel price index", value=fuel_index)

    if st.button("🔮 Predict demand & recommended price", type="primary"):
        payload = {
            "route_id": route_id, "cargo_type": cargo_type, "date": str(target_date),
            "distance_km": distance_km, "capacity_tons": capacity_tons, "fuel_index": fuel_index,
            "demand_lag_1": demand_lag_1, "demand_lag_7": demand_lag_7,
            "demand_roll_mean_7": demand_roll_mean_7, "demand_roll_mean_14": demand_roll_mean_14,
            "rate_lag_1": rate_lag_1, "rate_roll_mean_7": rate_roll_mean_7,
        }
        result = predict_for_route(payload)
        st.subheader("2. Prediction")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted demand", f"{result['predicted_demand_tons']} t")
        m2.metric("Recommended rate", f"${result['predicted_rate_per_kg']}/kg")
        m3.metric("Predicted utilization", f"{result['predicted_utilization']*100:.1f}%")
        st.info(f"**Recommended action:** {result['recommended_action']}")

    st.subheader("Recent demand trend")
    st.line_chart(hist.set_index("date")["demand_tons"].tail(90))
