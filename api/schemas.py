from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    route_id: str = Field(..., examples=["BLR-DXB"])
    cargo_type: str = Field(..., examples=["pharma"])
    date: str = Field(..., examples=["2026-07-20"], description="YYYY-MM-DD")
    distance_km: float = Field(..., examples=[2800])
    capacity_tons: float = Field(..., examples=[46.2])
    fuel_index: float = Field(..., examples=[102.3])
    demand_lag_1: float = Field(..., description="Demand (tons) 1 day ago")
    demand_lag_7: float = Field(..., description="Demand (tons) 7 days ago")
    demand_roll_mean_7: float = Field(..., description="7-day rolling mean demand")
    demand_roll_mean_14: float = Field(..., description="14-day rolling mean demand")
    rate_lag_1: float = Field(..., description="Rate ($/kg) 1 day ago")
    rate_roll_mean_7: float = Field(..., description="7-day rolling mean rate")


class PredictionResponse(BaseModel):
    route_id: str
    predicted_demand_tons: float
    predicted_rate_per_kg: float
    predicted_utilization: float
    recommended_action: str
