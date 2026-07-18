"""
CargoIQ API
--------------
FastAPI service exposing the demand-forecasting + dynamic-pricing
models over HTTP. This is the deployable artifact: run it locally
with uvicorn, or containerize it with the provided Dockerfile.

Run locally:
    uvicorn api.main:app --reload --port 8000

Then visit:
    http://localhost:8000/docs   (interactive Swagger UI)
"""

import sys
from pathlib import Path

# Allow importing from src/ when running as "uvicorn api.main:app" from repo root
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import PredictionRequest, PredictionResponse
from predict import predict_for_route

app = FastAPI(
    title="CargoIQ API",
    description="Air cargo demand forecasting & dynamic pricing engine (portfolio project inspired by IBS Software's iCargo).",
    version="1.0.0",
)

# Allow the Streamlit dashboard (or any frontend) to call this API from a different origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "service": "CargoIQ API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        result = predict_for_route(request.model_dump())
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Run `python src/train.py` first.",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
