# CargoIQ — Air Cargo Demand Forecasting & Dynamic Pricing Engine

An end-to-end, **deployable** ML system that predicts next-day air cargo
demand per route and recommends a dynamic price, inspired by how IBS
Software's **iCargo** platform describes its own capabilities: *"a
dynamic pricing engine, advanced forecasting algorithms, and
instant accounting reconciliation."* This project builds a small,
honest version of that idea from scratch, on synthetic data, so it's
safe to develop and demo without touching any real company data.

> All data here is **synthetically generated** (see `src/data_generation.py`).
> No real IBS Software or airline data is used anywhere.

---

## What it does

1. **Forecasts demand** (tons of cargo) for a given route/date, using
   historical trend, seasonality, and recent booking momentum.
2. **Recommends a dynamic price** ($/kg) that reacts to predicted
   utilization — the same idea behind revenue-management pricing
   engines in cargo/airline systems.
3. Serves both as a **REST API** (FastAPI) and an **interactive
   dashboard** (Streamlit), and ships as a **Docker container** ready
   to deploy to any free-tier host.

## Project structure

```
cargoiq/
├── data/
│   ├── raw/              # generated synthetic booking data (csv)
│   └── processed/        # engineered feature table (csv)
├── src/
│   ├── data_generation.py   # step 1: generate synthetic dataset
│   ├── preprocessing.py     # step 2: feature engineering
│   ├── model.py              # model definitions
│   ├── train.py               # step 3: train + evaluate + save models
│   └── predict.py             # inference logic used by the API/dashboard
├── api/
│   ├── main.py             # FastAPI app (the deployable service)
│   └── schemas.py          # request/response validation models
├── frontend/
│   └── streamlit_app.py    # interactive demo dashboard
├── models/                 # trained model artifacts (.joblib) + metrics.json
├── tests/
│   └── test_api.py         # pytest suite for the API
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Step-by-step: run it locally in VS Code

### 0. Prerequisites
- Python 3.10+ installed
- VS Code with the **Python extension** installed
- (Optional) Docker Desktop, if you want to test the container build

### 1. Open the project
```bash
cd cargoiq
code .
```

### 2. Create and activate a virtual environment
In the VS Code terminal (`` Ctrl+` ``):
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```
In VS Code, select this `.venv` as your interpreter: `Ctrl+Shift+P` →
**"Python: Select Interpreter"** → pick the one inside `.venv`.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate the synthetic dataset
```bash
python src/data_generation.py
```
→ creates `data/raw/cargo_bookings.csv`

### 5. Build features
```bash
python src/preprocessing.py
```
→ creates `data/processed/features.csv`

### 6. Train the models
```bash
cd src
python train.py
cd ..
```
→ creates `models/demand_model.joblib`, `models/rate_model.joblib`,
`models/feature_columns.json`, `models/metrics.json`

You should see output like:
```json
{
  "demand_model": {"mae": 3.74, "mape": 0.026, "r2": 0.99},
  "rate_model":   {"mae": 0.15, "mape": 0.046, "r2": 0.985}
}
```
(These are from a time-based train/test split — the model is tested
only on the most recent slice of dates, never on data it trained on,
so the numbers reflect genuine forecasting performance, not leakage.)

### 7. Run the API
```bash
uvicorn api.main:app --reload --port 8000
```
Open **http://localhost:8000/docs** — this gives you an interactive
Swagger UI to test the `/predict` endpoint directly in the browser.

### 8. Run the dashboard (in a second terminal)
```bash
streamlit run frontend/streamlit_app.py
```
Open **http://localhost:8501** — pick a route, a date, click predict,
and you'll see the forecasted demand, recommended rate, and a
recommended action (surge price / hold / promote).

### 9. Run the tests
```bash
pytest tests/ -v
```

---

## Running with Docker (what makes this "deployable", not just a script)

```bash
docker compose up --build
```
This builds one image, generates data, trains models, and starts:
- the API on **http://localhost:8000**
- the dashboard on **http://localhost:8501**

To build/run just the API image manually:
```bash
docker build -t cargoiq-api .
docker run -p 8000:8000 cargoiq-api
```

---

## Deploying it for real (free tier options)

Pick **one** of these to get a live public URL you can put on your
resume/LinkedIn and hand to interviewers:

### Option A — Render (easiest for the API)
1. Push this repo to GitHub.
2. Go to render.com → New → Web Service → connect your repo.
3. Environment: Docker. Render will detect the `Dockerfile` automatically.
4. Deploy. You get a public URL like `https://cargoiq-api.onrender.com`.

### Option B — Streamlit Community Cloud (easiest for the dashboard)
1. Push this repo to GitHub.
2. Go to share.streamlit.io → New app → point it at
   `frontend/streamlit_app.py`.
3. Add `requirements.txt` (already there) — it installs automatically.
4. Deploy. You get a public URL like `https://your-app.streamlit.app`.

### Option C — Railway or Fly.io (API + Docker, similar flow to Render)
Both auto-detect the `Dockerfile` the same way Render does — connect
the GitHub repo and deploy.

**Recommended combo for your resume:** deploy the API on Render *and*
the dashboard on Streamlit Cloud, and point the dashboard's optional
`requests` calls at the live Render URL instead of importing
`predict.py` directly, so the two services genuinely talk to each
other over the network — that's a more convincing "production-style"
architecture to describe in an interview.

---

## How this maps to what you'd say in an IBS Software interview

- **iCargo** (IBS's cargo platform) explicitly advertises a *dynamic
  pricing engine* and *advanced forecasting algorithms* — this
  project is a small, honest, from-scratch version of that same
  problem: given historical booking/utilization patterns, forecast
  demand and adjust price accordingly.
- You can speak to real ML engineering concerns, not just model
  accuracy: **time-based train/test splitting** (no data leakage on
  time series), **feature engineering** (lags, rolling windows,
  cyclical encoding), and **serving** (REST API + containerization).
- Be upfront in interviews that the dataset is **synthetic** — that's
  a strength, not a weakness: it shows you understood the problem
  well enough to simulate it realistically, without needing (or
  claiming to have) proprietary company data.

## Natural next steps (good talking points / things to extend)

- Swap `GradientBoostingRegressor` for `XGBoost`/`LightGBM` and compare.
- Add prediction intervals (quantile regression) instead of point estimates.
- Add a `/retrain` endpoint or a scheduled job to retrain on new data.
- Add authentication (API key) to the FastAPI service before public deployment.
- Log predictions to a database and build a simple accuracy-drift monitor.
