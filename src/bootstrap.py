import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

RAW_PATH = ROOT / "data" / "raw" / "cargo_bookings.csv"
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
DEMAND_MODEL_PATH = ROOT / "models" / "demand_model.joblib"
RATE_MODEL_PATH = ROOT / "models" / "rate_model.joblib"


def ensure_pipeline_ready():
    import data_generation
    import preprocessing
    import pandas as pd

    if not RAW_PATH.exists():
        print("[bootstrap] No raw data found — generating synthetic dataset...")
        df = data_generation.generate()
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_PATH, index=False)

    if not FEATURES_PATH.exists():
        print("[bootstrap] No feature table found — building features...")
        raw = pd.read_csv(RAW_PATH)
        features = preprocessing.build_features(raw)
        FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(FEATURES_PATH, index=False)

    if not DEMAND_MODEL_PATH.exists() or not RATE_MODEL_PATH.exists():
        print("[bootstrap] No trained models found — training now...")
        import train
        train.main()

    print("[bootstrap] Pipeline ready.")


if __name__ == "__main__":
    ensure_pipeline_ready()
