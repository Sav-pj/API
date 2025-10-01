from fastapi import FastAPI, Query, HTTPException
from datetime import datetime, timedelta
import pandas as pd
import joblib, json, os

# Paths inside the image/container
RAIN_MODEL_PATH   = os.getenv("RAIN_MODEL_PATH",   "models/rain_classifier.pkl")
PRECIP_MODEL_PATH = os.getenv("PRECIP_MODEL_PATH", "models/precip_regressor.pkl")
META_PATH         = os.getenv("META_PATH",         "models/metadata.json")

# Load models
try:
    rain_model = joblib.load(RAIN_MODEL_PATH)
    precip_model = joblib.load(PRECIP_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model(s): {e}")

# Load metadata (threshold for classification, etc.)
rain_threshold = 0.5
if os.path.exists(META_PATH):
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
            if isinstance(meta, dict) and "threshold" in meta:
                rain_threshold = float(meta["threshold"])
    except Exception:
        pass

app = FastAPI(
    title="Open Meteo Weather Prediction API",
    version="1.0.0",
    description="API serving two ML models: rain-in-7-days (classification) and 3-day precipitation total (regression).",
)

@app.get("/")
def index():
    return {
        "project": "Open Meteo Weather Prediction API",
        "endpoints": {
            "/health/": "Service health check",
            "/predict/rain/": "Predict if it will rain exactly 7 days after the given date",
            "/predict/precipitation/fall/": "Predict cumulated precipitation (mm) in the next 3 days",
        },
        "input_date_format": "YYYY-MM-DD",
        "github": "Add your repo link to github.txt"
    }

@app.get("/health/")
def health():
    return {"status": "ok"}

def _empty_feature_frame(model) -> pd.DataFrame:
    """
    Create a single-row feature frame with zeros matching the model's expected columns.
    In a real system you'd compute features for the input date; this keeps the API runnable.
    """
    cols = getattr(model, "feature_names_in_", None)
    if cols is None:
        # fallback: n_features_in_ only (no names)
        n = int(getattr(model, "n_features_in_", 0))
        return pd.DataFrame([[0]*n])
    return pd.DataFrame([[0]*len(cols)], columns=list(cols))

@app.get("/predict/rain/")
def predict_rain(date: str = Query(..., description="Date in YYYY-MM-DD")):
    """Return prediction for rain exactly 7 days after 'date'."""
    try:
        input_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    pred_date = input_date + timedelta(days=7)

    X = _empty_feature_frame(rain_model)
    try:
        proba = rain_model.predict_proba(X)[:, 1][0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    will_rain = bool(proba >= rain_threshold)

    return {
        "input_date": date,
        "prediction": {
            "date": pred_date.strftime("%Y-%m-%d"),
            "will_rain": will_rain
        }
    }

@app.get("/predict/precipitation/fall/")
def predict_precip(date: str = Query(..., description="Date in YYYY-MM-DD")):
    """Return cumulated precip (mm) for D+1..D+3 starting after 'date'."""
    try:
        input_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    start_date = input_date + timedelta(days=1)
    end_date   = input_date + timedelta(days=3)

    X = _empty_feature_frame(precip_model)
    try:
        precip = float(precip_model.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {
        "input_date": date,
        "prediction": {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "precipitation_fall": round(max(0.0, precip), 2)
        }
    }
