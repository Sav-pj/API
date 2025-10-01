# app/main.py
from fastapi import FastAPI, HTTPException, Query
from datetime import datetime, timedelta
import pandas as pd
import joblib, json, os, glob

def find_first(pattern, default=None):
    files = glob.glob(pattern)
    return files[0] if files else default

# Resolve file paths (prints help in logs)
RAIN_MODEL_PATH   = os.getenv("RAIN_MODEL_PATH",   find_first("models/*rain*classifier*.pkl") or "models/rain_classifier.pkl")
PRECIP_MODEL_PATH = os.getenv("PRECIP_MODEL_PATH", find_first("models/*precip*regressor*.pkl") or "models/precipitation_regressor.pkl")
META_MODEL1_PATH  = os.getenv("META_MODEL1_PATH",  find_first("models/*metadata*_model1*.json") or "models/metadata_model1.json")
META_MODEL2_PATH  = os.getenv("META_MODEL2_PATH",  find_first("models/*metadata*_model2*.json") or "models/metadata_model2.json")

print("Resolved paths:")
print(" RAIN_MODEL_PATH  =", RAIN_MODEL_PATH)
print(" PRECIP_MODEL_PATH=", PRECIP_MODEL_PATH)
print(" META_MODEL1_PATH =", META_MODEL1_PATH)
print(" META_MODEL2_PATH =", META_MODEL2_PATH)

# ---- CREATE THE FASTAPI INSTANCE NAMED *app* ----
app = FastAPI(
    title="Open Meteo Weather Prediction API",
    version="1.0.0",
    description="Rain +7d classifier and 3-day precipitation regressor.",
)

# Load models safely
try:
    rain_model = joblib.load(RAIN_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load rain model: {e}")

try:
    precip_model = joblib.load(PRECIP_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load precip model: {e}")

# Load metadata (thresholds)
rain_threshold = 0.5
try:
    if META_MODEL1_PATH and os.path.exists(META_MODEL1_PATH):
        with open(META_MODEL1_PATH, "r", encoding="utf-8") as f:
            meta1 = json.load(f)
            if isinstance(meta1, dict) and "threshold" in meta1:
                rain_threshold = float(meta1["threshold"])
except Exception:
    pass

def _empty_feature_frame(model) -> pd.DataFrame:
    cols = getattr(model, "feature_names_in_", None)
    if cols is None:
        n = int(getattr(model, "n_features_in_", 0))
        return pd.DataFrame([[0]*n])
    return pd.DataFrame([[0]*len(cols)], columns=list(cols))

@app.get("/")
def index():
    return {
        "project": "Open Meteo Weather Prediction API",
        "endpoints": {
            "/health/": "Service health check",
            "/predict/rain/": "Predict if it will rain exactly 7 days after the given date",
            "/predict/precipitation/fall/": "Predict cumulated precipitation (mm) in the next 3 days",
            "/docs": "Swagger UI",
            "/redoc": "ReDoc",
        },
        "input_date_format": "YYYY-MM-DD",
        "github": "Add repo link in github.txt"
    }

@app.get("/health/")
def health():
    return {"status": "ok"}

@app.get("/predict/rain/")
def predict_rain(date: str = Query(..., description="YYYY-MM-DD")):
    try:
        input_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    pred_date = input_date + timedelta(days=7)
    X = _empty_feature_frame(rain_model)
    try:
        proba = float(rain_model.predict_proba(X)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    will_rain = bool(proba >= rain_threshold)
    return {
        "input_date": date,
        "prediction": {"date": pred_date.strftime("%Y-%m-%d"), "will_rain": will_rain}
    }

@app.get("/predict/precipitation/fall/")
def predict_precip(date: str = Query(..., description="YYYY-MM-DD")):
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

# Optional: local run helper
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
