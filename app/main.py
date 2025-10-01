from fastapi import FastAPI
import joblib
import pandas as pd
import json

app = FastAPI(title="Weather Prediction API")

# Model paths
RAIN_MODEL_PATH = "models/rain_classifier.pkl"
PRECIP_MODEL_PATH = "models/precipitation_regressor.pkl"
META_MODEL1_PATH = "models/metadata_model1.json"
META_MODEL2_PATH = "models/metadata_model2.json"

# Load models
rain_model = joblib.load(RAIN_MODEL_PATH)
precip_model = joblib.load(PRECIP_MODEL_PATH)

# Load metadata (thresholds, feature list, etc.)
with open(META_MODEL1_PATH, "r") as f:
    meta1 = json.load(f)
with open(META_MODEL2_PATH, "r") as f:
    meta2 = json.load(f)

@app.get("/")
def home():
    return {"message": "Weather Prediction API is running!"}

@app.post("/predict/rain")
def predict_rain(data: dict):
    df = pd.DataFrame([data])
    pred_proba = rain_model.predict_proba(df)[0, 1]
    threshold = meta1.get("threshold", 0.5)  # default 0.5 if not in metadata
    pred = int(pred_proba >= threshold)
    return {"will_rain_plus7d": pred, "probability": float(pred_proba)}

@app.post("/predict/precip")
def predict_precip(data: dict):
    df = pd.DataFrame([data])
    pred = precip_model.predict(df)[0]
    return {"precip_3day_sum": float(pred)}
