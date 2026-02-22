from fastapi import APIRouter
import pickle
import numpy as np

router = APIRouter()

with open("models/ml_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

WEATHER_MAP  = {"clear": 0, "rain": 1, "heavy_rain": 2, "fog": 3}
INCIDENT_MAP = {"none": 0, "minor": 1, "road_work": 2, "major": 3}

@router.post("/predict")
def predict_traffic(data: dict):
    hour     = data["hour"]
    dow      = data["day_of_week"]
    month    = data["month"]
    weather  = WEATHER_MAP[data["weather"]]
    vehicles = data["vehicle_count"]
    incident = INCIDENT_MAP[data["incident"]]
    holiday  = int(data["is_holiday"])

    features = np.array([[
        hour, dow, month, weather, vehicles,
        incident, holiday,
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        1 if dow in [5,6] else 0,
        1 if (7<=hour<=10 or 17<=hour<=20) else 0
    ]])

    scaled = scaler.transform(features)
    pred   = model.predict(scaled)[0]
    proba  = model.predict_proba(scaled)[0]

    label_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    speed_map = {0: 55.0, 1: 32.0, 2: 14.0}
    reco_map  = {
        0: "Traffic is light. Normal travel expected.",
        1: "Moderate delays. Consider alternate routes.",
        2: "Heavy congestion. Avoid if possible."
    }

    return {
        "congestion_level": label_map[pred],
        "confidence": round(float(max(proba)) * 100, 2),
        "estimated_speed_kmh": speed_map[pred],
        "recommendation": reco_map[pred]
    }