import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="Medical Diagnostics API",
    description="Breast cancer diagnostic API powered by a Random Forest model.",
    version="2.0.0",
)

# Model path is configurable via env var so Docker can point to the shared volume
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    # Fallback to bundled model so the API can still boot if shared volume is empty.
    model = joblib.load("model.joblib")


class MedicalRecord(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float


@app.get("/")
def read_root():
    return {"message": "Medical API is running. Ready for diagnostics!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_diagnosis(record: MedicalRecord):
    data = np.array([[
        record.mean_radius,
        record.mean_texture,
        record.mean_perimeter,
        record.mean_area,
        record.mean_smoothness,
    ]])

    prediction = model.predict(data)
    probability = model.predict_proba(data)[0].tolist()

    if prediction[0] == 1:
        diagnosis = "Benign (Low risk)"
    else:
        diagnosis = "Malignant (High risk)"

    return {
        "diagnosis": diagnosis,
        "analyzed_features": 5,
        "probabilities": {
            "malignant": round(probability[0], 4),
            "benign":    round(probability[1], 4),
        },
    }
