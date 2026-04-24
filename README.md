# Medical Diagnostics System (SIS-3)

Breast cancer diagnostics demo project with:
1. **Model training + tracking** in MLflow
2. **Prediction API** in FastAPI
3. **Web frontend** in Streamlit
4. **Docker Compose orchestration** for all services

The model uses 5 tumor features from the Scikit-learn Breast Cancer dataset and returns:
- diagnosis (`Benign (Low risk)` or `Malignant (High risk)`)
- class probabilities

## What each file does

| File | Purpose |
|---|---|
| `main.py` | FastAPI app, loads model, exposes `/`, `/health`, `/predict` |
| `train_mlflow.py` | Trains RandomForest, logs params/metrics/artifacts to MLflow, registers model |
| `streamlit_app.py` | UI for entering 5 features and calling `/predict` |
| `docker-compose.yml` | Starts mlflow + trainer + fastapi + streamlit in correct order |
| `Dockerfile` | FastAPI image |
| `Dockerfile.trainer` | One-shot trainer image |
| `Dockerfile.streamlit` | Streamlit image |
| `model.joblib` | Bundled fallback model |
| `requirements.txt` | Python dependencies |

## Architecture and startup order

Compose starts services in this sequence:
1. **mlflow** (`localhost:5000`) starts first
2. **trainer** waits for MLflow health, trains model, writes `/app/model/model.joblib`, then exits
3. **fastapi** waits for successful trainer completion, then starts on `localhost:8000`
4. **streamlit** waits until FastAPI is healthy, then starts on `localhost:8501`

This avoids race conditions where API starts before the trained model is available.

## Prerequisites

- Docker Desktop (running)
- Docker Compose v2 (`docker compose`)

## Run the full system

From project root:

```powershell
docker compose down
docker compose up -d --build
docker compose ps
```

Expected state:
- `mlflow_server` -> `Up (healthy)`
- `model_trainer` -> `Exited (0)` (this is expected)
- `fastapi_app` -> `Up (healthy)`
- `streamlit_app` -> `Up`

## Demo the Streamlit frontend

1. Open: `http://localhost:8501`
2. Confirm status badge shows API is online
3. Enter or keep default feature values
4. Click **Run Diagnosis**
5. Verify response section shows:
   - diagnosis in English (`Benign (Low risk)` or `Malignant (High risk)`)
   - probabilities for `malignant` and `benign`
   - raw API payload/response in expanders

## Demo FastAPI directly

Open docs:
- Swagger UI: `http://localhost:8000/docs`

Quick checks from PowerShell:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health"
Invoke-RestMethod -Uri "http://localhost:8000/"
```

Prediction request:

```powershell
$payload = @{
  mean_radius = 14.0
  mean_texture = 19.0
  mean_perimeter = 90.0
  mean_area = 600.0
  mean_smoothness = 0.09
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/predict" `
  -ContentType "application/json" `
  -Body $payload
```

Example response:

```json
{
  "diagnosis": "Benign (Low risk)",
  "analyzed_features": 5,
  "probabilities": {
    "malignant": 0.1217,
    "benign": 0.8783
  }
}
```

## How to test and troubleshoot

### 1. Container status

```powershell
docker compose ps
```

### 2. Service logs

```powershell
docker compose logs --tail=200 fastapi
docker compose logs --tail=200 streamlit
docker compose logs --tail=200 trainer
docker compose logs --tail=200 mlflow
```

### 3. Rebuild from clean state

```powershell
docker compose down -v
docker compose up -d --build
```

### 4. Typical issues

| Symptom | Likely cause | Fix |
|---|---|---|
| FastAPI exits on startup | model file not available yet | ensure trainer completed successfully, then restart compose |
| Streamlit shows API offline | FastAPI not healthy or wrong URL | check `fastapi` logs and `FASTAPI_URL` env in compose |
| No MLflow UI | port conflict on `5000` | free port 5000 or remap port in compose |

## Run locally without Docker (optional)

```powershell
pip install -r requirements.txt
mlflow server --host 0.0.0.0 --port 5000
python train_mlflow.py
uvicorn main:app --reload
streamlit run streamlit_app.py
```

## Notes

- This project is for educational/demo use.
- It does not replace clinical diagnosis.
