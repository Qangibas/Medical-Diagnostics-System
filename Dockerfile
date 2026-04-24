FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
# model.joblib is loaded from the shared Docker volume at runtime
# (MODEL_PATH env var points to /app/model/model.joblib)
# Copy the original as fallback for local runs without Docker
COPY model.joblib .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
