"""
train_mlflow.py
---------------
Trains a RandomForestClassifier on the Breast Cancer dataset,
logs everything to MLflow (params, metrics, artifacts),
and registers the best model in the MLflow Model Registry.

Run:
    python train_mlflow.py
"""

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import joblib

# ── Configuration ──────────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI = "http://mlflow:5000"   # MLflow server inside Docker network
EXPERIMENT_NAME     = "medical-diagnostics"
MODEL_REGISTRY_NAME = "MedicalDiagnosticsRF"

# Hyperparameters to log
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth":    5,
    "random_state": 42,
    "max_features": "sqrt",
    "criterion":    "gini",
}

# Only the 5 features the API uses
FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
]

# ── Data preparation ───────────────────────────────────────────────────────────

def load_data():
    """Load Breast Cancer dataset and keep only the 5 API features."""
    raw = load_breast_cancer()
    # Feature indices: 0=radius, 1=texture, 2=perimeter, 3=area, 4=smoothness
    X = raw.data[:, :5]
    y = raw.target          # 1 = benign, 0 = malignant
    return X, y, raw.target_names

# ── Training & tracking ────────────────────────────────────────────────────────

def train_and_log():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Create (or reuse) the experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y, class_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RF_PARAMS["random_state"], stratify=y
    )

    with mlflow.start_run(run_name="RandomForest-v1") as run:
        run_id = run.info.run_id
        print(f"MLflow run started  →  run_id: {run_id}")

        # ── 1. Log hyperparameters ─────────────────────────────────────────────
        mlflow.log_params(RF_PARAMS)
        mlflow.log_param("test_size",    0.2)
        mlflow.log_param("n_features",   5)
        mlflow.log_param("dataset",      "breast_cancer_sklearn")

        # ── 2. Train ───────────────────────────────────────────────────────────
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_train, y_train)

        # ── 3. Evaluate & log metrics ──────────────────────────────────────────
        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "f1_score":  f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall":    recall_score(y_test, y_pred, average="weighted"),
            "roc_auc":   roc_auc_score(y_test, y_proba),
            "train_size": len(X_train),
            "test_size":  len(X_test),
        }
        mlflow.log_metrics(metrics)

        print("\n📊 Evaluation metrics:")
        for k, v in metrics.items():
            print(f"   {k:12s} = {v:.4f}" if isinstance(v, float) else f"   {k:12s} = {v}")

        # ── 4. Log model artifact ──────────────────────────────────────────────
        # Save locally first (so the FastAPI app can load it without MLflow)
        joblib.dump(clf, "model.joblib")
        print("\n💾 model.joblib saved locally")

        # Log the joblib file as an MLflow artifact
        mlflow.log_artifact("model.joblib", artifact_path="model_files")

        # Also log as an MLflow sklearn model (enables serving via `mlflow models serve`)
        signature = infer_signature(X_train, clf.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="sklearn_model",
            signature=signature,
            input_example=X_test[:3],
            registered_model_name=MODEL_REGISTRY_NAME,   # auto-registers on log
        )

        model_uri = f"runs:/{run_id}/sklearn_model"
        print(f"\n✅ Model logged at: {model_uri}")

    # ── 5. Register & transition to Production ─────────────────────────────────
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Fetch the latest version that was just registered
    latest_versions = client.get_latest_versions(
        MODEL_REGISTRY_NAME, stages=["None"]
    )
    if latest_versions:
        version = latest_versions[-1].version
        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(
            f"\n🏷️  Model '{MODEL_REGISTRY_NAME}' v{version} "
            "transitioned to 'Production'"
        )

    print("\n🎉 Training complete. Open the MLflow UI at http://localhost:5000")


if __name__ == "__main__":
    train_and_log()
