"""
streamlit_app.py
----------------
Streamlit frontend for the Medical Diagnostics API.

Run standalone:
    streamlit run streamlit_app.py

Or let docker-compose start it automatically.
"""

import streamlit as st
import requests
import json
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Diagnostics",
    page_icon="🏥",
    layout="centered",
)

API_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000")

# ── Helpers ────────────────────────────────────────────────────────────────────

def call_api(payload: dict) -> dict | None:
    """POST to /predict and return the JSON response, or None on error."""
    # Try Docker-internal URL first, then localhost for local dev
    for base in [API_URL, "http://localhost:8000"]:
        try:
            r = requests.post(f"{base}/predict", json=payload, timeout=5)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError:
            continue
        except requests.exceptions.HTTPError as exc:
            st.error(f"API error {exc.response.status_code}: {exc.response.text}")
            return None
    st.error("❌ Could not reach the FastAPI service. Is it running?")
    return None


def health_check() -> bool:
    for base in [API_URL, "http://localhost:8000"]:
        try:
            r = requests.get(f"{base}/", timeout=3)
            return r.ok
        except requests.exceptions.ConnectionError:
            continue
    return False


# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("🏥 Medical Diagnostics System")
st.markdown(
    "Breast cancer diagnostic tool powered by a **Random Forest** model "
    "trained on the Scikit-learn Breast Cancer dataset. "
    "Enter the five tumour measurements below and click **Run Diagnosis**."
)
st.divider()

# Connection status badge
if health_check():
    st.success("🟢 API service is online", icon="✅")
else:
    st.warning("🔴 API service is offline – predictions will fail", icon="⚠️")

st.subheader("🔬 Tumour Measurements")

col1, col2 = st.columns(2)

with col1:
    mean_radius = st.number_input(
        "Mean Radius",
        min_value=0.0, max_value=100.0, value=14.0, step=0.1,
        help="Mean of distances from center to points on the perimeter",
    )
    mean_texture = st.number_input(
        "Mean Texture",
        min_value=0.0, max_value=50.0, value=19.0, step=0.1,
        help="Standard deviation of gray-scale values",
    )
    mean_perimeter = st.number_input(
        "Mean Perimeter",
        min_value=0.0, max_value=300.0, value=90.0, step=0.5,
        help="Mean size of the core tumour perimeter",
    )

with col2:
    mean_area = st.number_input(
        "Mean Area",
        min_value=0.0, max_value=3000.0, value=600.0, step=5.0,
        help="Mean area of the tumour (typically 100–2500)",
    )
    mean_smoothness = st.number_input(
        "Mean Smoothness",
        min_value=0.0, max_value=1.0, value=0.09, step=0.001,
        format="%.4f",
        help="Local variation in radius lengths (0.05–0.16 typical)",
    )

st.divider()

# Sample data buttons
st.caption("📋 Quick-fill with example cases:")
ex_col1, ex_col2, ex_col3 = st.columns(3)

if ex_col1.button("🟢 Typical Benign"):
    st.session_state["ex"] = dict(
        mean_radius=12.5, mean_texture=16.0,
        mean_perimeter=80.0, mean_area=490.0, mean_smoothness=0.08
    )
    st.rerun()

if ex_col2.button("🔴 Typical Malignant"):
    st.session_state["ex"] = dict(
        mean_radius=20.0, mean_texture=28.0,
        mean_perimeter=135.0, mean_area=1260.0, mean_smoothness=0.12
    )
    st.rerun()

if ex_col3.button("🟡 Borderline Case"):
    st.session_state["ex"] = dict(
        mean_radius=15.5, mean_texture=22.5,
        mean_perimeter=100.0, mean_area=750.0, mean_smoothness=0.10
    )
    st.rerun()

# Apply quick-fill values (only affects next render via session state)
if "ex" in st.session_state:
    vals = st.session_state.pop("ex")
    mean_radius      = vals["mean_radius"]
    mean_texture     = vals["mean_texture"]
    mean_perimeter   = vals["mean_perimeter"]
    mean_area        = vals["mean_area"]
    mean_smoothness  = vals["mean_smoothness"]

# ── Prediction ──────────────────────────────────────────────────────────────────

if st.button("🔍 Run Diagnosis", type="primary", use_container_width=True):
    payload = {
        "mean_radius":     mean_radius,
        "mean_texture":    mean_texture,
        "mean_perimeter":  mean_perimeter,
        "mean_area":       mean_area,
        "mean_smoothness": mean_smoothness,
    }

    with st.spinner("Analysing…"):
        result = call_api(payload)

    if result:
        diagnosis: str = result.get("diagnosis", "")
        is_benign = "Benign" in diagnosis

        st.divider()
        st.subheader("📋 Diagnosis Result")

        if is_benign:
            st.success(f"### ✅ {diagnosis}", icon="💚")
            st.info(
                "The model predicts a **benign** tumour (low risk). "
                "Please consult a qualified physician to confirm results."
            )
        else:
            st.error(f"### ⚠️ {diagnosis}", icon="🔴")
            st.warning(
                "The model predicts a **malignant** tumour (high risk). "
                "Immediate consultation with a medical professional is strongly advised."
            )

        with st.expander("📡 Raw API response"):
            st.json(result)

        with st.expander("📥 Request payload sent"):
            st.json(payload)

# ── Sidebar ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        **Model:** Random Forest Classifier  
        **Dataset:** Breast Cancer Wisconsin  
        **Features used:** 5 out of 30  
        **Experiment tracking:** MLflow  

        ---
        **Links**
        - 📊 [MLflow UI](http://localhost:5000)
        - 📖 [API Docs](http://localhost:8000/docs)
        ---
        ⚠️ *This tool is for educational purposes only and must not replace professional medical advice.*
        """
    )

    st.header("📊 Feature Reference")
    st.markdown(
        """
        | Feature | Typical range |
        |---|---|
        | Mean Radius | 6 – 28 |
        | Mean Texture | 9 – 40 |
        | Mean Perimeter | 43 – 190 |
        | Mean Area | 143 – 2501 |
        | Mean Smoothness | 0.05 – 0.16 |
        """
    )
