import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page Config
st.set_page_config(
    page_title="Compressor Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# Load Model and Scaler
@st.cache_resource
def load_model():
    # Use absolute paths relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '../models/random_forest_model.pkl')
    scaler_path = os.path.join(current_dir, '../models/scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model()

# Header
st.title("⚙️ Compressor Predictive Maintenance Dashboard")
st.markdown("""
This dashboard predicts the probability of a compressor machine failing within the next **30 days**.
Use the sliders in the sidebar to simulate sensor readings.
""")

# Sidebar Inputs
st.sidebar.header("SIMULATION PARAMETERS")

def user_input_features():
    operating_hours = st.sidebar.slider("Operating Hours", 0, 10000, 5000)
    temperature = st.sidebar.slider("Temperature (°C)", 40.0, 120.0, 70.0)
    vibration = st.sidebar.slider("Vibration (mm/s)", 0.0, 10.0, 1.5)
    pressure = st.sidebar.slider("Pressure (PSI)", 80.0, 150.0, 110.0)
    last_service = st.sidebar.slider("Days Since Last Service", 1, 365, 100)
    
    data = {
        'Operating_Hours': operating_hours,
        'Temperature_C': temperature,
        'Vibration_mm_s': vibration,
        'Pressure_PSI': pressure,
        'Last_Service_Days': last_service
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Main Panel
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Sensor Input Values")
    st.dataframe(input_df)

    if model is not None and scaler is not None:
        # Preprocess Input
        input_scaled = scaler.transform(input_df)
        
        # Prediction
        prediction = model.predict(input_scaled)
        prediction_prob = model.predict_proba(input_scaled)[0][1]
        
        st.subheader("Prediction Result")
        
        # Gauge Chart for Probability
        st.metric(label="Failure Probability (Next 30 Days)", value=f"{prediction_prob:.2%}")
        
        # Logic for Alerts
        if prediction_prob > 0.5:
            st.error(f"⚠️ HIGH RISK: This machine is likely to fail! (Prob: {prediction_prob:.2%})")
            st.markdown("### 🛠️ Recommended Action")
            st.markdown("- **Schedule immediate maintenance.**")
            st.markdown("- **Contact Sales Team** to offer spare parts/service contract.")
            
            # Simple Sales Integration Mockup
            st.info("📧 Email draft generated for Sales Team: 'Client X Machine needs urgent service.'")
        else:
            st.success(f"✅ HEALTHY: Machine status is normal. (Prob: {prediction_prob:.2%})")
            st.markdown("### 📋 Status")
            st.markdown("- No immediate action required.")
            st.markdown("- Continue routine monitoring.")
            
    else:
        st.error("Model or Scaler not found. Please train the model first by running `src/model_trainer.py`.")

# Feature Importance Display (Optional)
with col2:
    st.subheader("Feature Importance Context")
    st.write("Current global feature importance for the model:")
    # Use absolute path for the image as well
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(current_dir, '../plots/feature_importance.png')
    
    if os.path.exists(plot_path):
        st.image(plot_path, caption='Model Feature Importance', use_column_width=True)
    else:
        st.info("Feature importance plot not found.")
