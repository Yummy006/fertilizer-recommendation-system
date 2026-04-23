import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Fertilizer Recommendation",
    page_icon="🌱",
    layout="centered"
)

# -------------------------------
# Load Model & Scaler
# -------------------------------
model = joblib.load("svm_model_original.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# Title
# -------------------------------
st.title("🌱 Fertilizer Recommendation System")
st.write("Enter soil and crop details below")

# -------------------------------
# Inputs
# -------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    soil_ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    nitrogen = st.number_input("Nitrogen", 0.0, 300.0, 80.0)
    temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0)

with col2:
    moisture = st.number_input("Soil Moisture (%)", 0.0, 100.0, 35.0)
    phosphorus = st.number_input("Phosphorus", 0.0, 300.0, 40.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)

with col3:
    potassium = st.number_input("Potassium", 0.0, 300.0, 60.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 5000.0, 1200.0)
    soil_type = st.selectbox("Soil Type", ['Clay', 'Silt', 'Sandy', 'Loamy'])

crop_type = st.selectbox("Crop Type", ['Cotton', 'Maize', 'Wheat', 'Potato', 'Rice', 'Sugarcane', 'Tomato'])
season = st.selectbox("Season", ['Kharif', 'Zaid', 'Rabi'])
irrigation = st.selectbox("Irrigation Type", ['Canal', 'Sprinkler', 'Rainfed', 'Drip'])

# -------------------------------
# Fertilizer Labels
# -------------------------------
fertilizer_labels = {
    0: "MOP",
    1: "Urea",
    2: "Zinc Sulphate",
    3: "Compost",
    4: "NPK",
    5: "DAP",
    6: "SSP"
}

# -------------------------------
# Training Columns (22 Features)
# -------------------------------
training_columns = [
    'Soil_pH', 'Soil_Moisture', 'Nitrogen_Level', 'Phosphorus_Level',
    'Potassium_Level', 'Temperature', 'Humidity', 'Rainfall',

    'Soil_Type_Loamy', 'Soil_Type_Sandy', 'Soil_Type_Silt',

    'Crop_Type_Maize', 'Crop_Type_Potato', 'Crop_Type_Rice',
    'Crop_Type_Sugarcane', 'Crop_Type_Tomato', 'Crop_Type_Wheat',

    'Season_Rabi', 'Season_Zaid',

    'Irrigation_Type_Drip', 'Irrigation_Type_Rainfed',
    'Irrigation_Type_Sprinkler'
]

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Fertilizer"):

    input_dict = {
        'Soil_pH': soil_ph,
        'Soil_Moisture': moisture,
        'Nitrogen_Level': nitrogen,
        'Phosphorus_Level': phosphorus,
        'Potassium_Level': potassium,
        'Temperature': temperature,
        'Humidity': humidity,
        'Rainfall': rainfall,
        'Soil_Type': soil_type,
        'Crop_Type': crop_type,
        'Season': season,
        'Irrigation_Type': irrigation
    }

    input_df = pd.DataFrame([input_dict])

    # Apply same encoding as training
    input_encoded = pd.get_dummies(
        input_df,
        columns=['Soil_Type', 'Crop_Type', 'Season', 'Irrigation_Type'],
        drop_first=True
    )

    # Match training columns
    input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_encoded)

    # Predict
    pred = model.predict(input_scaled)[0]

    # If numeric label
    fertilizer_name = fertilizer_labels.get(int(pred), pred)

    st.success(f"✅ Recommended Fertilizer: {fertilizer_name}")