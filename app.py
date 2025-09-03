import streamlit as st
import numpy as np
import joblib

# Title
st.title("Crop Recommendation System")

# Load saved model
model = joblib.load("crop_model.joblib")

# User inputs with sliders
N = st.slider('Nitrogen (N)', 0, 140, 20)
P = st.slider('Phosphorous (P)', 0, 140, 20)
K = st.slider('Potassium (K)', 0, 500, 30)
temperature = st.slider('Temperature (°C)', 0.0, 50.0, 25.0)
humidity = st.slider('Humidity (%)', 0.0, 100.0, 50.0)
ph = st.slider('pH', 0.0, 14.0, 7.0)
rainfall = st.slider('Rainfall (mm)', 0.0, 500.0, 50.0)

# Predict button
if st.button('Recommend Crop'):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)[0]
    st.success(f"Recommended Crop: {prediction}")
