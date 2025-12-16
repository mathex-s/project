import streamlit as st
import pickle
import numpy as np

# Load model
with open("price_pred.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📱 Mobile Price Prediction")

# Inputs (MUST match training features & order)
ram = st.number_input("RAM (GB)", min_value=1, max_value=32, step=1)
storage = st.number_input("Storage (GB)", min_value=8, max_value=512, step=8)
camera = st.number_input("Camera (MP)", min_value=5, max_value=200, step=1)
battery = st.number_input("Battery (mAh)", min_value=1000, max_value=7000, step=100)

if st.button("Predict Price"):
    # ✅ VERY IMPORTANT: 2D array
    input_data = np.array([[ram, storage, camera, battery]])

    # Debug check
    st.write("Input shape:", input_data.shape)
    st.write("Model expects features:", model.n_features_in_)

    prediction = model.predict(input_data)
    st.success(f"💰 Predicted Price: ₹ {int(prediction[0])}")
