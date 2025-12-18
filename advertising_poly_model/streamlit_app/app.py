import streamlit as st
import pickle
import numpy as np
import os

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "advertising_poly_model.pkl")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# -------------------------------
# App UI
# -------------------------------
st.title("Advertising Sales Prediction App")
st.write("Enter advertising budget details")

tv = st.number_input("TV Advertising Budget", min_value=0.0)
radio = st.number_input("Radio Advertising Budget", min_value=0.0)
newspaper = st.number_input("Newspaper Advertising Budget", min_value=0.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Sales"):
    try:
        input_data = np.array([[tv, radio, newspaper]])
        prediction = model.predict(input_data)

        st.success(f"📈 Predicted Sales: {prediction[0]:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
