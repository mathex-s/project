import streamlit as st
import pickle
import numpy as np
import os

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Price Prediction", page_icon="📊")

st.title("📱 Price Prediction App")

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "price_pred.pkl")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

st.success("Model loaded successfully ✅")

# -----------------------------
# User Input Section
# -----------------------------
st.subheader("Enter Input Values")

# 👉 CHANGE these inputs IF your model uses different features
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3, feature4]])
    
    prediction = model.predict(input_data)

    st.success(f"🔮 Predicted Result: **{prediction[0]}**")
