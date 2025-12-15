import streamlit as st
import pickle
import os
import numpy as np

st.set_page_config(page_title="Product Defect Detection", layout="centered")

st.title("🔍 Product Defect Detection App")

# ---------------- LOAD MODEL SAFELY ---------------- #
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "defect_pre.pkl")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ---------------- USER INPUT ---------------- #
st.subheader("Enter Product Parameters")

feature1 = st.number_input("Feature 1", min_value=0.0)
feature2 = st.number_input("Feature 2", min_value=0.0)
feature3 = st.number_input("Feature 3", min_value=0.0)

# ---------------- PREDICTION ---------------- #
if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("❌ Defective Product")
    else:
        st.success("✅ Non-Defective Product")
