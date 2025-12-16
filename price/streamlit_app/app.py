import streamlit as st
import pickle
import os
import numpy as np

# -------------------------------
# Load model safely
# -------------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "price_pred.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Mobile Price Prediction", layout="centered")

st.title("📱 Mobile Price Prediction App")
st.write("Enter the mobile specifications to predict the price.")

# Example input fields (modify according to your dataset)
ram = st.number_input("RAM (GB)", min_value=1, max_value=32, value=4)
storage = st.number_input("Storage (GB)", min_value=8, max_value=512, value=64)
battery = st.number_input("Battery (mAh)", min_value=1000, max_value=7000, value=4000)
camera = st.number_input("Camera (MP)", min_value=5, max_value=200, value=48)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):
    try:
        input_data = np.array([[ram, storage, battery, camera]])
        prediction = model.predict(input_data)

        st.success(f"💰 Predicted Mobile Price: ₹ {int(prediction[0])}")

    except Exception as e:
        st.error("Prediction failed. Check model input shape.")
        st.write(e)
