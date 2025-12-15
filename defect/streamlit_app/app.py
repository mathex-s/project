import streamlit as st
import pickle
import os
import numpy as np

st.set_page_config(page_title="Product Defect Detection", layout="centered")

st.title("🔍 Product Defect Detection App")

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "defect_pre.pkl")

    # DEBUG: show files in directory
    if not os.path.exists(model_path):
        st.error("❌ defect_pre.pkl NOT FOUND")
        st.write("📂 Files in current directory:")
        st.write(os.listdir(base_dir))
        st.stop()

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
