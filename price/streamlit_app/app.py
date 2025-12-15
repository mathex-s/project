import streamlit as st
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Product Defect Detection",
    page_icon="🔍",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    with open("price_pred (1).pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Title
st.title("🔍 Product Defect Detection")
st.write("Predict whether a product is **Defective** or **Not Defective**")

st.divider()

# Input fields
st.subheader("Enter Product Details")

feature1 = st.number_input("Feature 1", min_value=0.0, step=0.1)
feature2 = st.number_input("Feature 2", min_value=0.0, step=0.1)
feature3 = st.number_input("Feature 3", min_value=0.0, step=0.1)

# Predict button
if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(input_data)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("❌ Defective Product")
    else:
        st.success("✅ Non-Defective Product")

st.divider()

st.caption("Machine Learning based Product Defect Detection")
