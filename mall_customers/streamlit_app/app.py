import streamlit as st
import pickle
import numpy as np
import os

# Page config
st.set_page_config(page_title="Hierarchical Clustering", layout="centered")

st.title("🔗 Hierarchical Clustering App")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    # ✅ CORRECT FILE NAME HERE
    model_path = os.path.join(os.path.dirname(__file__), "hierarchical_model.pkl")
    
    with open(model_path, "rb") as file:
        model = pickle.load(file)
        
    return model

model = load_model()
st.success("Model loaded successfully!")

# ---------------- USER INPUT ----------------
st.subheader("Enter Input Features")

f1 = st.number_input("Feature 1", value=0.0)
f2 = st.number_input("Feature 2", value=0.0)
f3 = st.number_input("Feature 3", value=0.0)
f4 = st.number_input("Feature 4", value=0.0)

input_data = np.array([[f1, f2, f3, f4]])

# ---------------- PREDICTION ----------------
if st.button("Predict Cluster"):
    try:
        cluster = model.fit_predict(input_data)
        st.success(f"🧩 Predicted Cluster: {cluster[0]}")
    except Exception as e:
        st.error(f"Error: {e}")
