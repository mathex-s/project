import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(page_title="Hierarchical Clustering App", layout="centered")

st.title("🔗 Hierarchical Clustering Application")
st.write("Predict cluster for given input data")

# Load model
@st.cache_resource
def load_model():
    with open("hierarchical_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

st.success("Model loaded successfully!")

# ---------------- INPUT SECTION ----------------
st.subheader("Enter Feature Values")

# CHANGE number of inputs based on your dataset
f1 = st.number_input("Feature 1", value=0.0)
f2 = st.number_input("Feature 2", value=0.0)
f3 = st.number_input("Feature 3", value=0.0)
f4 = st.number_input("Feature 4", value=0.0)

# Convert to numpy array
input_data = np.array([[f1, f2, f3, f4]])

# ---------------- PREDICTION ----------------
if st.button("Predict Cluster"):
    try:
        cluster = model.fit_predict(input_data)
        st.success(f"🧩 Predicted Cluster: {cluster[0]}")
    except Exception as e:
        st.error(f"Error: {e}")
