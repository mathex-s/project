import streamlit as st
import numpy as np
import joblib
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Hierarchical Clustering App",
    layout="centered"
)

st.title("🔗 Hierarchical Clustering App")
st.write("Mall Customers Segmentation using Hierarchical Clustering")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "hierarchical_model (1).pkl"   # ✅ exact file name
    )
    model = joblib.load(model_path)
    return model

model = load_model()
st.success("Model loaded successfully!")

# ---------------- INPUT SECTION ----------------
st.subheader("Enter Customer Details")

age = st.number_input(
    "Age (Years)",
    min_value=0,
    max_value=100,
    value=30
)

annual_income = st.number_input(
    "Annual Income (k$)",
    min_value=0,
    max_value=200,
    value=60
)

spending_score = st.number_input(
    "Spending Score (1–100)",
    min_value=1,
    max_value=100,
    value=50
)

gender = st.selectbox(
    "Gender",
    options=["Male", "Female"]
)

# Encode gender (use ONLY if gender was used during training)
gender_encoded = 1 if gender == "Male" else 0

# ---------------- FEATURE ORDER ----------------
# IMPORTANT: Must match training order
X = np.array([[age, annual_income, spending_score, gender_encoded]])

# ---------------- PREDICTION ----------------
if st.button("Predict Customer Cluster"):
    try:
        # Hierarchical clustering requires at least 2 samples
        X = np.array([
            [age, annual_income, spending_score, gender_encoded],
            [age + 1, annual_income + 1, spending_score + 1, gender_encoded]
        ])

        cluster = model.fit_predict(X)

        st.success(f"🧩 Customer belongs to Cluster: {int(cluster[0])}")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed using Streamlit & Scikit-Learn")
