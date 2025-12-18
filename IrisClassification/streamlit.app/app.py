import numpy as np
import os
import pickle
import streamlit as st

@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "svm_iris_model.pkl")

    with open(MODEL_PATH, "rb") as file:
        obj = pickle.load(file)

    if isinstance(obj, tuple):
        model = obj[0]
        scaler = obj[1] if len(obj) > 1 else None
    else:
        model = obj
        scaler = None

    return model, scaler

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Iris Flower Prediction 🌸")
st.write("Enter flower measurements")

# 🌼 Iris dataset feature names
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0)
sepal_width  = st.number_input("Sepal Width (cm)", min_value=0.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0)
petal_width  = st.number_input("Petal Width (cm)", min_value=0.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    try:
        input_data = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])

        # Apply scaler if available
        if scaler is not None:
            input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)[0]

        # Iris class labels
        class_names = {
            0: "Setosa",
            1: "Versicolor",
            2: "Virginica"
        }

        st.success(f"🌸 Predicted Flower: **{class_names.get(prediction, prediction)}**")

    except Exception as e:
        st.error(f"Prediction error: {e}")
