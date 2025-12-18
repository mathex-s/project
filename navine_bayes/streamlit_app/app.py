import os
import pickle
import streamlit as st

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "naive_bayes_model.pkl")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


# -------------------------------
# App UI
# -------------------------------
st.title("Diabetes Prediction App")
st.write("Enter patient details to predict diabetes")

pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0, step=1)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    try:
        input_data = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]])

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("🟥 Result: Diabetic")
        else:
            st.success("🟩 Result: Non-Diabetic")

        # Probability (if available)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)
            st.write("Prediction Probability:")
            st.write(f"Non-Diabetic: {proba[0][0]:.2f}")
            st.write(f"Diabetic: {proba[0][1]:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
