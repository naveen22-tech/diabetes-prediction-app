import streamlit as st
import numpy as np
import pickle
import os

# Load model and scaler
model = pickle.load(open("model/diabetes_model.pkl", "rb"))
scaler_path = "model/scaler.pkl"
scaler = None
if os.path.exists(scaler_path):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
else:
    scaler = None

st.title("🩺 Diabetes Prediction System")

st.write("Enter patient details below:")

# Input fields
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.number_input("Age", 1, 120)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    # Require scaler
    if scaler is None:
        st.error("Scaler not found. Run `python train.py` to create model/scaler.pkl before predicting.")
    else:
        # Scale
        input_data = scaler.transform(input_data)

        # Predict
        result = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if result == 1:
            st.error(f"⚠️ Diabetic (Risk: {prob:.2f})")
        else:
            st.success(f"✅ Non-Diabetic (Risk: {prob:.2f})")