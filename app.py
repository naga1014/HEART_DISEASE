import streamlit as st
import numpy as np
import pickle 
from keras.models import load_model

# load model
model = load_model("model.h5")

# load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

st.title("❤️ Heart Disease Prediction App")

st.write("Enter Patient Details")

# input fields
age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0,1])
cp = st.number_input("Chest Pain Type", 0, 3)
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1])
restecg = st.number_input("Rest ECG", 0, 2)
thalach = st.number_input("Max Heart Rate")
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("Oldpeak")
slope = st.number_input("Slope", 0, 2)
ca = st.number_input("Number of Major Vessels", 0, 4)
thal = st.number_input("Thal", 0, 3)

# prediction button
if st.button("Predict"):

    input_data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                            thalach,exang,oldpeak,slope,ca,thal]])

    # scale input
    input_scaled = scaler.transform(input_data)

    # prediction
    prediction = model.predict(input_scaled)

    if prediction[0][0] > 0.5:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease")