import streamlit as st
import pickle
import numpy as np

# Load model and scaler
ridge_model = pickle.load(open('20-Project1/Models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('20-Project1/Models/scaler.pkl', 'rb'))

st.title("Algerian Forest Fire Prediction")

st.write("Enter the required features below:")

# User Inputs
Temperature = st.number_input("Temperature")
RH = st.number_input("RH")
Ws = st.number_input("Ws")
Rain = st.number_input("Rain")
FFMC = st.number_input("FFMC")
DMC = st.number_input("DMC")
ISI = st.number_input("ISI")
Classes = st.number_input("Classes")
Region = st.number_input("Region")

# Prediction Button
if st.button("Predict"):

    input_data = np.array([[
        Temperature,
        RH,
        Ws,
        Rain,
        FFMC,
        DMC,
        ISI,
        Classes,
        Region
    ]])

    scaled_data = standard_scaler.transform(input_data)

    prediction = ridge_model.predict(scaled_data)

    st.success(f"Predicted FWI: {prediction[0]:.2f}")