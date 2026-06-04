import streamlit as st
import pickle
import numpy as np

# Load model and scaler
ridge_model = pickle.load(open('Models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('Models/scaler.pkl', 'rb'))

# st.title("Algerian Forest Fire Prediction")

# st.write("Enter the required features below:")

# # User Inputs
# Temperature = st.number_input("Temperature")
# RH = st.number_input("RH")
# Ws = st.number_input("Ws")
# Rain = st.number_input("Rain")
# FFMC = st.number_input("FFMC")
# DMC = st.number_input("DMC")
# ISI = st.number_input("ISI")
# Classes = st.number_input("Classes")
# Region = st.number_input("Region")

st.title("🔥 Algerian Forest Fire Prediction")

col1, col2 = st.columns(2)

with col1:
    Temperature = st.number_input( "Temperature", min_value=0, step=1 )
    RH = st.number_input( "Relative Humidity (RH)", min_value=0.0, format="%.2f")
    Ws = st.number_input( "Wind Speed (Ws)", min_value=0.0, format="%.2f")
    Rain = st.number_input( "Rain", min_value=0.0, format="%.2f" )

with col2:
    FFMC = st.number_input("FFMC", format="%.2f")
    DMC = st.number_input("DMC", format="%.2f")
    ISI = st.number_input("ISI", format="%.2f")
    Classes = st.number_input( "Classes", min_value=0, max_value=1 , step=1 )
    Region = st.number_input( "Region", min_value=0, max_value=1 , step=1 )

# Convert integer fields explicitly
Temperature = int(Temperature)
Classes = int(Classes)
Region = int(Region)

# Prediction Button
if st.button("Predict"):

    input_data = np.array([[ Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region ]])

    scaled_data = standard_scaler.transform(input_data)

    prediction = ridge_model.predict(scaled_data)

    st.success(f"Predicted FWI: {prediction[0]:.2f}")