import pandas as pd
import streamlit as st
import os
import joblib

st.set_page_config(page_title='Covid Risk Predictor',layout='centered')

# loading model,scaler and columns
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, '..', 'Pickel_files', 'logistic_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, '..', 'Pickel_files', 'scaler.pkl'))
columns = joblib.load(os.path.join(BASE_DIR, '..', 'Pickel_files', 'columns.pkl'))

# Header
st.title("🩺 COVID-19 Risk Prediction System")
st.markdown(
    "This tool predicts COVID risk based on symptoms and clinical indicators."
)

st.divider()

# Patient info
st.subheader('👤Patient Information')

age = st.number_input('Age',min_value=1,max_value=100,value=30)
gender = st.selectbox('Gender',['Male','Female'])
gender = 1 if gender == 'Male' else 0


# Symptoms
st.subheader("😷 Symptoms")

fever = st.checkbox("Fever")
dry_cough = st.checkbox("Dry Cough")
fatigue = st.checkbox("Fatigue")
headache = st.checkbox("Headache")
sore_throat = st.checkbox("sore throat")
shortness_of_breath = st.checkbox("Shortness of Breath")
loss_of_smell = st.checkbox("Loss of Smell")
loss_of_taste = st.checkbox("Loss of Taste")
chest_pain = st.checkbox("Chest Pain")
asthma = st.checkbox("Asthma")
diabetes = st.checkbox("Diabetes")
heart_disease = st.checkbox("Heart Disease")

# Clinical Measurement
st.subheader("🫁 Clinical Measurements")

oxygen = st.number_input("Oxygen Level (%)", min_value=70, max_value=100, value=98)
temperature = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=36.8)

# Binary conversion
low_oxygen = 1 if oxygen < 94 else 0
fever_temp = 1 if temperature > 37.5 else 0

# connection history
st.subheader("🌍 Connection History")

travel_history = st.checkbox("Recent Travel History")
contact_with_patient = st.checkbox("Contact with COVID Patient")

# Prediction

if st.button('Predict COVID Risk'):
    input_df=pd.DataFrame({
        'age':[age],
        'gender':[gender],
        'fever': [int(fever)],
        'dry_cough': [int(dry_cough)],
        'sore_throat': [int(sore_throat)],
        'fatigue': [int(fatigue)],
        'headache': [int(headache)],
        'shortness_of_breath': [int(shortness_of_breath)],
        'loss_of_smell': [int(loss_of_smell)],
        'loss_of_taste': [int(loss_of_taste)],
        'chest_pain': [int(chest_pain)],
        'asthma': [int(asthma)],
        'diabetes': [int(diabetes)],
        'heart_disease': [int(heart_disease)],
        'travel_history': [int(travel_history)],
        'contact_with_patient': [int(contact_with_patient)],
        'low_oxygen': [low_oxygen],
        'fever_temp': [fever_temp]
        })
    
    # Scale age
    input_df[['age']] = scaler.transform(input_df[['age']])

    # Enforce training column order
    input_df = input_df[columns]

    #Prediction
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.divider()

    if pred == 1:
        st.error(f"⚠️ COVID POSITIVE RISK\n\nProbability: {proba:.2%}")
    else:
        st.success(f"✅ LOW COVID RISK\n\nProbability: {proba:.2%}")
