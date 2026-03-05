import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler , OneHotEncoder
import pickle

model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl' , 'rb') as f:
    scaler = pickle.load(f)

with open('one_hot_encoder.pkl' , 'rb')  as f:
    onehot_encoder=pickle.load(f)


st.title("Customer Churn Prediction")
creditscore = st.number_input("Enter Credit Score", min_value=300, max_value=850, value=600)
geography = st.selectbox("Select Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Select Gender", ["Male", "Female"])
age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Enter Tenure", min_value=0, max_value=10, value=3)
balance = st.number_input("Enter Balance", min_value=0.0, value=1000.0)
num_of_products = st.number_input("Enter Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary = st.number_input("Enter Estimated Salary", min_value=0.0, value=50000.0)

input_data = pd.DataFrame({
    'CreditScore': [creditscore],
    'Geography': [geography],
    'Gender': [1 if gender=="Male" else 0],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == "Yes" else 0],
    'IsActiveMember': [1 if is_active_member == "Yes" else 0],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder.transform(input_data[['Geography']]).toarray()
input_data = pd.concat([input_data, pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))], axis=1).drop(columns=['Geography'], axis=1)

scaler_input = scaler.transform(input_data)
prediction_probability = model.predict(scaler_input)[0][0]
if prediction_probability > 0.5:
    st.write(f"The customer is likely to churn with a probability of {prediction_probability:.2f}")
else:
    st.write(f"The customer is unlikely to churn with a probability of {prediction_probability:.2f}")

