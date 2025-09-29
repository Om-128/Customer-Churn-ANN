import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

#Load the trained model
model = load_model('model.h5')

#Import one hot encoder for geography
with open('one_hot_encoder_geo.pkl', 'rb') as f:
    one_hot_encoder_geo = pickle.load(f)

#Import label encoder for gender
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

#Import standard scaler
with open('scaler.pkl', 'rb') as f:
    standard_scaler = pickle.load(f)


#Streamlit app
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn probability.")

#Input fields
geography = st.selectbox("Geography", one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenture = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', ['0', '1'])
is_active_member = st.selectbox('Is Active Member', ['1', '0'])


#Submit button
if st.button("Submit"):

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenture],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [int(has_cr_card)],
        'IsActiveMember': [int(is_active_member)],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

    #Compine with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

    #Scale the input data
    input_data_scaled = standard_scaler.transform(input_data)

    #predict churn
    prediction = model.predict(input_data_scaled)
    churn_probability = prediction[0][0]

    st.write("Churn probability", churn_probability)

    if prediction >= 0.5:
        st.error(f"The customer is likely to churn.")
    else:
        st.success(f"The customer is unlikely to churn.")