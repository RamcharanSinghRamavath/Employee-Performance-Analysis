# x order -->> "years at company, monthly salry, overtime hours, promotions"
## scaler is exported as scaler.pkl
#model is exportd as model.pkl
#scaler is imported as model.pkl
import streamlit as st

st.title("Connection Test")
st.write("If you see this, the app is running correctly.")

import joblib 
import numpy as np
scaler = joblib.load("Scaler.pkl")
model = joblib.load("model.pkl")

st.title("Employee Performance Predictions ")
st.divider()

st.write("you can get a performance estimatiojn for employee after entering the values and pressing ")
st.divider()
years = st.number_input("Enter the Years at company  ", min_value = 0, max_value= 15, value = 2)
salary = st.number_input("Enter the monthly Salary ", min_value =  1000, max_value = 100000,value = 5000 )
overtime = st.number_input("Enter the overtime hours ", min_value = 0, max_value = 100, value  = 0)
promotions = st.number_input("Enter the , Promotions", min_value=0, max_value=10, value= 0 )
satisfaction  =  st.number_input("enter  the employeee satisfaction ", min_value= 1.0, max_value=10.0, value=2.0)
x = [years, salary, overtime, promotions, satisfaction]

st.divider()
predictionbutton = st.button("Predict the Performance Score ")
st.divider()
if predictionbutton:

    x1 = np.array(x)
    x_array = scaler.transform([x1])
    prediction = model.predict(x_array)[0]

    st.balloons()
    st.write(f"Prediction for the Performance Score is{prediction}")



else:
    st.write("Please enter the button for Employeee Performance Prediction")