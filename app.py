import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')

# Define the feature names as used during training
feature_names = ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
                 'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
                 'Warehouse_Processing_Time']

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Create input fields for each feature
Delivery_Distance = st.slider('Delivery Distance (km)', 0.0, 100.0, 50.0)
Traffic_Congestion = st.slider('Traffic Congestion (1-5)', 1, 5, 3)
Weather_Condition = st.slider('Weather Condition (1-5)', 1, 5, 3)
Delivery_Slot = st.slider('Delivery Slot (1-3)', 1, 3, 2)
Driver_Experience = st.slider('Driver Experience (years)', 0, 30, 10)
Num_Stops = st.slider('Number of Stops', 0, 15, 5)
Vehicle_Age = st.slider('Vehicle Age (years)', 0, 20, 5)
Road_Condition_Score = st.slider('Road Condition Score (1-5)', 1, 5, 3)
Package_Weight = st.slider('Package Weight (kg)', 0.0, 50.0, 25.0)
Fuel_Efficiency = st.slider('Fuel Efficiency (km/l)', 5.0, 30.0, 15.0)
Warehouse_Processing_Time = st.slider('Warehouse Processing Time (minutes)', 0, 120, 60)

# Create a DataFrame from user inputs
input_data = pd.DataFrame([{
    'Delivery_Distance': Delivery_Distance,
    'Traffic_Congestion': Traffic_Congestion,
    'Weather_Condition': Weather_Condition,
    'Delivery_Slot': Delivery_Slot,
    'Driver_Experience': Driver_Experience,
    'Num_Stops': Num_Stops,
    'Vehicle_Age': Vehicle_Age,
    'Road_Condition_Score': Road_Condition_Score,
    'Package_Weight': Package_Weight,
    'Fuel_Efficiency': Fuel_Efficiency,
    'Warehouse_Processing_Time': Warehouse_Processing_Time
}])

# Make prediction when button is clicked
if st.button('Predict Delivery Delay'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]

    if prediction[0] == 1:
        st.error(f'Prediction: Delivery is LIKELY to be Delayed (Probability: {prediction_proba[0]:.2f})')
    else:
        st.success(f'Prediction: Delivery is LIKELY to be On Time (Probability: {1 - prediction_proba[0]:.2f})')

# To make this a runnable app.py, save the content to a file.
# The actual saving to file will be done in the next step to avoid issues.
