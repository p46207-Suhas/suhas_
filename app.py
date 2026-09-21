!pip install streamlit
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('delivery_delay.sav')

# Define the feature names in the correct order as used during training
feature_names = [
    'Delivery_Distance',
    'Traffic_Congestion',
    'Weather_Condition',
    'Delivery_Slot',
    'Driver_Experience',
    'Num_Stops',
    'Vehicle_Age',
    'Road_Condition_Score',
    'Package_Weight',
    'Fuel_Efficiency',
    'Warehouse_Processing_Time'
]

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Create input fields for each feature
delivery_distance = st.slider('Delivery Distance', 0.0, 100.0, 50.0)
traffic_congestion = st.selectbox('Traffic Congestion', [1, 2, 3, 4, 5])
weather_condition = st.selectbox('Weather Condition', [1, 2, 3])
delivery_slot = st.selectbox('Delivery Slot', [1, 2, 3])
driver_experience = st.slider('Driver Experience (years)', 0, 30, 10)
num_stops = st.slider('Number of Stops', 0, 10, 5)
vehicle_age = st.slider('Vehicle Age (years)', 0, 20, 5)
road_condition_score = st.selectbox('Road Condition Score', [1, 2, 3, 4, 5])
package_weight = st.slider('Package Weight (kg)', 0.0, 50.0, 10.0)
fuel_efficiency = st.slider('Fuel Efficiency (km/l)', 0.0, 30.0, 15.0)
warehouse_processing_time = st.slider('Warehouse Processing Time (minutes)', 0, 100, 50)

# Create a button to make predictions
if st.button('Predict Delivery Delay'):
    # Create a DataFrame from the input values, ensuring correct order and column names
    input_data = pd.DataFrame([[delivery_distance, traffic_congestion, weather_condition,
                                delivery_slot, driver_experience, num_stops, vehicle_age,
                                road_condition_score, package_weight, fuel_efficiency,
                                warehouse_processing_time]], 
                               columns=feature_names)
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f'**Delivery Delay Predicted!** (Probability: {prediction_proba[0][1]:.2f})')
    else:
        st.success(f'**No Delivery Delay Predicted.** (Probability: {prediction_proba[0][0]:.2f})')
        
st.markdown(
    """
    ### How to run this Streamlit app:
    1.  Save the code above as `app.py` in the same directory as `delivery_delay.sav`.
    2.  Open your terminal or command prompt.
    3.  Navigate to that directory.
    4.  Install Streamlit if you haven't already: `pip install streamlit`
    5.  Run the app: `streamlit run app.py`
    """
)
