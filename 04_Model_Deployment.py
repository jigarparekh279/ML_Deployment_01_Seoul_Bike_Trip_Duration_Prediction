import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import joblib
import os

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

import streamlit as st

########################################################################

# Capture the path on current folder on cloud
curr_path = os.path.dirname(os.path.realpath(__file__))

feat_cols = ['Distance', 'PLong', 'DLatd', 'Haversine', 
             'Phour', 'Pmin', 'Dhour', 'Dmin', 'Temp', 'Solar']

scalar = joblib.load('models/scalar.joblib')
ann = keras.models.load_model('models/ANN.h5')
                                                            
def predict_duration(attributes: np.ndarray):
    # Retrun bike trip duration value
    scaled_attributes = scalar.transform(attributes)
    print(scaled_attributes)
    pred = ann.predict(scaled_attributes)
    return int(pred[0,0])
                              
########################################################################                              
                              
st.set_page_config(page_title="Seoul Bike Trip Duration Prediction App",
                   page_icon="🛴", layout="wide")

with st.form("prediction_form"):

    st.header("Enter the Deciding Factors:")

    distance  = st.number_input("Distance: ", value=0, format="%d")
    haversine = st.number_input("Haversine: ")
    plong = st.number_input("Pickup Longitude: ")
    dlatd = st.number_input("Dropoff Latitude: ")
    phour = st.slider("Pickup Hour: ", 0, 23, value=0, format="%d")
    pmin  = st.slider("Pickup Minute: ", 0, 59, value=0, format="%d")
    dhour = st.slider("Dropoff Hour: ", 0, 23, value=0, format="%d")
    dmin  = st.slider("Dropoff Minute: ", 0, 59, value=0, format="%d")
    temp  = st.number_input("Temp: ")
    solar = st.number_input("Solar: ")

    submit_val = st.form_submit_button("Predict Duration")

if submit_val:
    # If submit is pressed == True
    attributes = np.array([distance, plong, dlatd, haversine, phour,
                        pmin, dhour, dmin, temp, solar]).reshape(1,-1)

    if attributes.shape == (1,10):
        print("Attributes are valid")
        
        value = predict_duration(attributes=attributes)

        st.header("Here are the results:")
        st.success(f"The Duration predicted is {value} mins")