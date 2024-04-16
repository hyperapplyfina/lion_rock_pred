import streamlit as st
import numpy as np
import pandas as pd
import datetime
import pickle
from sklearn import ensemble

@st.cache_data
def load_model():
    speed_model = pickle.load(open('speed_model.sav', 'rb'))
    volume_model = pickle.load(open('volume_model.sav', 'rb'))
    occupancy_model = pickle.load(open('occupancy_model.sav', 'rb'))
    return speed_model, volume_model, occupancy_model
    
speed_model, volume_model, occupancy_model = load_model()

d = st.date_input("date", datetime.date(2024, 4, 20))

direction = st.radio("travelling direction",
    ["North Bound", "South Bound"])

if (direction == "North Bound"):
    did= 2
else:
    did = 1
    
lane = st.radio("lane",
        ["Slow", "Middle", "Fast"])

if (lane == "Slow"):
    lane_= 0
elif (lane == "Middle"):
    lane_= 1
else:
    lane_= 2


holiday = st.radio("is the day a holiday?",
    ["Yes", "No"])

if (holiday == "Yes"):
    hol= 1
else:
    hol = 0

rainfall = st.slider("Past 60-Minutes Rainfall in mm", min_value = 0.0, max_value = 95.3, value = 0.334)
humidity = st.slider("Relative Humidity in %", min_value = 20.6, max_value = 98.0, value = 77.48)
temp = st.slider("Air Temperature in degree C", min_value = 7.5, max_value = 36.6, value = 24.72)
windspeed = st.slider("Wind Speed in km/hr", min_value = 0.0, max_value = 25.9, value = 5.17)

datetime_obj = datetime.datetime.combine(d, datetime.datetime.min.time())

week_day = datetime_obj.weekday()
mon = d.month

#detector_id

df = pd.DataFrame(columns=['detector_id', 'lane_id', 'Hour', 'Month', 'Weekday', 'is_holiday',
       'Past 60-Minutes Rainfall in mm', 'Relative Humidity in %',
       'Air Temperature in degree C', 'Wind Speed in km/hr'])


for i in range(0, 24):
    df = df.append(pd.Series([did, lane_, i, mon,week_day, hol, rainfall, humidity, temp, windspeed]
, index=df.columns), ignore_index=True)

speed_pred = speed_model.predict(df)
volume_pred = volume_model.predict(df)
occupancy_pred = occupancy_model.predict(df)


pred_df = pd.DataFrame({'speed': speed_pred, 'volume': volume_pred, 'occupancy': occupancy_pred})


st.line_chart(pred_df)

#st.line_chart(data=None, *, x=None, y=None, color=None, width=0, height=0, use_container_width=True)
