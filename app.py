import streamlit as st
from login import login_user
from utils.map_utils import display_route_map
from utils.prediction_utils import predict_traffic
import pandas as pd
import sqlite3
import os

# Title
st.set_page_config(page_title="Smart Traffic System - Gweru", layout="wide")
st.title("🚦 Smart Traffic Intelligence System – Gweru")

# User login
user = login_user()
if not user:
    st.stop()

# Load data
data_path = os.path.join("data", "simulated_traffic_data.csv")
df = pd.read_csv(data_path, parse_dates=['timestamp'])

# Sidebar inputs
st.sidebar.header("Select Route")
origin = st.sidebar.selectbox("Origin", df['location'].unique())
destination = st.sidebar.selectbox("Destination", df['location'].unique())

# Prediction logic
if st.sidebar.button("Predict Route"):
    # Store prediction in session state to keep it after rerun
    prediction = predict_traffic(df, origin, destination)
    st.session_state.prediction = prediction  # Store prediction in session state
    
    # Save to history
    conn = sqlite3.connect("history/user_history.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS history (
        username TEXT, origin TEXT, destination TEXT,
        congestion TEXT, accident_risk TEXT, fuel TEXT, timestamp TEXT
    )""")
    conn.execute("INSERT INTO history VALUES (?, ?, ?, ?, ?, ?, ?)", (
        user, origin, destination,
        prediction['congestion_level'], prediction['accident_risk'],
        prediction['fuel_consumption'], str(pd.Timestamp.now())
    ))
    conn.commit()
    conn.close()

# Display prediction if it exists in session state
if 'prediction' in st.session_state:
    st.subheader("Prediction Results")
    prediction = st.session_state.prediction
    st.write(prediction)
    display_route_map(origin, destination, prediction)

# Show prediction history
if st.sidebar.button("View History"):
    st.subheader("Prediction History")
    conn = sqlite3.connect("history/user_history.db")
    rows = conn.execute("SELECT * FROM history WHERE username = ?", (user,)).fetchall()
    conn.close()
    st.dataframe(pd.DataFrame(rows, columns=[
        "Username", "Origin", "Destination", "Congestion", "Accident Risk", "Fuel", "Timestamp"
    ]))
