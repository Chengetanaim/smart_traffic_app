import streamlit as st
from login import login_user
import pandas as pd
import sqlite3
import os
import folium
from streamlit_folium import st_folium
import numpy as np
import random
import datetime

def predict_traffic(df, origin, destination, selected_time=None):
    """
    Predict traffic conditions based on origin, destination, and time.
    
    Parameters:
    - df: DataFrame containing traffic data
    - origin: Starting location
    - destination: Ending location
    - selected_time: Optional time for prediction (if None, uses current time)
    
    Returns:
    - Dictionary with prediction results
    """
    # If no time provided, use current time
    if selected_time is None:
        now = datetime.datetime.now()
        current_hour = now.hour
    else:
        # Extract hour from provided datetime
        current_hour = selected_time.hour
    
    # Filter data for the specified hour
    hourly_data = df[df['timestamp'].dt.hour == current_hour]
    
    # Get the most recent data for origin and destination
    origin_data = hourly_data[hourly_data['location'] == origin]
    destination_data = hourly_data[hourly_data['location'] == destination]
    
    # If no data available for the specific hour, fallback to the entire dataset
    if origin_data.empty:
        origin_data = df[df['location'] == origin]
    if destination_data.empty:
        destination_data = df[df['location'] == destination]
    
    # Calculate average metrics between origin and destination
    if not origin_data.empty and not destination_data.empty:
        # Get the most recent entries for each location
        origin_recent = origin_data.sort_values(by="timestamp").iloc[-1]
        dest_recent = destination_data.sort_values(by="timestamp").iloc[-1]
        
        # Average the metrics (weighted more toward origin as that's where traffic starts)
        congestion_level = round((origin_recent['congestion_level'] * 0.7 + 
                                 dest_recent['congestion_level'] * 0.3), 1)
        
        traffic_volume = round((origin_recent.get('traffic_volume', 1000) * 0.7 + 
                              dest_recent.get('traffic_volume', 1000) * 0.3))
        
        accident_prob = (origin_recent['accident_probability'] * 0.7 + 
                         dest_recent['accident_probability'] * 0.3)
        
        fuel_consumption = (origin_recent['fuel_consumption_l_per_100km'] * 0.6 + 
                           dest_recent['fuel_consumption_l_per_100km'] * 0.4)
        
        # Convert numerical congestion to descriptive labels
        if congestion_level <= 3:
            congestion_label = "low"
        elif congestion_level <= 6:
            congestion_label = "medium"
        else:
            congestion_label = "high"
        
        return {
            "congestion_level": congestion_label,
            "congestion_value": congestion_level,
            "accident_risk": f"{accident_prob*100:.1f}%",
            "fuel_consumption": f"{fuel_consumption:.2f} L/100km",
            "origin_congestion": origin_recent['congestion_level'],
            "destination_congestion": dest_recent['congestion_level'],
            "time_of_day": current_hour
        }
    else:
        # Fallback if no data available
        return {
            "congestion_level": "unknown",
            "congestion_value": 0,
            "accident_risk": "N/A",
            "fuel_consumption": "N/A",
            "time_of_day": current_hour
        }


# Initialize session state variables if they don't exist
if 'map_displayed' not in st.session_state:
    st.session_state.map_displayed = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'current_origin' not in st.session_state:
    st.session_state.current_origin = None
if 'current_destination' not in st.session_state:
    st.session_state.current_destination = None
if 'current_time' not in st.session_state:
    st.session_state.current_time = None
if 'view_history' not in st.session_state:
    st.session_state.view_history = False
if 'history_data' not in st.session_state:
    st.session_state.history_data = None

# Dictionary of location coordinates in Harare
location_coords = {
    "CBD": [-17.8292, 31.0522],  # Central Harare
    "Avondale": [-17.8003, 31.0353],
    "Borrowdale": [-17.7425, 31.1039],
    "Highlands": [-17.8028, 31.0819],
    "Mbare": [-17.8656, 31.0361],
    "Mount Pleasant": [-17.7686, 31.0353],
    "Westgate": [-17.8056, 30.9944]
}

def generate_intermediate_points(start, end, num_points=3, variance=0.003):
    """Generate realistic route with intermediate points between start and end"""
    # Create direct line
    lat_diff = end[0] - start[0]
    lng_diff = end[1] - start[1]
    
    points = []
    for i in range(1, num_points + 1):
        # Calculate position along direct line
        frac = i / (num_points + 1)
        mid_lat = start[0] + lat_diff * frac
        mid_lng = start[1] + lng_diff * frac
        
        # Add some randomness to simulate roads
        # Perpendicular deviation for realistic roads
        angle = np.arctan2(lat_diff, lng_diff) + np.pi/2
        deviation = random.uniform(-variance, variance)
        mid_lat += deviation * np.sin(angle)
        mid_lng += deviation * np.cos(angle)
        
        points.append([mid_lat, mid_lng])
    
    return points

def display_route_map(origin, destination, prediction):
    """Display a map with a realistic route between origin and destination"""
    # Create a map centered between the two points
    center_lat = (location_coords[origin][0] + location_coords[destination][0]) / 2
    center_lng = (location_coords[origin][1] + location_coords[destination][1]) / 2
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13)
    
    # Add markers for origin and destination
    folium.Marker(
        location_coords[origin], 
        tooltip=f"Origin: {origin}", 
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)
    
    folium.Marker(
        location_coords[destination], 
        tooltip=f"Destination: {destination}", 
        icon=folium.Icon(color="red", icon="flag")
    ).add_to(m)
    
    # Generate realistic route with intermediate points
    # Use a deterministic seed based on origin and destination to always get the same route
    random.seed(f"{origin}_{destination}")
    intermediate_points = generate_intermediate_points(
        location_coords[origin], 
        location_coords[destination],
        num_points=4
    )
    # Reset the random seed
    random.seed()
    
    # Create full route
    route_points = [location_coords[origin]] + intermediate_points + [location_coords[destination]]
    
    # Determine line color based on congestion level
    congestion = prediction.get('congestion_level', 'medium')
    if congestion == 'low':
        line_color = 'green'
    elif congestion == 'medium':
        line_color = 'orange'
    else:  # high
        line_color = 'red'
    
    # Add the route line
    folium.PolyLine(
        route_points, 
        color=line_color, 
        weight=4, 
        opacity=0.8, 
        tooltip=f"Congestion: {congestion}"
    ).add_to(m)
    
    # Add congestion indicators at intermediate points
    for i, point in enumerate(intermediate_points):
        if i % 2 == 0:  # Add indicators at some points to avoid cluttering
            folium.CircleMarker(
                location=point,
                radius=5,
                color=line_color,
                fill=True,
                fill_opacity=0.7,
                tooltip=f"Traffic: {congestion} congestion"
            ).add_to(m)
    
    # Display a popup with the traffic prediction
    time_info = f"Time: {prediction.get('time_of_day', 'N/A')}:00" if 'time_of_day' in prediction else ""
    
    html = f"""
        <div style="width: 200px; font-family: Arial;">
            <h4>Traffic Prediction</h4>
            <b>Route:</b> {origin} to {destination}<br>
            <b>{time_info}</b><br>
            <b>Congestion:</b> {prediction.get('congestion_level', 'N/A')}<br>
            <b>Traffic Volume:</b> {prediction.get('congestion_value', 'N/A')} vehicles/hr<br>
            <b>Accident Risk:</b> {prediction.get('accident_risk', 'N/A')}<br>
            <b>Fuel Consumption:</b> {prediction.get('fuel_consumption', 'N/A')}<br>
        </div>
    """
    
    # Add popup at the middle of the route
    mid_point = intermediate_points[len(intermediate_points)//2]
    folium.Popup(html, max_width=300).add_to(
        folium.Marker(mid_point, icon=folium.Icon(color="blue", icon="info-sign"))
    )
    
    # Display the map in Streamlit with a fixed key to prevent rerendering
    return st_folium(m, width=700, height=500, key=f"map_{origin}_{destination}")

# Function to get user history
@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_user_history(username):
    conn = sqlite3.connect("history/user_history.db")
    rows = conn.execute("SELECT * FROM history WHERE username = ?", (username,)).fetchall()
    conn.close()
    
    if rows:
        return pd.DataFrame(rows, columns=[
            "Username", "Origin", "Destination", "Congestion", "Accident Risk", 
            "Fuel", "Selected Time", "Prediction Time"
        ])
    return None

# Set page config
st.set_page_config(page_title="Smart Traffic System - Harare", layout="wide")
st.title("Traffic Flow Modelling")

# User login
user = login_user()
if not user:
    st.stop()

# Load data
data_path = os.path.join("data", "simulated_traffic_data.csv")
df = pd.read_csv(data_path, parse_dates=['timestamp'])

# Create sidebar
with st.sidebar:
    st.header("Select Route")
    
    # Route selection
    origin = st.selectbox("Origin", sorted(list(location_coords.keys())), key='origin_select')
    destination = st.selectbox("Destination", sorted(list(location_coords.keys())), key='destination_select')
    
    # Time selection
    st.subheader("Select Time")
    use_current_time = st.checkbox("Use Current Time", value=True)
    
    if use_current_time:
        selected_time = datetime.datetime.now()
        st.info(f"Current time: {selected_time.strftime('%H:%M')}")
    else:
        hour = st.slider("Hour", 0, 23, datetime.datetime.now().hour)
        minute = st.slider("Minute", 0, 59, 0, step=15)
        selected_time = datetime.datetime.now().replace(hour=hour, minute=minute)
        st.info(f"Selected time: {selected_time.strftime('%H:%M')}")
    
    # Prevent same origin and destination
    if origin == destination:
        st.warning("Please select different locations for origin and destination.")
    
    # Prediction button
    predict_pressed = st.button("Predict Route", key="predict_button")
    
    # History toggle
    history_pressed = st.button("View History", key="history_button")
    if history_pressed:
        st.session_state.view_history = not st.session_state.view_history
        if st.session_state.view_history:
            st.session_state.history_data = get_user_history(user)

# Handle prediction logic
if predict_pressed and origin != destination:
    # Store prediction in session state
    prediction = predict_traffic(df, origin, destination, selected_time)
    st.session_state.prediction = prediction
    st.session_state.current_origin = origin
    st.session_state.current_destination = destination
    st.session_state.current_time = selected_time
    st.session_state.map_displayed = None  # Force map refresh
    st.session_state.view_history = False  # Close history view
    
    # Save to history
    conn = sqlite3.connect("history/user_history.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS history (
        username TEXT, origin TEXT, destination TEXT,
        congestion TEXT, accident_risk TEXT, fuel TEXT, 
        selected_time TEXT, timestamp TEXT
    )""")
    conn.execute("INSERT INTO history VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
        user, origin, destination,
        prediction['congestion_level'], prediction['accident_risk'],
        prediction['fuel_consumption'], selected_time.strftime('%H:%M'),
        str(pd.Timestamp.now())
    ))
    conn.commit()
    conn.close()

# Main content area
if st.session_state.view_history:
    # Display history view
    st.subheader("Prediction History")
    if st.session_state.history_data is not None:
        st.dataframe(st.session_state.history_data, use_container_width=True)
    else:
        st.info("No prediction history found.")
        
    # Add a button to close history view
    if st.button("Close History"):
        st.session_state.view_history = False
        st.experimental_rerun()
        
else:
    # Display prediction if it exists in session state
    if st.session_state.prediction is not None:
        # Create containers for prediction results
        time_display = st.session_state.current_time.strftime('%H:%M') if st.session_state.current_time else "current time"
        st.subheader(f"Route: {st.session_state.current_origin} to {st.session_state.current_destination} at {time_display}")
        
        # Create columns for results
        map_col, details_col = st.columns([2, 1])
        
        with map_col:
            # Display map
            display_route_map(
                st.session_state.current_origin,
                st.session_state.current_destination,
                st.session_state.prediction
            )
        
        with details_col:
            # Display prediction details
            st.subheader("Prediction Results")
            prediction = st.session_state.prediction
            
            st.markdown(f"""
            **Time of Day:** {prediction.get('time_of_day', 'N/A')}:00  
            **Congestion Level:** {prediction['congestion_level']}  
            **Traffic Volume:** {prediction.get('congestion_value', 'N/A')} vehicles/hour  
            **Accident Risk:** {prediction['accident_risk']}  
            **Fuel Consumption:** {prediction['fuel_consumption']}  
            """)
            
            # Add some recommendations based on the prediction
            st.subheader("Recommendations")
            if prediction['congestion_level'] == 'high':
                st.warning("Consider traveling during off-peak hours or using alternative routes.")
                
                # Add time-based recommendations
                hour = prediction.get('time_of_day', 0)
                if 7 <= hour <= 9:
                    st.warning("Morning peak hours. Consider delaying travel if possible.")
                elif 16 <= hour <= 18:
                    st.warning("Evening peak hours. Consider alternative routes.")
                    
            elif prediction['congestion_level'] == 'medium':
                st.info("Moderate traffic expected. Plan for some delays.")
            else:
                st.success("Traffic is light. Good time to travel.")
                
            # Show alternative times if congestion is high
            if prediction['congestion_level'] == 'high':
                st.subheader("Better Times to Travel")
                
                # Find better times from the dataset
                better_times = []
                
                # Get unique hours from the dataset
                unique_hours = sorted(df['timestamp'].dt.hour.unique())
                
                # Check each hour
                for hour in unique_hours:
                    # Skip the current selected hour
                    if hour == prediction.get('time_of_day'):
                        continue
                        
                    # Create a test time
                    test_time = selected_time.replace(hour=hour)
                    
                    # Get prediction for this hour
                    test_prediction = predict_traffic(df, origin, destination, test_time)
                    
                    # If congestion is lower, add to better times
                    if test_prediction['congestion_level'] == 'low' or (
                        prediction['congestion_level'] == 'high' and 
                        test_prediction['congestion_level'] == 'medium'
                    ):
                        better_times.append((hour, test_prediction['congestion_level']))
                
                # Display better times
                if better_times:
                    better_times_html = "<ul>"
                    for hour, level in better_times[:3]:  # Show top 3 better times
                        am_pm = "AM" if hour < 12 else "PM"
                        display_hour = hour if hour <= 12 else hour - 12
                        if display_hour == 0:
                            display_hour = 12
                        better_times_html += f"<li>{display_hour}:00 {am_pm} - {level} congestion</li>"
                    better_times_html += "</ul>"
                    st.markdown(better_times_html, unsafe_allow_html=True)
                else:
                    st.info("No significantly better times found in the dataset.")
    else:
        # Show welcome message if no prediction has been made
        st.info("👈 Select origin and destination locations, time, then click 'Predict Route' to start.")