import streamlit as st
import pandas as pd
import sqlite3
import os
import folium
from streamlit_folium import st_folium
import numpy as np
import random
import datetime
import hashlib
from itertools import combinations

# Initialize session state variables if they don't exist
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
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
if 'show_admin_panel' not in st.session_state:
    st.session_state.show_admin_panel = False
if 'show_user_creation' not in st.session_state:
    st.session_state.show_user_creation = False
if 'alternative_routes' not in st.session_state:
    st.session_state.alternative_routes = None

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    return stored_password == hashlib.sha256(provided_password.encode()).hexdigest()

def login_user():
    """Handle user login process"""
    if not st.session_state.logged_in:
        st.subheader("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid credentials")
        return None
    else:
        st.sidebar.success(f"Logged in as {st.session_state.username}")
        return st.session_state.username

def authenticate_user(username, password):
    """Check if username and password are correct"""
    conn = sqlite3.connect("data/traffic_management.db")
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        
        if result is None:
            return False
        
        stored_password = result[0]
        return verify_password(stored_password, password)
    finally:
        conn.close()

def calculate_route_distance(route_points):
    """Calculate approximate distance of a route in kilometers"""
    total_distance = 0
    for i in range(len(route_points) - 1):
        lat1, lon1 = route_points[i]
        lat2, lon2 = route_points[i + 1]
        
        # Haversine formula for distance
        R = 6371  # Earth's radius in kilometers
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2) * np.sin(dlat/2) + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * 
             np.sin(dlon/2) * np.sin(dlon/2))
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        total_distance += distance
    
    return total_distance

def find_alternative_routes(origin, destination, df, selected_time, max_routes=3):
    """Find alternative routes between origin and destination"""
    alternatives = []
    
    # Get all possible intermediate locations
    all_locations = list(location_coords.keys())
    intermediate_locations = [loc for loc in all_locations if loc not in [origin, destination]]
    
    # Direct route first (if not blocked and not high congestion)
    direct_prediction = predict_traffic(df, origin, destination, selected_time)
    if not direct_prediction.get('blocked', False) and direct_prediction['congestion_level'] != 'high':
        alternatives.append({
            'route': [origin, destination],
            'route_name': f"Direct: {origin} → {destination}",
            'prediction': direct_prediction,
            'total_distance': calculate_route_distance([location_coords[origin], location_coords[destination]]),
            'route_type': 'direct'
        })
    
    # Generate alternative routes through intermediate points
    for intermediate in intermediate_locations:
        # Route: Origin → Intermediate → Destination
        route_points = [origin, intermediate, destination]
        
        # Check if any segment is blocked
        segment1_blocked = is_route_blocked(origin, intermediate)
        segment2_blocked = is_route_blocked(intermediate, destination)
        
        if segment1_blocked or segment2_blocked:
            continue
        
        # Get predictions for both segments
        segment1_prediction = predict_traffic(df, origin, intermediate, selected_time)
        segment2_prediction = predict_traffic(df, intermediate, destination, selected_time)
        
        # Calculate overall route metrics
        avg_congestion_value = round((segment1_prediction['congestion_value'] + segment2_prediction['congestion_value']) / 2)
        
        # Determine overall congestion level
        if avg_congestion_value <= 3:
            overall_congestion = "low"
        elif avg_congestion_value <= 6:
            overall_congestion = "medium"
        else:
            overall_congestion = "high"
        
        # Calculate total distance
        route_coords = [location_coords[loc] for loc in route_points]
        total_distance = calculate_route_distance(route_coords)
        
        # Create combined prediction
        combined_prediction = {
            'congestion_level': overall_congestion,
            'congestion_value': avg_congestion_value,
            'accident_risk': f"{((float(segment1_prediction['accident_risk'].rstrip('%')) + float(segment2_prediction['accident_risk'].rstrip('%'))) / 2):.1f}%",
            'fuel_consumption': f"{((float(segment1_prediction['fuel_consumption'].split()[0]) + float(segment2_prediction['fuel_consumption'].split()[0])) / 2):.2f} L/100km",
            'time_of_day': selected_time.hour if selected_time else datetime.datetime.now().hour,
            'blocked': False,
            'segments': [segment1_prediction, segment2_prediction]
        }
        
        alternatives.append({
            'route': route_points,
            'route_name': f"Via {intermediate}: {origin} → {intermediate} → {destination}",
            'prediction': combined_prediction,
            'total_distance': total_distance,
            'route_type': 'alternative'
        })
    
    # Sort alternatives by congestion level and distance
    def route_score(route):
        congestion_score = {'low': 1, 'medium': 2, 'high': 3}[route['prediction']['congestion_level']]
        distance_penalty = route['total_distance'] * 0.1  # Small penalty for longer routes
        return congestion_score + distance_penalty
    
    alternatives.sort(key=route_score)
    
    return alternatives[:max_routes]

def predict_traffic(df, origin, destination, selected_time=None):
    """Predict traffic conditions based on origin, destination, and time."""
    # Check if route is blocked
    if is_route_blocked(origin, destination):
        return {
            "congestion_level": "blocked",
            "congestion_value": 10,
            "accident_risk": "100%",
            "fuel_consumption": "N/A",
            "time_of_day": selected_time.hour if selected_time else datetime.datetime.now().hour,
            "blocked": True,
            "blocked_reason": get_blocked_reason(origin, destination)
        }
    
    # If no time provided, use current time
    if selected_time is None:
        now = datetime.datetime.now()
        current_hour = now.hour
    else:
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
        origin_recent = origin_data.sort_values(by="timestamp").iloc[-1]
        dest_recent = destination_data.sort_values(by="timestamp").iloc[-1]
        
        congestion_level = round((origin_recent['congestion_level'] * 0.7 + 
                                 dest_recent['congestion_level'] * 0.3))
        
        traffic_volume = round((origin_recent.get('traffic_volume', 1000) * 0.7 + 
                              dest_recent.get('traffic_volume', 1000) * 0.3))
        
        accident_prob = (origin_recent['accident_probability'] * 0.7 + 
                         dest_recent['accident_probability'] * 0.3)
        
        fuel_consumption = (origin_recent['fuel_consumption_l_per_100km'] * 0.6 + 
                           dest_recent['fuel_consumption_l_per_100km'] * 0.4)
        
        # Convert numerical congestion to descriptive labels
        if congestion_level <= 6:
            congestion_label = "low"

        elif congestion_level <= 10:
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
            "time_of_day": current_hour,
            "blocked": False
        }
    else:
        return {
            "congestion_level": "unknown",
            "congestion_value": 0,
            "accident_risk": "N/A",
            "fuel_consumption": "N/A",
            "time_of_day": current_hour,
            "blocked": False
        }

def is_route_blocked(origin, destination):
    """Check if a route is blocked in the database"""
    os.makedirs("data", exist_ok=True)
    
    conn = sqlite3.connect("data/traffic_management.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS blocked_routes (
            origin TEXT,
            destination TEXT,
            reason TEXT,
            blocked_by TEXT,
            blocked_at TEXT,
            PRIMARY KEY (origin, destination)
        )
    """)
    
    cursor.execute("""
        SELECT 1 FROM blocked_routes 
        WHERE (origin = ? AND destination = ?)
        OR (origin = ? AND destination = ?)
    """, (origin, destination, destination, origin))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def get_blocked_reason(origin, destination):
    """Get the reason for a blocked route"""
    conn = sqlite3.connect("data/traffic_management.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT reason FROM blocked_routes 
        WHERE (origin = ? AND destination = ?)
        OR (origin = ? AND destination = ?)
    """, (origin, destination, destination, origin))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else "Route blocked due to incident"

def block_route(origin, destination, reason, username):
    """Block a route in the database"""
    conn = sqlite3.connect("data/traffic_management.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO blocked_routes 
        (origin, destination, reason, blocked_by, blocked_at)
        VALUES (?, ?, ?, ?, ?)
    """, (origin, destination, reason, username, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()

def unblock_route(origin, destination):
    """Unblock a route in the database"""
    conn = sqlite3.connect("data/traffic_management.db")
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM blocked_routes 
        WHERE (origin = ? AND destination = ?)
        OR (origin = ? AND destination = ?)
    """, (origin, destination, destination, origin))
    conn.commit()
    conn.close()

def get_all_blocked_routes():
    """Get all currently blocked routes"""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/traffic_management.db")
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS blocked_routes (
            origin TEXT,
            destination TEXT,
            reason TEXT,
            blocked_by TEXT,
            blocked_at TEXT,
            PRIMARY KEY (origin, destination)
        )
    """)
    
    df = pd.read_sql("SELECT * FROM blocked_routes", conn)
    conn.close()
    return df

def create_user(username, password, is_admin=False):
    """Create a new user in the database"""
    conn = sqlite3.connect("data/traffic_management.db")
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO users (username, password, is_admin)
            VALUES (?, ?, ?)
        """, (username, hash_password(password), int(is_admin)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def initialize_database():
    """Initialize the database with required tables"""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/traffic_management.db")
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS blocked_routes (
            origin TEXT,
            destination TEXT,
            reason TEXT,
            blocked_by TEXT,
            blocked_at TEXT,
            PRIMARY KEY (origin, destination)
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            is_admin INTEGER DEFAULT 0
        )
    """)
    
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM users WHERE username = ?", ("admin",))
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO users (username, password, is_admin)
            VALUES (?, ?, ?)
        """, ("admin", hash_password("admin"), 1))
    
    conn.commit()
    conn.close()

# Initialize database
initialize_database()

# Dictionary of location coordinates in Harare
location_coords = {
    "CBD": [-17.8292, 31.0522],
    "Avondale": [-17.8003, 31.0353],
    "Borrowdale": [-17.7425, 31.1039],
    "Highlands": [-17.8028, 31.0819],
    "Mbare": [-17.8656, 31.0361],
    "Mount Pleasant": [-17.7686, 31.0353],
    "Westgate": [-17.8056, 30.9944]
}

def generate_intermediate_points(start, end, num_points=3, variance=0.003):
    """Generate realistic route with intermediate points between start and end"""
    lat_diff = end[0] - start[0]
    lng_diff = end[1] - start[1]
    
    points = []
    for i in range(1, num_points + 1):
        frac = i / (num_points + 1)
        mid_lat = start[0] + lat_diff * frac
        mid_lng = start[1] + lng_diff * frac
        
        angle = np.arctan2(lat_diff, lng_diff) + np.pi/2
        deviation = random.uniform(-variance, variance)
        mid_lat += deviation * np.sin(angle)
        mid_lng += deviation * np.cos(angle)
        
        points.append([mid_lat, mid_lng])
    
    return points

def display_route_map(origin, destination, prediction, alternative_routes=None):
    """Display a map with routes and alternatives"""
    center_lat = (location_coords[origin][0] + location_coords[destination][0]) / 2
    center_lng = (location_coords[origin][1] + location_coords[destination][1]) / 2
    m = folium.Map(location=[center_lat, center_lng], zoom_start=12)
    
    # Add markers for all locations
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
    
    # Display alternative routes if available
    if alternative_routes:
        colors = ['blue', 'purple', 'orange', 'darkgreen']
        
        for i, route_info in enumerate(alternative_routes):
            route = route_info['route']
            pred = route_info['prediction']
            
            # Determine line style based on route type and congestion
            if route_info['route_type'] == 'direct':
                line_color = colors[0] if not pred.get('blocked', False) else 'black'
                line_weight = 6
                line_opacity = 0.9
            else:
                line_color = colors[min(i + 1, len(colors) - 1)]
                line_weight = 4
                line_opacity = 0.7
            
            # Generate route points
            if len(route) == 2:  # Direct route
                random.seed(f"{route[0]}_{route[1]}")
                intermediate_points = generate_intermediate_points(
                    location_coords[route[0]], 
                    location_coords[route[1]],
                    num_points=4
                )
                route_points = [location_coords[route[0]]] + intermediate_points + [location_coords[route[1]]]
            else:  # Route with intermediate stops
                route_points = []
                for j in range(len(route) - 1):
                    start_coord = location_coords[route[j]]
                    end_coord = location_coords[route[j + 1]]
                    
                    random.seed(f"{route[j]}_{route[j+1]}")
                    intermediate = generate_intermediate_points(start_coord, end_coord, num_points=2)
                    
                    if j == 0:
                        route_points.append(start_coord)
                    route_points.extend(intermediate)
                    route_points.append(end_coord)
            
            random.seed()
            
            # Add route line
            congestion_text = "BLOCKED" if pred.get('blocked', False) else pred.get('congestion_level', 'unknown')
            route_tooltip = f"{route_info['route_name']} - {congestion_text.upper()} - {route_info['total_distance']:.1f}km"
            
            folium.PolyLine(
                route_points, 
                color=line_color, 
                weight=line_weight, 
                opacity=line_opacity, 
                tooltip=route_tooltip
            ).add_to(m)
            
            # Add intermediate location markers for alternative routes
            if len(route) > 2:
                for intermediate_loc in route[1:-1]:
                    folium.Marker(
                        location_coords[intermediate_loc],
                        tooltip=f"Via: {intermediate_loc}",
                        icon=folium.Icon(color="gray", icon="info-sign", prefix="fa")
                    ).add_to(m)
    
    else:
        # Original single route display
        random.seed(f"{origin}_{destination}")
        intermediate_points = generate_intermediate_points(
            location_coords[origin], 
            location_coords[destination],
            num_points=4
        )
        random.seed()
        
        route_points = [location_coords[origin]] + intermediate_points + [location_coords[destination]]
        
        if prediction.get('blocked', False):
            line_color = 'black'
            congestion = "BLOCKED"
        else:
            congestion = prediction.get('congestion_level', 'medium')
            if congestion == 'low':
                line_color = 'green'
            elif congestion == 'medium':
                line_color = 'orange'
            else:
                line_color = 'red'
        
        folium.PolyLine(
            route_points, 
            color=line_color, 
            weight=4, 
            opacity=0.8, 
            tooltip=f"Status: {congestion}"
        ).add_to(m)
    
    return st_folium(m, width=700, height=500, key=f"map_{origin}_{destination}_alt")

def get_user_history(username):
    """Get prediction history for a user"""
    os.makedirs("history", exist_ok=True)
    
    conn = sqlite3.connect("history/user_history.db")
    
    conn.execute("""CREATE TABLE IF NOT EXISTS history (
        username TEXT, origin TEXT, destination TEXT,
        congestion TEXT, accident_risk TEXT, fuel TEXT, 
        selected_time TEXT, timestamp TEXT
    )""")
    
    rows = conn.execute("SELECT * FROM history WHERE username = ?", (username,)).fetchall()
    conn.close()
    
    if rows:
        return pd.DataFrame(rows, columns=[
            "Username", "Origin", "Destination", "Congestion", "Accident Risk", 
            "Fuel", "Selected Time", "Prediction Time"
        ])
    return None

def save_to_history(user, origin, destination, prediction, selected_time):
    """Save a prediction to user history"""
    os.makedirs("history", exist_ok=True)
    
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

def create_sample_data():
    """Create simulated traffic data if it doesn't exist"""
    os.makedirs("data", exist_ok=True)
    
    data_path = os.path.join("data", "simulated_traffic_data.csv")
    
    if os.path.exists(data_path):
        return
        
    locations = list(location_coords.keys())
    
    end_time = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
    start_time = end_time - datetime.timedelta(days=7)
    date_range = pd.date_range(start=start_time, end=end_time, freq='H')
    
    data = []
    
    for location in locations:
        for timestamp in date_range:
            hour = timestamp.hour
            
            base_congestion = 2
            
            if 7 <= hour <= 9:
                base_congestion = 7 + random.uniform(-1, 1)
            elif 16 <= hour <= 18:
                base_congestion = 8 + random.uniform(-1, 1)
            elif 10 <= hour <= 15 or hour == 19:
                base_congestion = 4 + random.uniform(-1, 1)
            else:
                base_congestion = 2 + random.uniform(-1, 1)
            
            if location == "CBD":
                location_factor = 1.3
            elif location in ["Borrowdale", "Avondale", "Mbare"]:
                location_factor = 1.1
            else:
                location_factor = 0.9
                
            congestion_level = max(1, min(10, base_congestion * location_factor))
            
            accident_probability = min(0.95, max(0.05, congestion_level / 20 + random.uniform(-0.05, 0.05)))
            fuel_consumption = 5 + (congestion_level / 2)
            
            congestion_level = round(congestion_level + random.uniform(-0.5, 0.5), 1)
            accident_probability = round(accident_probability + random.uniform(-0.02, 0.02), 3)
            fuel_consumption = round(fuel_consumption + random.uniform(-0.5, 0.5), 2)
            
            traffic_volume = int(congestion_level * 150 + random.uniform(-50, 50))
            
            data.append({
                'timestamp': timestamp,
                'location': location,
                'congestion_level': congestion_level,
                'traffic_volume': traffic_volume,
                'accident_probability': accident_probability,
                'fuel_consumption_l_per_100km': fuel_consumption
            })
    
    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False)

# Create sample data if needed
create_sample_data()

# Set page config
st.set_page_config(page_title="Smart Traffic System - Harare", layout="wide")
st.title("Traffic Flow Modelling")

# User login
user = login_user()
if not st.session_state.logged_in:
    st.stop()

# Display logout button in the sidebar
with st.sidebar:
    st.write(f"Logged in as: {st.session_state.username}")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.prediction = None
        st.session_state.view_history = False
        st.session_state.alternative_routes = None
        st.rerun()

# Check if user is admin
conn = sqlite3.connect("data/traffic_management.db")
cursor = conn.cursor()
cursor.execute("SELECT is_admin FROM users WHERE username = ?", (st.session_state.username,))
is_admin = cursor.fetchone()
conn.close()

is_admin = is_admin and is_admin[0] == 1 if is_admin else False

# Load data
data_path = os.path.join("data", "simulated_traffic_data.csv")
if os.path.exists(data_path):
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
else:
    st.error("Traffic data file not found. Please ensure the data file exists.")
    st.stop()

# Create sidebar
with st.sidebar:
    st.header("Select Route")
    
    origin = st.selectbox("Origin", sorted(list(location_coords.keys())), key='origin_select')
    destination = st.selectbox("Destination", sorted(list(location_coords.keys())), key='destination_select')
    
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
    
    if origin == destination:
        st.warning("Please select different locations for origin and destination.")
    
    predict_pressed = st.button("Predict Route", key="predict_button", disabled=(origin == destination))
    
    # Add checkbox for showing alternatives
    show_alternatives = st.checkbox("Show Alternative Routes", value=True)
    
    history_pressed = st.button("View History", key="history_button")
    if history_pressed:
        st.session_state.view_history = not st.session_state.view_history
        if st.session_state.view_history:
            st.session_state.history_data = get_user_history(st.session_state.username)
    
    # Admin controls
    if is_admin:
        st.divider()
        st.subheader("Admin Controls")
        
        if st.button("Admin Panel"):
            st.session_state.show_admin_panel = not st.session_state.show_admin_panel
        
        if st.button("Create New User"):
            st.session_state.show_user_creation = not st.session_state.show_user_creation

# Admin Panel
if is_admin and st.session_state.show_admin_panel:
    with st.expander("🚨 Admin Panel", expanded=True):
        st.subheader("Route Management")
        
        st.write("### Block Route")
        block_origin = st.selectbox("Block Origin", sorted(list(location_coords.keys())), key='block_origin')
        block_dest = st.selectbox("Block Destination", sorted(list(location_coords.keys())), key='block_dest')
        block_reason = st.text_input("Reason for blocking")
        
        if st.button("🚧 Block Route"):
            if block_origin == block_dest:
                st.error("Origin and destination must be different")
            else:
                block_route(block_origin, block_dest, block_reason, st.session_state.username)
                st.success(f"Route {block_origin} to {block_dest} blocked successfully")
        
        # Unblock route section
        st.write("### Unblock Route")
        blocked_routes = get_all_blocked_routes()
        if not blocked_routes.empty:
            route_to_unblock = st.selectbox(
                "Select route to unblock",
                [f"{row['origin']} to {row['destination']} ({row['reason']})" 
                 for _, row in blocked_routes.iterrows()]
            )
            
            if st.button("✅ Unblock Route"):
                selected_origin, selected_dest = route_to_unblock.split(" to ")[0], route_to_unblock.split(" to ")[1].split(" (")[0]
                unblock_route(selected_origin, selected_dest)
                st.success(f"Route {selected_origin} to {selected_dest} unblocked")
                st.rerun()
        else:
            st.info("No currently blocked routes")
        
        # View all blocked routes
        st.write("### Currently Blocked Routes")
        if not blocked_routes.empty:
            st.dataframe(blocked_routes)
        else:
            st.info("No routes are currently blocked")

# User Creation Panel
if is_admin and st.session_state.show_user_creation:
    with st.expander("👤 Create New User", expanded=True):
        st.subheader("User Creation")
        
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        is_admin_user = st.checkbox("Admin privileges")
        
        if st.button("Create User"):
            if new_username and new_password:
                if create_user(new_username, new_password, is_admin_user):
                    st.success(f"User {new_username} created successfully")
                else:
                    st.error("Username already exists")
            else:
                st.error("Please provide both username and password")

# Main content area
if st.session_state.view_history:
    # Display history view
    st.subheader("Prediction History")
    if st.session_state.history_data is not None and not st.session_state.history_data.empty:
        st.dataframe(st.session_state.history_data, use_container_width=True)
    else:
        st.info("No prediction history found.")
        
    # Add a button to close history view
    if st.button("Close History"):
        st.session_state.view_history = False
        st.rerun()
        
else:
    # Display prediction if it exists in session state
    if st.session_state.prediction is not None:
        # Create containers for prediction results
        time_display = st.session_state.current_time.strftime('%H:%M') if st.session_state.current_time else "current time"
        st.subheader(f"Route: {st.session_state.current_origin} to {st.session_state.current_destination} at {time_display}")
        
        # Create columns for results
        map_col, details_col = st.columns([2, 1])
        
        with map_col:
            # Display map with alternatives if available
            display_route_map(
                st.session_state.current_origin,
                st.session_state.current_destination,
                st.session_state.prediction,
                st.session_state.alternative_routes
            )
        
        with details_col:
            # Display prediction details
            st.subheader("Route Options")
            
            # Display alternative routes if available
            if st.session_state.alternative_routes:
                for i, route_info in enumerate(st.session_state.alternative_routes):
                    prediction = route_info['prediction']
                    route_name = route_info['route_name']
                    distance = route_info['total_distance']
                    
                    # Create expandable section for each route
                    with st.expander(f"🚗 {route_name} ({distance:.1f}km)", expanded=(i == 0)):
                        if prediction.get('blocked', False):
                            st.error("### 🚨 ROUTE BLOCKED")
                            st.markdown(f"""
                            **Reason:** {prediction.get('blocked_reason', 'Accident or incident')}  
                            **Status:** AVOID THIS ROUTE  
                            """)
                            
                            if is_admin:
                                route_key = f"unblock_{route_info['route'][0]}_{route_info['route'][-1]}_{i}"
                                if st.button("Unblock This Route", key=route_key):
                                    unblock_route(route_info['route'][0], route_info['route'][-1])
                                    st.success("Route unblocked")
                                    st.session_state.prediction = None
                                    st.session_state.alternative_routes = None
                                    st.rerun()
                        else:
                            # Color coding for congestion levels
                            congestion_level = prediction['congestion_level']
                            if congestion_level == 'low':
                                congestion_color = "🟢"
                                recommendation = "✅ **RECOMMENDED** - Light traffic"
                            elif congestion_level == 'medium':
                                congestion_color = "🟡"
                                recommendation = "⚠️ Moderate traffic expected"
                            else:
                                congestion_color = "🔴"
                                recommendation = "❌ Heavy traffic - consider alternatives"
                            
                            st.markdown(f"### {congestion_color} {congestion_level.upper()} TRAFFIC")
                            st.markdown(f"*{recommendation}*")
                            
                            st.markdown(f"""
                            **Time:** {time_display}  
                            **Distance:** {distance:.1f} km  
                            **Congestion Level:** {congestion_level}  
                            **Traffic Volume:** {prediction.get('congestion_value', 'N/A')} vehicles/hour  
                            **Accident Risk:** {prediction['accident_risk']}  
                            **Fuel Consumption:** {prediction['fuel_consumption']}  
                            """)
                            
                            # Show segment details for multi-segment routes
                            if 'segments' in prediction and len(prediction['segments']) > 1:
                                st.write("**Route Segments:**")
                                for j, segment in enumerate(prediction['segments']):
                                    segment_from = route_info['route'][j]
                                    segment_to = route_info['route'][j + 1]
                                    st.write(f"• {segment_from} → {segment_to}: {segment['congestion_level']} congestion")
                
                # Overall recommendations
                st.subheader("📋 Recommendations")
                
                # Find the best route
                best_route = None
                for route_info in st.session_state.alternative_routes:
                    if not route_info['prediction'].get('blocked', False):
                        if best_route is None:
                            best_route = route_info
                        elif (route_info['prediction']['congestion_level'] == 'low' and 
                              best_route['prediction']['congestion_level'] != 'low'):
                            best_route = route_info
                        elif (route_info['prediction']['congestion_level'] == best_route['prediction']['congestion_level'] and
                              route_info['total_distance'] < best_route['total_distance']):
                            best_route = route_info
                
                if best_route:
                    if best_route['prediction']['congestion_level'] == 'low':
                        st.success(f"✅ **Best Option:** {best_route['route_name']} - Light traffic, {best_route['total_distance']:.1f}km")
                    elif best_route['prediction']['congestion_level'] == 'medium':
                        st.info(f"ℹ️ **Recommended:** {best_route['route_name']} - Moderate traffic, {best_route['total_distance']:.1f}km")
                    else:
                        st.warning("⚠️ All routes have heavy traffic. Consider traveling at a different time.")
                
                # Time-based recommendations
                current_hour = st.session_state.current_time.hour
                if 7 <= current_hour <= 9:
                    st.warning("🕐 **Peak Hours:** Morning rush hour. Consider delaying travel if possible.")
                elif 16 <= current_hour <= 18:
                    st.warning("🕐 **Peak Hours:** Evening rush hour. Heavy traffic expected on all routes.")
                elif 22 <= current_hour or current_hour <= 5:
                    st.info("🌙 **Off-Peak:** Night time travel - generally lighter traffic.")
                
                # Fuel efficiency tip
                if best_route and best_route['prediction']['congestion_level'] != 'high':
                    fuel_consumption = float(best_route['prediction']['fuel_consumption'].split()[0])
                    if fuel_consumption < 7:
                        st.success("⛽ **Fuel Efficient:** This route offers good fuel economy.")
                    elif fuel_consumption > 9:
                        st.warning("⛽ **High Fuel Consumption:** Consider carpooling or public transport.")
                
            else:
                # Original single route display (fallback)
                prediction = st.session_state.prediction
                
                if prediction.get('blocked', False):
                    st.error("### 🚨 ROUTE BLOCKED")
                    st.markdown(f"""
                    **Route:** {st.session_state.current_origin} to {st.session_state.current_destination}  
                    **Reason:** {prediction.get('blocked_reason', 'Accident or incident')}  
                    **Time:** {time_display}  
                    **Status:** AVOID THIS ROUTE  
                    """)
                    
                    if is_admin:
                        if st.button("Unblock This Route"):
                            unblock_route(st.session_state.current_origin, st.session_state.current_destination)
                            st.success("Route unblocked")
                            st.session_state.prediction = None
                            st.rerun()
                else:
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
    else:
        # Show welcome message if no prediction has been made
        st.info("👈 Select origin and destination locations, time, then click 'Predict Route' to start.")

# Handle prediction logic
if predict_pressed and origin != destination:
    # Store prediction in session state
    prediction = predict_traffic(df, origin, destination, selected_time)
    st.session_state.prediction = prediction
    st.session_state.current_origin = origin
    st.session_state.current_destination = destination
    st.session_state.current_time = selected_time
    st.session_state.map_displayed = None
    st.session_state.view_history = False
    
    # Find alternative routes if requested or if main route has issues
    if show_alternatives or prediction.get('blocked', False) or prediction['congestion_level'] == 'high':
        alternative_routes = find_alternative_routes(origin, destination, df, selected_time)
        st.session_state.alternative_routes = alternative_routes
    else:
        st.session_state.alternative_routes = None
    
    # Save to history
    save_to_history(st.session_state.username, origin, destination, prediction, selected_time)
    
    # Trigger a rerun to update the UI
    st.rerun()