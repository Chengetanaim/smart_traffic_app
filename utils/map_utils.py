import streamlit as st
import folium
from streamlit_folium import st_folium

location_coords = {
    "CBD": [-17.8292, 31.0522],  # Central Harare
    "Avondale": [-17.8003, 31.0353],
    "Borrowdale": [-17.7425, 31.1039],
    "Highlands": [-17.8028, 31.0819],
    "Mbare": [-17.8656, 31.0361],
    "Mount Pleasant": [-17.7686, 31.0353],
    "Westgate": [-17.8056, 30.9944]
}

def display_route_map(origin, destination, prediction):
    m = folium.Map(location=location_coords[origin], zoom_start=13)
    folium.Marker(location_coords[origin], tooltip="Origin", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(location_coords[destination], tooltip="Destination", icon=folium.Icon(color="red")).add_to(m)
    folium.PolyLine([location_coords[origin], location_coords[destination]], color="blue", weight=3).add_to(m)
    st_folium(m, width=700)
