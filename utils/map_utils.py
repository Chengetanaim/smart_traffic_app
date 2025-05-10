import streamlit as st
import folium
from streamlit_folium import st_folium

location_coords = {
    "CBD": [-19.4570, 29.8130],
    "Mkoba": [-19.4568, 29.8663],
    "Ascot": [-19.4386, 29.8094],
    "Senga": [-19.4311, 29.8356],
    "Lundi Park": [-19.4350, 29.8261],
    "Ridgemont": [-19.4600, 29.8010],
    "Mambo": [-19.4423, 29.7892]
}

def display_route_map(origin, destination, prediction):
    m = folium.Map(location=location_coords[origin], zoom_start=13)
    folium.Marker(location_coords[origin], tooltip="Origin", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(location_coords[destination], tooltip="Destination", icon=folium.Icon(color="red")).add_to(m)
    folium.PolyLine([location_coords[origin], location_coords[destination]], color="blue", weight=3).add_to(m)
    st_folium(m, width=700)
