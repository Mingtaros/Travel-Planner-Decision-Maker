import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import json
import os

# Streamlit interface
st.title("Singapore Attractions Route Optimizer")

# Custom CSS for smaller font size
st.markdown(
    """
    <style>
    .small-font {
        font-size:14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Data Input Section
st.sidebar.header("Enter Trip Details")
name = st.text_input("Enter your name", "John Doe")
age = st.number_input("Enter your age", min_value=1, max_value=100, value=25)
date = st.date_input("Enter travel date")
time = st.time_input("Enter travel time")
description = st.text_area("Enter trip description", "Fun trip around Singapore")

transport_mode = st.sidebar.selectbox("Choose transport mode", ["walk", "bike", "drive"])

# Collect input data
input_data = {
    "name": name,
    "age": age,
    "date": str(date),
    "time": str(time),
    "description": description,
    "transport_mode": transport_mode
}

# Write input data to JSON file
if st.button("Plan Itinerary for Me"):
    json_file_path = os.path.join(os.getcwd(), "input_data.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(input_data, json_file, indent=4)
    st.success(f"Input data saved to {json_file_path}")

    # Provide download link
    with open(json_file_path, 'rb') as file:
        btn = st.download_button(
            label="Activate LLM",
            data=file,
            file_name="input_data.json",
            mime="application/json"
        )

# Read JSON file (NEW)
with open(r'C:\Users\Valerian Yap\Travel-Planner-Decision-Maker\Travel-Planner-Decision-Maker\notebooks\test_locations.json', 'r') as f:
    locations_data = json.load(f)

# Extract coordinates and names from JSON data
attraction_coords = [(loc['lat'], loc['lng']) for loc in locations_data if 'lat' in loc and 'lng' in loc]
attraction_names = [loc['name'] for loc in locations_data if 'name' in loc]
attraction_desc = [loc['description'] for loc in locations_data if 'description' in loc]

# attractions = st.sidebar.text_area("Enter attraction coordinates (lat,lon)",
#                                    "1.3114,103.8531\n1.2977,103.8499\n1.2843,103.8600")

# # Process attractions
# attraction_coords = [tuple(map(float, line.split(','))) for line in attractions.strip().split('\n')]

# Fetch road network data around the centroid of given attractions
centroid_lat = sum(lat for lat, lon in attraction_coords) / len(attraction_coords)
centroid_lon = sum(lon for lat, lon in attraction_coords) / len(attraction_coords)

G = ox.graph_from_point((centroid_lat, centroid_lon), dist=4000, network_type=transport_mode, simplify=True)

# Find optimized route through all attractions (simple approach: sequentially connect)
route_coords = []
for i in range(len(attraction_coords) - 1):
    start_node = ox.nearest_nodes(G, attraction_coords[i][1], attraction_coords[i][0])
    end_node = ox.nearest_nodes(G, attraction_coords[i + 1][1], attraction_coords[i + 1][0])
    route = nx.astar_path(G, start_node, end_node, weight="length")
    route_segment = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
    route_coords.extend(route_segment)

# Create columns for layout
col1, col2 = st.columns([2, 1])

# Map display in the first column
with col1:
    m = folium.Map(location=[centroid_lat, centroid_lon], zoom_start=14)

    # Add Marker Cluster
    marker_cluster = MarkerCluster().add_to(m)
    for idx, coord in enumerate(attraction_coords):
        folium.Marker(coord, popup=attraction_names[idx]).add_to(marker_cluster)

    # Add route polyline
    folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.8,
                    tooltip=f"Optimized Route ({transport_mode})").add_to(m)

    # Display map in Streamlit
    st_folium(m, width=700, height=500)

# Itinerary summary in the second column
with col2:
    st.header("Itinerary Summary")
    for idx, name in enumerate(attraction_names):
        st.markdown(f"<div class='small-font'><strong>{idx + 1}. {name}</strong></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-font'>{attraction_desc[idx]}</div>", unsafe_allow_html=True)

st.success("Map and itinerary loaded successfully!")

