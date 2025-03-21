import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

# Streamlit interface
st.title("Singapore Attractions Route Optimizer")

# Data Input Section
st.sidebar.header("Enter Trip Details")

transport_mode = st.sidebar.selectbox("Choose transport mode", ["walk", "bike", "drive"])

attractions = st.sidebar.text_area("Enter attraction coordinates (lat,lon)",
                                   "1.3114,103.8531\n1.2977,103.8499\n1.2843,103.8600")

# Process attractions
attraction_coords = [tuple(map(float, line.split(','))) for line in attractions.strip().split('\n')]

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

# Map display
m = folium.Map(location=[centroid_lat, centroid_lon], zoom_start=14)

# Add Marker Cluster
marker_cluster = MarkerCluster().add_to(m)
for idx, coord in enumerate(attraction_coords):
    folium.Marker(coord, popup=f"Attraction {idx+1}").add_to(marker_cluster)

# Add route polyline
folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.8,
                tooltip=f"Optimized Route ({transport_mode})").add_to(m)

# Display map in Streamlit
st_folium(m, width=700, height=500)

st.success("Map loaded successfully!")