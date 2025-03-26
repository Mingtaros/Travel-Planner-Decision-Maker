import streamlit as st
import osmnx as ox
import folium
import json
import os
import networkx as nx
import logging
from datetime import datetime
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import sys

from alns_main import alns_main

def setup_logging():
    """
    Configure application logging.
    
    Sets up both file and console logging with timestamps and appropriate
    log levels. Log files are stored in the 'log' directory with filenames
    that include the current timestamp.
    """
    # Create logs directory if it doesn't exist
    os.makedirs("log", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"log/streamlit_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)

ox.settings.use_cache = True
GRAPH_FILE = "./data/sgGraph/singapore_graph.graphml"
@st.cache_data  # Ensures caching in Streamlit
def load_graph():
    """Load the Singapore road network graph, using a cached version if available."""
    if os.path.exists(GRAPH_FILE):
        st.info("Loading cached Singapore road network...")
        logger.info("Loading cached Singapore road network...")
        return ox.load_graphml(GRAPH_FILE)
    
    st.warning("Downloading Singapore road network... (This may take a while)")
    logger.info("Downloading Singapore road network...")
    G = ox.graph_from_place("Singapore", network_type="all", simplify=True)
    
    # Save the graph to cache for future use
    ox.save_graphml(G, GRAPH_FILE)
    st.success("Graph downloaded and cached successfully!")
    logger.info("Graph downloaded and cached successfully!")
    
    return G

G = load_graph()

st.title("My Intelligent Travel Buddy – Automatic Itinerary (MITB – AI）- Singapore Edition")
st.sidebar.header("Trip Details")

personas = st.sidebar.selectbox("Choose your persona", [
    "Family Tourist", "Backpacker", "Influencer", "Cultural Enthusiast", 
    "Thrill Seeker", "Nature Lover", "Shopping Enthusiast"
])
nums_of_date = st.sidebar.number_input("Number of Days (1-5)", min_value=1, max_value=5, value=3)
budget = st.sidebar.number_input("Budget", min_value=200, value=500)
description = st.sidebar.text_area("Trip Description", "Your trip description here...")

if st.sidebar.button("Generate Itinerary"):
    user_input = {"personas": personas, "date": nums_of_date,
        "budget": budget, "description": description}
    json_path = "../../user_input.json"
    with open(json_path, 'w') as f:
        json.dump(user_input, f, indent=4)
    st.sidebar.success(f"Data saved to {json_path}")

def find_route_between_points(G, start_point, end_point):
    """
    Find the shortest path between two points using driving network
    """
    try:
        start_node = ox.nearest_nodes(G, start_point[1], start_point[0])
        end_node = ox.nearest_nodes(G, end_point[1], end_point[0])
        route = nx.shortest_path(G, start_node, end_node, weight="length")
        return [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
    except nx.NetworkXNoPath:
        # Fallback to direct line if no route found
        return [start_point, end_point]

# itinerary_file = "../../results/transit_time/best_itinerary_20250325_154140.json"
# if not os.path.exists(itinerary_file):
#     st.error("Itinerary file not found!")
#     st.stop()

# with open(itinerary_file, "r", encoding="utf-8") as file:
#     data = json.load(file)

# Route colors with their names
route_colors = [
    ("blue", "Blue"), 
    ("red", "Red"), 
    ("green", "Green"), 
    ("purple", "Purple"), 
    ("orange", "Orange"), 
    ("darkred", "Dark Red"), 
    ("darkgreen", "Dark Green")
]

# Function to create map
def create_map():
    
    alns_data = alns_main()

    logger.info("Itinerary data loaded successfully!")
    
    # Prepare locations and map
    locations = []
    for day in alns_data["days"]:
        locations.extend(day["locations"])

    center_lat = sum(loc["lat"] for loc in locations) / len(locations)
    center_lng = sum(loc["lng"] for loc in locations) / len(locations)
    
    # Create map
    m = folium.Map(location=[center_lat, center_lng], zoom_start=12)
    marker_cluster = MarkerCluster().add_to(m)

    # Draw routes for each day
    for day_index, day in enumerate(alns_data["days"]):
        day_locations = day["locations"]
        color, color_name = route_colors[day_index % len(route_colors)]
        
        # Draw route between locations in the day
        if len(day_locations) > 1:
            for i in range(len(day_locations) - 1):
                start_point = (day_locations[i]["lat"], day_locations[i]["lng"])
                end_point = (day_locations[i+1]["lat"], day_locations[i+1]["lng"])
                route_segment = find_route_between_points(G, start_point, end_point)
                folium.PolyLine(route_segment, color=color, weight=5, opacity=0.8, 
                                tooltip=f"Day {day_index + 1} Route").add_to(m)

    # Add markers
    for loc in locations:
        name = loc.get('name', '')
        description = loc.get('description', '')
        popup_text = f"""
        <b>{name}</b><br>
        {description}<br>
        Arrival: {loc.get('arrival_time', 'N/A')}<br>
        Departure: {loc.get('departure_time', 'N/A')}<br>
        Duration: {loc.get('duration', 0)} min<br>
        Cost: ${loc.get('cost', 0)}
        """
        folium.Marker([loc["lat"], loc["lng"]], popup=popup_text).add_to(marker_cluster)

    return m, alns_data

# Function to display detailed sidebar overview
def display_detailed_overview(data):
    st.sidebar.header("Itinerary Overview")
    for day_index, day in enumerate(data["days"]):
        # Get color name for the day
        _, color_name = route_colors[day_index % len(route_colors)]

        st.sidebar.markdown("---")
        st.sidebar.write(f"Day {day_index + 1} ({color_name} Route)")
        for loc in day["locations"]:
            # Display location details
            st.sidebar.markdown(f"**{loc.get('name', 'Unnamed Location')}**")
            
            # Description (if exists)
            if loc.get('description'):
                st.sidebar.write(loc['description'])
            
            # Arrival time
            if loc.get('arrival_time'):
                st.sidebar.write(f"Arrival: {loc['arrival_time']}")
            
            # Duration and cost
            duration = loc.get('duration', 0)
            cost = loc.get('cost', 0)
            st.sidebar.write(f"Duration: {duration} min, Cost: ${cost}")
            st.sidebar.markdown(" ")
            st.sidebar.markdown(" ")
            

# Button to show map and overview
if st.button("Show Trip Map"):
    # Create and display map
    m, alns_data = create_map()
    folium_static(m)
    
    # Display detailed overview
    display_detailed_overview(alns_data)
    
    st.success("Map loaded successfully!")
    logger.info("Map loaded successfully!")