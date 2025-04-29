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
import numpy as np

import json
import pandas as pd
import random

from alns_main import alns_main

#==================
import time
import random

from agentic.multiagent import (
    get_json_from_query,
    get_combine_json_data,
    find_alternative_of_affected_pois,
    update_itinerary_llm,
    update_itinerary_closest_alternatives,
    rebuild_full_itinerary,
)


st.set_page_config(page_title="Travel Itinerary Planner", layout="wide")

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

def display_itinerary():
    """Display the itinerary in a structured table format."""
    alns_data = st.session_state["alns_data"]
    if "days" in alns_data:
        all_days = []
        for day in alns_data["days"]:
            for event in day["locations"]:
                all_days.append({
                    "Day": day["day"],
                    "Location": event["name"],
                    "Type": event["type"],
                    "Arrival Time": event["arrival_time"],
                    "Departure Time": event["departure_time"],
                    "Transport Mode": event["transit_from_prev"],
                    "Transit Duration": event["transit_duration"],
                    "Transit Cost": event["transit_cost"],
                    "Activity Duration": event.get("duration", 0),
                    "Satisfaction Rating": event["satisfaction"],
                    "Cost": event["cost"],
                    "Rest Time": event["rest_duration"],
                })
        df = pd.DataFrame(all_days)
        st.dataframe(df, use_container_width=True)
        st.session_state.messages.append({
            "role": "assistant",
            "content": [
                (
                    "dataframe",
                    df
                )
            ]
        })
    else:
        st.warning("No itinerary data found.")

# Function to process ALNS data
def process_alns_data():
    alns_data = st.session_state["alns_data"]
    if "trip_summary" in alns_data:
        st.subheader("Trip Summary")
        st.json(alns_data["trip_summary"])
        st.session_state.messages.append({
            "role": "assistant",
            "content": [
                (
                    "markdown",
                    "## Trip Summary"
                ),
                (
                    "json",
                    alns_data["trip_summary"]
                )
            ]
        })

    if "attractions_visited" in alns_data:
        st.subheader("Attractions Visited")
        st.write(", ".join(alns_data["attractions_visited"]))
        st.session_state.messages.append({
            "role": "assistant",
            "content": [
                (
                    "markdown",
                    "## Attractions Visited"
                ),
                (
                    "markdown",
                    ", ".join(alns_data["attractions_visited"])
                )
            ]
        })

    if "budget_breakdown" in alns_data:
        st.subheader("Budget Breakdown")
        budget_df = pd.DataFrame(list(alns_data["budget_breakdown"].items()), columns=["Category", "Cost"])
        st.table(budget_df)
        st.session_state.messages.append({
            "role": "assistant",
            "content": [
                (
                    "markdown",
                    "## Budget Breakdown"
                ),
                (
                    "dataframe",
                    budget_df
                )
            ]
        })

    if "transport_summary" in alns_data:
        st.subheader("Transport Summary")
        st.json(alns_data["transport_summary"])
        st.session_state.messages.append({
            "role": "assistant",
            "content": [
                (
                    "markdown",
                    "## Transport Summary"
                ),
                (
                    "json",
                    alns_data["transport_summary"]
                )
            ]
        })

    if "rest_summary" in alns_data:
        st.subheader("Rest Summary")
        st.json(alns_data["rest_summary"])
        st.session_state.messages.append({
            "role": "assistant",
            "content": [
                (
                    "markdown",
                    "## Rest Summary"
                ),
                (
                    "json",
                    alns_data["rest_summary"]
                )
            ]
        })

#==================

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

def display_map():
    """Display locations on a map using Folium."""
    route_map = st.session_state["route_map"]
    if route_map:
        folium_static(route_map)
    else:
        st.warning("No location data available for mapping.")

def prepare_map(alns_data):
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
    
    return m


# Function to create map
def generate_itinerary(user_input):
    start_time = time.time()
    # exports out the data/POI_data.json based on the given query from streamlit otherwise, its a default "how to make a bomb"
    # get_json_from_query(query=user_input['description'], debug_mode=True)

    # aggregation between kb and recommendations, deduplicates, and randomnisation
    get_combine_json_data()

    multiagent_time = time.time()
    multiagent_runtime = multiagent_time - start_time
    logger.info(f"Multi-Agent runs for {multiagent_runtime:.2f} s")

    alns_data = alns_main(user_input=user_input, 
                        #   alns_input=alns_input,
                          llm_path="./data/alns_inputs/")
    alns_time = time.time()
    alns_runtime = alns_time - multiagent_time
    total_runtime = alns_time - start_time

    logger.info(f"ALNS runs for {alns_runtime:.2f} s")
    logger.info(f"MultiAgent + ALNS Solution runs for {total_runtime:.2f} s")
    logger.info("Itinerary data loaded successfully!")

    m = prepare_map(alns_data)

    st.session_state["itinerary_ready"] = True
    st.session_state["route_map"], st.session_state["alns_data"] = m, alns_data


def make_content_to_string(content):
    content_type, content_value = content

    if content_type == "dataframe":
        return content_value.to_string()
    elif content_type == "markdown":
        return str(content_value)
    elif content_type == "json":
        return json.dumps(content_value, indent=4)
    
    raise ValueError(f"Content Type '{content_type}' not handled.")


def update_itinerary(user_input, feedback_prompt, itinerary_table, approach=0):
    logger.info("Itinerary Table:")
    logger.info(itinerary_table)
    logger.info(f"Feedback Prompt: {feedback_prompt}")

    if approach == 0:
        # 2B: using LLM only, try to update the itinerary in the same table
        updated_days = update_itinerary_llm(st.session_state["alns_data"], feedback_prompt)
        updated_itinerary = rebuild_full_itinerary(updated_days, st.session_state["alns_data"]).model_dump()
        logger.info(f"Updated Itinerary: {updated_itinerary}")

    else: # 2C and 2D
        # 2C: find affected POIs, find alternatives of the POIs
        #     update the itinerary afterwards
        poi_suggestions = find_alternative_of_affected_pois(itinerary_table, feedback_prompt, top_n=5)

        logger.info("Affected POIs and their Alternatives:")
        logger.info(json.dumps(poi_suggestions, indent=4, default=str))

        updated_days = update_itinerary_closest_alternatives(st.session_state["alns_data"], feedback_prompt, poi_suggestions)
        updated_itinerary = rebuild_full_itinerary(updated_days, st.session_state["alns_data"]).model_dump()

        if approach == 2: # use 2D on top of 2C result
            # TODO
            # 2D: Using the updated itinerary, try to check feasibility, rerun if not feasible.
            pass


    ### PLACEHOLDER
    # alns_data = alns_main(user_input=user_input, llm_path="./data/alns_inputs/")
    logger.info("Itinerary data loaded successfully!")
    # m = prepare_map(alns_data)
    m = prepare_map(updated_itinerary)
    
    print(feedback_prompt)
    st.session_state["itinerary_ready"] = True
    # st.session_state["route_map"], st.session_state["alns_data"] = m, alns_data
    st.session_state["route_map"], st.session_state["alns_data"] = m, updated_itinerary

G = load_graph()

# State to track whether itinerary has been generated
if "itinerary_ready" not in st.session_state:
    st.session_state["itinerary_ready"] = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# User Inputs
st.sidebar.header("Trip Inputs")
num_days = st.sidebar.number_input("Number of Days (1-5)", min_value=1, max_value=5, value=3)
budget = st.sidebar.number_input("Budget", min_value=100, value=500)
description = st.sidebar.text_area("Trip Description", "Your trip description here...")

st.title("My Intelligent Travel Buddy – Automatic Itinerary (MITB – AI）- Singapore Edition")

st.sidebar.header("Navigation")
page = "Itinerary" 

# State to track whether itinerary has been generated
if "itinerary_ready" not in st.session_state:
    st.session_state["itinerary_ready"] = False
if "alns_data" not in st.session_state:
    st.session_state["alns_data"] = None
if "route_map" not in st.session_state:
    st.session_state["route_map"] = None

if page == "Itinerary":
    st.header("Your Optimized Itinerary")
    if st.button("Generate Itinerary"):
        user_input = {
            "num_days": num_days,
            "budget": budget,
            "description": description
        }
        generate_itinerary(user_input)
    
    if st.session_state["itinerary_ready"]:
        display_itinerary()
        page = st.sidebar.radio("Go to", ["Itinerary", "Trip Details", "Map", "About"])

if st.session_state["itinerary_ready"]:
    if page == "Trip Details":
        st.header("Detailed Trip Insights")
        process_alns_data()
    elif page == "Map":
        st.header("Map of Your Itinerary")
        display_map()
    elif page == "About":
        st.header("About")
        st.write("This app helps plan your optimized travel itineraries using ALNS algorithms.")

    # when itinerary is ready, have the feedback loop
    feedback_prompt = st.chat_input("Any feedback about the result?")
    if feedback_prompt:
        st.chat_message("user").markdown(feedback_prompt)
        st.session_state.messages.append({
            "role": "user",
            "content": [("markdown", feedback_prompt)]
        })
        user_input = { # for alns, regardless of feedback
            "num_days": num_days,
            "budget": budget,
            "description": description
        }

        # get last tabular itinerary in messages
        itinerary_table = None
        for message in st.session_state.messages[::-1]:
            for content in message["content"][::-1]:
                if content[0] == "dataframe":
                    itinerary_table = content[1]
                    logging.info("masuk sini")
                    break

        update_itinerary(user_input, feedback_prompt, itinerary_table, approach=1) # only taking the last itinerary
        if st.session_state["itinerary_ready"]:
            display_itinerary()
            # no need to re-show radio "Go to", will have duplicate otherwise

st.sidebar.write("---")
st.sidebar.write("Developed with ❤️ using Streamlit.")
