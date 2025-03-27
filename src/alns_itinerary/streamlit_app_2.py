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

import json
import pandas as pd
import random

from alns_main import alns_main
from data.llm_batch_process import process_and_save

#==================
from pydantic import BaseModel, Field
from typing import List
import time

import json
from dotenv import load_dotenv
import os
import random

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq

from agno.document.chunking.fixed import FixedSizeChunking
from agno.document.chunking.agentic import AgenticChunking

from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.googlesearch import GoogleSearchTools

from agno.knowledge.csv import CSVKnowledgeBase
from agno.vectordb.pgvector import PgVector

# ========================================================
# Load environment variables & classess for Pydantic Base Models
# ========================================================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

class IntentResponse(BaseModel):
    intent: str = Field(..., description="The detected intent of the query. Options: 'food', 'attraction', 'both'. Returns 'malicious' if the query is malicious.")

class HawkerRecommendation(BaseModel):
    hawker_name: str = Field(..., description="The name of the Hawker Centre.")
    dish_name: str = Field(..., description="The name of the dish that is recommended.")
    description: str = Field(..., description="A short description of the dish and why it's recommended.")
    average_price: float = Field(..., description="The maximum price in SGD of the dish, retrieved from web sources.")
    ratings: float = Field(..., description="The Google rating of the Hawker Centre, range from 1 to 5.")
    satisfaction_score: float = Field(..., description="The Satisfaction Score after unstanding the travller preference and the Google rating of the Hawker Centre, range from 1 to 5.")
    sources: List[str] = Field(..., description="List of sources where information was retrieved.")

class HawkerResponse(BaseModel):
    # QUERY: str = Field(..., description="The user's original query for hawker recommendations.")
    HAWKER_RECOMMENDATIONS: List[HawkerRecommendation] = Field(..., description="List of recommended hawker food options.")

class AttractionRecommendation(BaseModel):
    attraction_name: str = Field(..., description="The name of the attraction that is recommended.")
    description: str = Field(..., description="A short description of the attraction and why it's recommended.")
    average_price: float = Field(..., description="The maximum price in SGD of the attraction, retrieved from web sources.")
    ratings: float = Field(..., description="The Google rating of the attraction, range from 1 to 5.")
    satisfaction_score: float = Field(..., description="The Satisfaction Score after unstanding the travller preference and the Google rating of the Hawker Centre, range from 1 to 5.")
    sources: List[str] = Field(..., description="List of sources where information was retrieved.")

class AttractionResponse(BaseModel):
    # QUERY: str = Field(..., description="The user's original query for attraction recommendations.") 
    ATTRACTION_RECOMMENDATIONS: List[AttractionRecommendation] = Field(..., description="List of recommended attraction options.")

#==================
# mutli agent part

def get_preference_kb():
    preference_kb = CSVKnowledgeBase(
        path="data/locationData/csv/",
        # Table name: ai.csv_documents
        vector_db=PgVector(
            table_name="sg_attraction_hawker",
            db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        ),
    )
    return preference_kb

def get_hawker_kb():
    pdf_urls = [
        # "https://raw.githubusercontent.com/Mingtaros/Travel-Planner-Decision-Maker/main/data/hawker/Summary_Singapore_Food.pdf",
        # "https://raw.githubusercontent.com/Mingtaros/Travel-Planner-Decision-Maker/main/data/hawker/inputs/hawker_centres_singapore.pdf",
        "https://raw.githubusercontent.com/Mingtaros/Travel-Planner-Decision-Maker/main/data/locationData/20_hawker.pdf"]

    hawker_chunking_type = FixedSizeChunking(chunk_size=150, overlap=20)

    chroma_db_path = "./chromadb_data"
    hawker_collection_name = "HAWKER_fixedchunk_final" #depends on hawker chunking type and name appropriately
    hawker_db = ChromaDb(
        collection=hawker_collection_name, 
        path=chroma_db_path,
        persistent_client=True   # Enable persistence
    )

    hawker_kb = PDFUrlKnowledgeBase(urls=pdf_urls,
                                    chunking_strategy=hawker_chunking_type,
                                    vector_db=hawker_db,
                                    )
    
    return hawker_kb

def get_attraction_kb():
    pdf_urls = [
        # "https://raw.githubusercontent.com/Mingtaros/Travel-Planner-Decision-Maker/main/data/hawker/Summary_Singapore_Food.pdf",
        # "https://raw.githubusercontent.com/Mingtaros/Travel-Planner-Decision-Maker/main/data/attraction/inputs/Singapore_Attractions_Guide.pdf"
        "https://raw.githubusercontent.com/Mingtaros/Travel-Planner-Decision-Maker/main/data/locationData/67_attractions.pdf"

    ]

    attraction_chunking_type = FixedSizeChunking(chunk_size=350, overlap=50)

    chroma_db_path = "./chromadb_data"
    attraction_collection_name = "ATTRACTION_fixedchunk_final" #depends on hawker chunking type and name appropriately
    attraction_db = ChromaDb(
        collection=attraction_collection_name, 
        path=chroma_db_path,
        persistent_client=True   # Enable persistence
    )

    attraction_kb = PDFUrlKnowledgeBase(urls=pdf_urls,
                                    chunking_strategy=attraction_chunking_type,
                                    vector_db=attraction_db,
                                    )
    return attraction_kb

def create_preference_agent():
    csv_kb = get_preference_kb()
    preference_agent = Agent(
            name="Satisfaction Suitability Agent",
            model=OpenAIChat(
                id="gpt-4o",  # or any model you prefer
                response_format="json", # depends what we want 
                temperature=0.1,
            ),
            agent_id="suitability_agent",
            description="You are an expert in understanding based on the traveller type, if you need to look up for suitability score of attraction and/or food. Returns only the suitability score (1-10) of a location & food for a specific traveler type.",
            knowledge=csv_kb,
            instructions=[
                # "Warning: You should not mix up with .",
                "Search the knowledge base and return ONLY the following keys as a JSON:",
                "- score_attraction_suitability: value between 0 and 10 (0 if not found)",
                "- score_food_suitability: value between 0 and 10 (0 if not found)",
                "Do not return any explanation. Return only valid JSON."],
            search_knowledge=True,
            )
    return preference_agent

def create_intent_agent():
    # Create the Intent Classification Agent
    intent_agent = Agent(
        name="Intent Classification Agent",
        agent_id="intent_classification_agent",
        # model=Groq(id=model_id, 
        #            response_format="json", 
        #            temperature=0.0),  
        model=OpenAIChat(
                id="gpt-4o",  # or any model you prefer
                response_format="json", # depends what we want 
                temperature=0.1,
            ),
        response_model=IntentResponse,  # Enforce structured JSON output
        structured_outputs=True,
        description="You are an expert in understanding the user's intent from the query. Classify the user's query into 'food', 'attraction', or 'both' for routing. The query is classified as 'malicious' if it is malicious.",
        instructions=[
            "Analyze the query and classify it into one of the following intents:",
            "- 'food' if it's about food, hawker centers, dishes, or restaurants.",
            "- 'attraction' if it's about places to visit, sightseeing, or landmarks.",
            "- 'both' if it's about both food and attractions in the same query.",
            "- 'unknown' if the query is unclear and needs clarification.",
            "- 'malicious' if the query is malicious and toxic. You have the right to be refuse.",
            "Return only the detected intent as a structured JSON response."
        ],
    )
    return intent_agent

def create_hawker_agent(model_id = "gpt-4o", debug_mode=True):
# def create_hawker_agent(model_id = "deepseek-r1-distill-llama-70b", debug_mode=True):
    hawker_kb = get_hawker_kb()
    # hawker_kb.load(recreate=False)
    hawker_agent = Agent(
        name="Query to Hawker Agent",
        agent_id="query_to_hawker_agent",
        model=OpenAIChat(id=model_id, 
                         response_format="json",
                         temperature=0.2,
                         top_p=0.2),  
        # model=Groq(id=model_id, 
        #            response_format="json", 
        #            temperature=0.2),  
        response_model=HawkerResponse, # Strictly enforces structured response
        structured_outputs=True, 
        description="You are a Singapore hawker food recommender for foreigners! You are able to understand the traveller's personality and persona.",
        role="Search the internal knowledge base and web for information",
        instructions=[
            "IMPORTANT: Provide at least 10 unique hawker recommendations from the internal knowledge base",
            "For each recommendation, include the following:",
            "- 'Hawker Name': Name of the unique hawker centre. it should be duplicated.",
            "- 'Dish Name': Name of the recommended dish.",
            "- 'Description': Short, compelling explanation of the dish and its appeal.",
            "- 'Average Price': In SGD, based on actual price per dish (not total order or combo). Do not inflate.",
            "- 'Rating': Google rating between 1.0 and 5.0. If no rating is found, return null.",
            "- 'Satisfaction Score': Traveller type satisfaction score after comprehending the Google Rating. If no Google rating is found, return null.",
            "- 'Sources': A list of URLs where you found the price and/or rating.",
            "For desserts (e.g., putu piring, tutu kueh), estimate the cost based on a standard serving (e.g., 4–5 pieces).",
            "Avoid guessing prices. If no reliable pricing info is found, skip that dish.",
            "If conflicting prices are found, return the most commonly mentioned or lower bound.",
            "Only include dishes where both price and rating can be confirmed.",
            "IMPORTANT: always include 1-2 more hawkers places that you have not selected from the internal knowledge base."
        ],
        knowledge=hawker_kb,
        search_knowledge=True,

        tools=[DuckDuckGoTools(search=True,
                            # news=False,
                            fixed_max_results=3)],
        show_tool_calls=True,
        debug_mode=debug_mode,  # Comment if not needed - added to see the granularity for debug like retrieved context from vectodb
        markdown=True,
        # add_references=True, # enable RAG by adding references from AgentKnowledge to the user prompt.
    )
    return hawker_agent

def create_attraction_agent(model_id = "gpt-4o", debug_mode=True):
# def create_attraction_agent(model_id = "deepseek-r1-distill-llama-70b", debug_mode=True):
    attraction_kb = get_attraction_kb()
    # attraction_kb.load(recreate=False)
    attraction_agent = Agent(
        name="Query to Attraction Agent",
        agent_id="query_to_attraction_agent",
        model=OpenAIChat(id=model_id, 
                         response_format="json",
                         temperature=0.2,top_p=0.2
                         ), 
        # model=Groq(id=model_id, 
        #            response_format="json", 
        #            temperature=0.2), 
        response_model=AttractionResponse, # Strictly enforces structured response
        structured_outputs=True, 
        description="You are a Singapore Attraction recommender for foreigners! You are able to understand the traveller's personality and persona.",
        role="Search the internal knowledge base",
        instructions=[
            "IMPORTANT: Provide at least 30 unique attraction recommendations from the knowledge base.",
            "For each attraction, include the following:",
            "- 'Attraction Name'",
            "- 'Description' (why it is recommended, who it is suited for)",
            "- 'Entrance Fee' (in SGD). If it is free, return 0. If not, retrieve the adult entrance fee from an official or trusted source. Do not guess.",
            "- 'Rating' between 1 and 5 (preferably Google rating or TripAdvisor). If not found, return null.",
            "- 'Satisfaction Score': Traveller type satisfaction score after comprehending the Google Rating. If no Google rating is found, return null.",
            "- 'Duration' of visit, which is typically around 2 hours unless otherwise stated.",
            "- 'Sources' (a list of URLs where you found the entrance fee or rating).",
            "If an attraction's entrance fee or rating cannot be verified, skip that attraction and replace it with another one.",
            "Do not invent prices. Use only information retrieved from web search or internal PDF documents."
        ],
        knowledge=attraction_kb,
        search_knowledge=True,

        tools=[DuckDuckGoTools(search=True,
                            # news=True,
                            fixed_max_results=3),
            GoogleSearchTools()],
        show_tool_calls=True,
        debug_mode=debug_mode,  # Comment if not needed - added to see the granularity for debug like retrieved context from vectodb
        markdown=True,    
        # add_references=True, # enable RAG by adding references from AgentKnowledge to the user prompt.

    )
    return attraction_agent

def get_combine_json_data(path = "./data/POI_data.json", at_least_hawker = 10, at_least_attraction = 30):
    # Read the JSON file
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    ### This is for Hawker
    hawker_names_llm = [entry['Hawker Name'] for entry in data["Hawker"]]
    df_h = pd.read_csv("./data/singapore_20_food_with_scores.csv")
    hawker_names_kb = df_h["Hawker Name"].to_list()
    filtered_hawker_names = [name for name in hawker_names_llm if name in hawker_names_kb]
    remaining_hawkers = [name for name in hawker_names_kb if name not in filtered_hawker_names]
    num_to_take_hawker = at_least_hawker - len(filtered_hawker_names)
    print(num_to_take_hawker)
    sampled_hawkers = random.sample(remaining_hawkers, k=min(num_to_take_hawker, len(remaining_hawkers)))
    filtered_rows_h = df_h[df_h['Hawker Name'].isin(sampled_hawkers)]

    # Step 2: Convert to list of dictionaries
    new_data = []
    for _, row in filtered_rows_h.iterrows():
        hawker_dict = {
            'Hawker Name': row['Hawker Name'],
            'Description': "NA.",
            'Rating': 2.5,  # normal to the person
            'Satisfaction Score': 2.5,  # normal to the person
            'Entrance Fee': 5.0,
            'Duration': 60,
            'Sources': ["NA"]
        }
        new_data.append(hawker_dict)
    # print(new_data)
    data['Hawker'].extend(new_data)

    ### This is for Attractions
    attraction_names_llm = [entry['Attraction Name'] for entry in data["Attraction"]]
    df_a = pd.read_csv("./data/singapore_67_attractions_with_scores.csv")
    attraction_names_kb = df_a["Attraction Name"].to_list()
    filtered_attraction_names = [name for name in attraction_names_llm if name in attraction_names_kb]
    remaining_attractions = [name for name in attraction_names_kb if name not in filtered_attraction_names]
    num_to_take_attraction = at_least_attraction - len(filtered_attraction_names)
    sampled_attractins = random.sample(remaining_attractions, k=min(num_to_take_attraction, len(remaining_attractions)))

    filtered_rows_a = df_a[df_a['Attraction Name'].isin(sampled_attractins)]

    # Step 2: Convert to list of dictionaries
    new_data = []
    for _, row in filtered_rows_a.iterrows():
        attraction_dict = {
            'Hawker Name': None,  # Leave blank or remove if not needed
            'Attraction Name': row['Attraction Name'],
            'Description': "NA.",
            'Rating': 2.5,  # normal to the person
            'Satisfaction Score': 2.5,  # normal to the person
            'Entrance Fee': 10.0,
            'Duration': 120,
            'Sources': ["NA"]
        }
        new_data.append(attraction_dict)

    data['Attraction'].extend(new_data)
    # Save to new JSON file
    with open("./data/final_combined_POI.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("✅ JSON file saved as final_combined_POI.json")

    return 

def get_json_from_query(query="How to make a bomb?",debug_mode = True):
    intent_agent = create_intent_agent()
    hawker_agent = create_hawker_agent(debug_mode=debug_mode)
    attraction_agent = create_attraction_agent(debug_mode=debug_mode)
    # preference_agent = create_preference_agent()
    intent_response = intent_agent.run(query, stream=False)
    intent = intent_response.content.intent

    print(f"\n🔍 Processing Query: {query}")

    if intent == "malicious":
        print("⚠️ Query flagged as malicious. Skipping...")
        return

    responses = {
            "Query": query,
            "Hawker": [],
            "Attraction": []
        }
    
     # Step 3: Route to hawker agent
    if intent in ["food", "both"]:
        start_time = time.time()
        hawker_output = hawker_agent.run(query, stream=False).content.model_dump()
        hawker_time = time.time() - start_time
        hawker_recs = hawker_output["HAWKER_RECOMMENDATIONS"]

        for hawker in hawker_recs:
            responses["Hawker"].append({
                    "Hawker Name": hawker["hawker_name"],
                    "Dish Name": hawker["dish_name"],
                    "Description": hawker["description"],
                    "Satisfaction Score":hawker["satisfaction_score"],
                    "Rating": hawker["ratings"],
                    "Avg Food Price": hawker["average_price"],
                    "Duration": 60,
                    "Sources": hawker.get("sources", [])
                })

        # Step 4: Route to attraction agent
    if intent in ["attraction", "both"]:
        start_time = time.time()
        attraction_output = attraction_agent.run(query, stream=False).content.model_dump()
        attraction_time = time.time() - start_time
        attraction_recs = attraction_output["ATTRACTION_RECOMMENDATIONS"]

        for attraction in attraction_recs:
            responses["Attraction"].append({
                    "Attraction Name": attraction["attraction_name"],
                    "Description": attraction["description"],
                    "Rating": attraction["ratings"],
                    "Satisfaction Score":attraction["satisfaction_score"],
                    "Entrance Fee": attraction["average_price"],
                    "Duration": 120,
                    "Sources": attraction.get("sources", [])
                })
            # responses["Metrics"] = {
            #                             # "Intent Agent Time (s)": round(intent_time, 2),
            #                         "Hawker Agent Time (s)": round(hawker_time or 0, 2),
            #                         "Attraction Agent Time (s)": round(attraction_time or 0, 2),
            #                             # "Intent Agent Tokens": intent_usage if intent_usage else {},
            #                             # "Hawker Agent Tokens": hawker_usage if hawker_usage else {},
            #                             # "Attraction Agent Tokens": attraction_usage if attraction_usage else {}
            #                         }

        # Step 5: Prepare hardcoded MOO parameters
    moo_params = {
            "Budget": 100,
            "Number of days": 3,
            "params": [0.3, 0.3, 0.4]
        }
    
    query_num = "special"
    # Step 6: Create subfolder based on query number
    subfolder_path = os.path.join("data/alns_inputs", f"{query_num}")
    os.makedirs(subfolder_path, exist_ok=True)

    poi_path = os.path.join(subfolder_path, "POI_data.json")
    moo_path = os.path.join(subfolder_path, "moo_parameters.json")

    with open(poi_path, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=4)

    with open(moo_path, "w", encoding="utf-8") as f:
        json.dump(moo_params, f, indent=4)

    print(f"✅ Saved to: {subfolder_path}")

    return


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

# Function to create map
def create_map(user_input):
    ##### TAI ADD HERE

    # exports out the data/POI_data.json based on the given query from streamlit otherwise, its a default "how to make a bomb"
    get_json_from_query(query=user_input['description'],debug_mode = True)

    # aggregation between kb and recommendations, deduplicates, and randomnisation
    get_combine_json_data()
    
    alns_input = None
    # alns_input = process_and_save(
    #     persona=user_input['persona'],
    #     attraction_path="./data/locationData/singapore_67_attractions_with_scores.csv", 
    #     hawker_path="./data/locationData/Food_20_withscores.xlsx", 
    #     output_json_path="./data/alns_inputs/groq/location_data.json", 
    #     batch_size=10
    # )
    
    ##### TAI END HERE
    alns_data = alns_main(user_input=user_input, alns_input=alns_input)

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

G = load_graph()

st.title("My Intelligent Travel Buddy – Automatic Itinerary (MITB – AI）- Singapore Edition")
st.sidebar.header("Trip Details")

persona = st.sidebar.selectbox("Choose your persona", [
    "Family Tourist", "Backpacker", "Influencer", "Cultural Enthusiast", 
    "Thrill Seeker", "Nature Lover", "Shopping Enthusiast"
])
num_days = st.sidebar.number_input("Number of Days (1-5)", min_value=1, max_value=5, value=3)
budget = st.sidebar.number_input("Budget", min_value=200, value=500)
description = st.sidebar.text_area("Trip Description", "Your trip description here...")

if st.sidebar.button("Generate Itinerary"):
    user_input = {"persona": persona, "num_days": num_days,
        "budget": budget, "description": description}
    json_path = "../../user_input.json"
    with open(json_path, 'w') as f:
        json.dump(user_input, f, indent=4)
    st.sidebar.success(f"Data saved to {json_path}")
    
    # Create and display map
    m, alns_data = create_map(user_input)
    folium_static(m)
    
    # Display detailed overview
    display_detailed_overview(alns_data)
    
    st.success("Map loaded successfully!")
    logger.info("Map loaded successfully!")

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
# if st.button("Show Trip Map"):
#     # Create and display map
#     m, alns_data = create_map()
#     folium_static(m)
    
#     # Display detailed overview
#     display_detailed_overview(alns_data)
    
#     st.success("Map loaded successfully!")
#     logger.info("Map loaded successfully!")