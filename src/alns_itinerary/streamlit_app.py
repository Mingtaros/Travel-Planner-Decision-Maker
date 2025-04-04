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
from data.llm_batch_process import process_and_save

#==================
from pydantic import BaseModel, Field
from textwrap import dedent
from typing import List
import time

import json
from dotenv import load_dotenv
import os
import random

from agno.agent import Agent
from agno.models.openai import OpenAIChat
# from agno.models.groq import Groq

from agno.document.chunking.fixed import FixedSizeChunking
from agno.document.chunking.agentic import AgenticChunking

from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.googlesearch import GoogleSearchTools

from agno.knowledge.csv import CSVKnowledgeBase
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.pgvector import PgVector

# ========================================================
# Load environment variables & classess for Pydantic Base Models
# ========================================================
load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

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

class IntentResponse(BaseModel):
    intent: str = Field(..., description="The detected intent of the query. Options: 'food', 'attraction', 'both'. Returns 'malicious' if the query is malicious.")

class VariableResponse(BaseModel):
    alns_weights: List[float]

class HawkerDetail(BaseModel):
    hawker_name: str = Field(..., description="The name of the Hawker Centre. The name must match the one from DB.")
    dish_name: str = Field(..., description="The specific dish name offered in this Hawker Centre that is recommended.")
    average_price: float = Field(..., description="The maximum price in SGD of the dish, retrieved from web sources. If unavailable, make a guess.")
    satisfaction_score: float = Field(..., description="The Satisfaction Score that the traveler would get from coming to this hawker. Ranges from 1 to 5. 1 being the least satisfactory for this traveler, and 5 being the most satisfactory. Pick a number that most suited the traveler's persona.")
    duration: int = Field(..., description="The average duration of eating in this hawker centre IN MINUTES, retrieved from web sources. If unavailable, make an approximation.")

class HawkerResponse(BaseModel):
    HAWKER_DETAILS: List[HawkerDetail] = Field(..., description="List of detailed hawker food options.")

class AttractionDetail(BaseModel):
    attraction_name: str = Field(..., description="The name of the attraction. The name must match the one from DB.")
    average_price: float = Field(..., description="The maximum price in SGD of the attraction, retrieved from web sources. If unavailable, make a guess. If free, return 0.")
    satisfaction_score: float = Field(..., description="The Satisfaction Score that the traveler would get from coming to this attraction. Ranges from 1 to 5. 1 being the least satisfactory for this traveler, and 5 being the most satisfactory. Pick a number that most suited the traveler's persona.")
    duration: int = Field(..., description="The estimated duration the traveller would spend in this place IN MINUTES, retrieved from web sources. If unavailable, make an approximation.")

class AttractionResponse(BaseModel):
    ATTRACTION_DETAILS: List[AttractionDetail] = Field(..., description="List of detailed attraction options.")

class CodeResponse(BaseModel):
    is_feasible: List[str] = Field(..., description="List of additional constraints to add to the `is_feasible function` of VRPSolution.")
    is_feasible_insertion: List[str] = Field(..., description="List of additional constraints to add to the `is_feasible_insertion` function of VRPSolution.")

#==================
# mutli agent part

def get_hawker_kb(batch_no):
    hawker_kb = CSVKnowledgeBase(
        path=f"data/locationData/csv/hawkers/{batch_no}",
        vector_db=PgVector(
            table_name=f"sg_hawkers_{batch_no}",
            db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        ),
    )
    
    return hawker_kb

def get_attraction_kb(batch_no):
    attraction_kb = CSVKnowledgeBase(
        path=f"data/locationData/csv/attractions/{batch_no}",
        vector_db=PgVector(
            table_name=f"sg_attractions_{batch_no}",
            db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        ),
    )

    return attraction_kb

def get_vrp_code_kb():
    code_kb = TextKnowledgeBase(
        path=f"src/alns_itinerary/alns",
        formats=[".py"],
        vector_db=PgVector(
            table_name="vrp_alns_codes",
            db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        ),
        num_documents=5,
    )

    return code_kb

def create_variable_agent():
    # Create the Variable Extraction Agent
    variable_agent = Agent(
        name="Travel Variable Extractor",
        agent_id="variable_extraction_agent",
        model=OpenAIChat(
            id="gpt-4o",  
            response_format="json",
            temperature=0.1,
        ),
        # model=Groq(
        #     id="deepseek-r1-distill-llama-70b",
        #     response_format={ "type": "json_object" },
        #     temperature=0.2
        # ),
        response_model=VariableResponse,  # Ensure structured output matches the schema
        description="You are an expert in optimized itinerary planning. Your task is to generate weights for the Adaptive Large Neighborhood Search (ALNS) algorithm. These weights will help in optimizing travel itineraries based on a user's persona.",
        instructions=dedent("""\
        Your response must strictly follow this JSON format:
        {
            "alns_weights": {
                "budget_priority": <weight_value>,
                "time_priority": <weight_value>,
                "satisfaction_priority": <weight_value>
            }
        }
        """)
    )
    return variable_agent

def create_intent_agent():
    # Create the Intent Classification Agent
    intent_agent = Agent(
        name="Intent Classification Agent",
        agent_id="intent_classification_agent",
        model=OpenAIChat(
                id="gpt-4o",  # or any model you prefer
                response_format="json", # depends what we want 
                temperature=0.1,
            ),
        # model=Groq(id="deepseek-r1-distill-llama-70b", 
        #            response_format={ "type": "json_object" }, 
        #            temperature=0.0),  
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

def create_hawker_agent(model_id = "gpt-4o", batch_no=0, debug_mode=True):
# Create the Hawker Agent to get all the relevant Hawkers/POIs based on the routed query from Supervisor Agent
# def create_hawker_agent(model_id="deepseek-r1-distill-llama-70b", batch_no=0, debug_mode=True):
    hawker_kb = get_hawker_kb(batch_no)
    hawker_kb.load(recreate=False)
    hawker_agent = Agent(
        name="Query to Hawker Agent",
        agent_id="query_to_hawker_agent",
        model=OpenAIChat(id=model_id, 
                         response_format="json",
                         temperature=0.2,
                         top_p=0.2),  
        # model=Groq(id=model_id, 
        #            response_format={ "type": "json_object" }, 
        #            temperature=0.2),  
        response_model=HawkerResponse, # Strictly enforces structured response
        structured_outputs=True, 
        description="You are a Singapore Hawker food expert! You are able to understand the traveller's personality and persona.",
        role="Search the internal knowledge base and web for information",
        instructions=[
            "IMPORTANT: Provide details on all of the hawker centres from the internal knowledge base",
            "For each hawker, include the following:",
            "- 'hawker_name': Name of the unique hawker centre. Only use the hawker names in internal knowledge.",
            "- 'dish_name': Name of the specific dish from this hawker centre that you recommend the traveller to try.",
            "- 'average_price': In SGD, based on actual price per dish (not total order or combo). Do not inflate.",
            "- 'satisfaction_score': Traveller type satisfaction score.",
            "- 'duration': duration of visit in minutes, retrieve this information from a trusted source. If unavailable, lease estimate the amount of duration required based on the tourist's personality and preference.",
            "If conflicting prices are found, return the most commonly mentioned or lower bound.",
            "If the price per dish isn't available, try to find price per person.",
            "Try your best to use ONLY information retrieved from web search or internal knowledge base.",
            "Return the output in List of JSON format. Do not provide any summaries, analyses, or other additional content."
        ],
        knowledge=hawker_kb,
        search_knowledge=True,
        tools=[DuckDuckGoTools(search=True,
                            # news=False,
                            fixed_max_results=3,)],
        show_tool_calls=True,
        debug_mode=debug_mode,  # Comment if not needed - added to see the granularity for debug like retrieved context from vectodb
        markdown=True,
        # add_references=True, # enable RAG by adding references from AgentKnowledge to the user prompt.
    )
    return hawker_agent

def create_attraction_agent(model_id = "gpt-4o", batch_no=0, debug_mode=True):
# def create_attraction_agent(model_id="deepseek-r1-distill-llama-70b", batch_no=0, debug_mode=True):
    attraction_kb = get_attraction_kb(batch_no)
    attraction_kb.load(recreate=False)
    attraction_agent = Agent(
        name="Query to Attraction Agent",
        agent_id="query_to_attraction_agent",
        model=OpenAIChat(id=model_id, 
                         response_format="json",
                         temperature=0.2,top_p=0.2
                         ), 
        # model=Groq(id=model_id, 
        #            response_format={ "type": "json_object" }, 
        #            temperature=0.2), 
        response_model=AttractionResponse, # Strictly enforces structured response
        structured_outputs=True, 
        description="You are a Singapore Attraction expert! You are able to understand the traveller's personality and persona.",
        role="Search the internal knowledge base",
        instructions=[
            "IMPORTANT: Provide details on all of the attractions from the knowledge base.",
            "For each attraction, include the following:",
            "- 'attraction_name': Name of the attraction. Only use attraction names in the internal knowledge.",
            "- 'average_price': Entrance Fee (in SGD). If it is free, return 0. If not, retrieve the adult entrance fee from an official or trusted source. Do not guess.",
            "- 'satisfaction_score': Traveller type satisfaction score after comprehending the Google Rating. If no Google rating is found, return null.",
            "- 'duration': duration of visit in minutes, retrieve this information from a trusted source. If unavailable, lease estimate the amount of duration required based on the tourist's personality and preference.",
            "If an attraction's entrance fee or rating cannot be verified, use a guess from similar attractions.",
            "Try your best to use ONLY information retrieved from web search or internal knowledge base.",
            "Return the output in List of JSON format. Do not provide any summaries, analyses, or other additional content."
        ],
        knowledge=attraction_kb,
        search_knowledge=True,
        tools=[DuckDuckGoTools(search=True,
                            # news=True,
                            fixed_max_results=3)],
        show_tool_calls=True,
        debug_mode=debug_mode,  # Comment if not needed - added to see the granularity for debug like retrieved context from vectodb
        markdown=True,
        # add_references=True, # enable RAG by adding references from AgentKnowledge to the user prompt.
    )
    return attraction_agent

def create_code_agent(model_id="gpt-4o", debug_mode=True):
    code_kb = get_vrp_code_kb()
    code_kb.load(recreate=False)
    code_agent = Agent(
        name="Query to Code Agent",
        agent_id="query_to_code_agent",
        model=OpenAIChat(id=model_id, 
                         response_format="json",
                         temperature=0.2,top_p=0.2
                         ), 
        # model=Groq(id=model_id, 
        #            response_format={ "type": "json_object" }, 
        #            temperature=0.2), 
        response_model=CodeResponse,
        structured_outputs=True,
        description="You are an OR engineer programming ALNS. You are able to understand the traveller's persona and unique needs and translate it into a code.",
        role="Reference the internal knowledge base and make the necessary constraints.",
        instructions=[
            "IMPORTANT: Read the code for VRPSolution. This is an ALNS implementation of the travel itinerary problem.",
            "In VRPSolution class, look for the `is_feasible` and `is_feasible_insertion` function.",
            "Based on the traveller's persona and description, make new constraints to add to these functions if necessary.",
            "ONLY add the code pieces if it's necessary. If it's not needed, return an empty list for both `is_feasible` and `is_feasible_insertion` function.",
            "For `is_feasible` function, look for a comment line `# <ADD NEW FEASIBILITY CHECK HERE>`. That's where to put the code.",
            "For `is_feasible_insertion` function, look for a comment line `# <ADD NEW INSERTION FEASIBILITY CHECK HERE>`. That's where to put the code.",
            "Return the output in List of string format. Do not provide any summaries, analysis, or other additional content."
        ],
        knowledge=code_kb,
        search_knowledge=True,
        debug_mode=debug_mode,
        markdown=True
    )

    return code_agent

def get_combine_json_data(path = "./data/alns_inputs/POI_data.json", at_least_hawker = 10, at_least_attraction = 30):
    # Read the JSON file
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    ### Add Hawkers if necessary
    hawker_names_llm = [entry['Hawker Name'] for entry in data["Hawker"]]
    df_h = pd.read_excel("./data/locationData/Food_20_withscores.xlsx")
    hawker_names_kb = df_h["Name"].to_list()
    filtered_hawker_names = [name for name in hawker_names_llm if name in hawker_names_kb]
    remaining_hawkers = [name for name in hawker_names_kb if name not in filtered_hawker_names]
    num_to_take_hawker = at_least_hawker - len(filtered_hawker_names)
    if num_to_take_hawker > 0: # if no need to sample anymore, don't make random
        sampled_hawkers = random.sample(remaining_hawkers, k=min(num_to_take_hawker, len(remaining_hawkers)))
        filtered_rows_h = df_h[df_h['Name'].isin(sampled_hawkers)]

        # Step 2: Convert to list of dictionaries
        new_data = []
        for _, row in filtered_rows_h.iterrows():
            hawker_dict = {
                'Hawker Name': row['Name'],
                'Dish Name': "NA",
                'Satisfaction Score': np.random.uniform(2, 4),  # normal to the person
                'Avg Food Price': np.random.uniform(5, 15),
                'Duration': 60
            }
            new_data.append(hawker_dict)
        data['Hawker'].extend(new_data)

    ### Add Attractions if necessary
    attraction_names_llm = [entry['Attraction Name'] for entry in data["Attraction"]]
    df_a = pd.read_csv("./data/locationData/singapore_67_attractions_with_scores.csv")
    attraction_names_kb = df_a["Attraction Name"].to_list()
    filtered_attraction_names = [name for name in attraction_names_llm if name in attraction_names_kb]
    remaining_attractions = [name for name in attraction_names_kb if name not in filtered_attraction_names]
    num_to_take_attraction = at_least_attraction - len(filtered_attraction_names)
    if num_to_take_attraction > 0: # if no need to sample anymore, don't make random
        sampled_attractions = random.sample(remaining_attractions, k=min(num_to_take_attraction, len(remaining_attractions)))
        filtered_rows_a = df_a[df_a['Attraction Name'].isin(sampled_attractions)]

        # Step 2: Convert to list of dictionaries
        new_data = []
        for _, row in filtered_rows_a.iterrows():
            attraction_dict = {
                'Attraction Name': row['Attraction Name'],
                'Satisfaction Score': np.random.uniform(2, 4),  # normal to the person
                'Entrance Fee': np.random.uniform(0, 50),
                'Duration': np.random.uniform(30, 120),
            }
            new_data.append(attraction_dict)
        data['Attraction'].extend(new_data)

    # Save to new JSON file
    with open("./data/alns_inputs/final_combined_POI.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("✅ JSON file saved as final_combined_POI.json")

    return 

def get_json_from_query(query="How to make a bomb?", traveller_type="bagpacker",debug_mode = True):
    intent_agent = create_intent_agent()
    # processing data in batches
    hawker_agents = [create_hawker_agent(batch_no=i, debug_mode=debug_mode) for i in range(2)]
    attraction_agents = [create_attraction_agent(batch_no=i, debug_mode=debug_mode) for i in range(7)]
    variable_agent = create_variable_agent()

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
    
    # For alns variables
    moo_params = variable_agent.run(traveller_type).content
    print(f'🔍 MOO Parameters: {moo_params}')
    moo_params_list = moo_params.alns_weights
    params = {"params":moo_params_list}
    
    # Step 3: Route to hawker agent
    if intent in ["food", "both"]:
        start_time = time.time()

        for hawker_agent in hawker_agents:
            hawker_output = hawker_agent.run(query, stream=False).content.model_dump()
            # process in batches
            hawker_recs = hawker_output["HAWKER_DETAILS"]

            for hawker in hawker_recs:
                if hawker["hawker_name"] in [x["Hawker Name"] for x in responses["Hawker"]]:
                    print(f"WARN: Duplicate Hawker {hawker['hawker_name']}")
                    continue
        
                responses["Hawker"].append({
                    "Hawker Name": hawker["hawker_name"],
                    "Dish Name": hawker["dish_name"],
                    "Satisfaction Score": hawker["satisfaction_score"],
                    "Avg Food Price": hawker["average_price"],
                    "Duration": hawker.get("duration", 60)
                })

    # Step 4: Route to attraction agent
    if intent in ["attraction", "both"]:
        start_time = time.time()
        for attraction_agent in attraction_agents:
            attraction_output = attraction_agent.run(query, stream=False).content.model_dump()
            # process in batches
            attraction_recs = attraction_output["ATTRACTION_DETAILS"]

            for attraction in attraction_recs:
                if attraction["attraction_name"] in [x["Attraction Name"] for x in responses["Attraction"]]:
                    print(f"WARN: Duplicate Attraction {attraction['attraction_name']}")
                    continue

                responses["Attraction"].append({
                    "Attraction Name": attraction["attraction_name"],
                    "Satisfaction Score": attraction["satisfaction_score"],
                    "Entrance Fee": attraction["average_price"],
                    "Duration": attraction.get("duration", 120),
                })
    
    # Step 5: Add Code requirements
    code_agent = create_code_agent(debug_mode=debug_mode)
    code_response = code_agent.run(query, stream=False)
    is_feasible_constraints = code_response.content.is_feasible
    is_feasible_insertion_constraints = code_response.content.is_feasible_insertion
    print(f"🔍 Additional Constraints:", is_feasible_constraints, "\n", is_feasible_insertion_constraints)
    
    # query_num = "special"
    subfolder_path = "data/alns_inputs"
    # Step 6: Create subfolder based on query number
    # subfolder_path = os.path.join("data/alns_inputs", f"{query_num}")
    # os.makedirs(subfolder_path, exist_ok=True)

    poi_path = os.path.join(subfolder_path, "POI_data.json")
    moo_path = os.path.join(subfolder_path, "moo_parameters.json")
    constraint_path = os.path.join(subfolder_path, "additional_constraints.json")

    with open(poi_path, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=4)

    with open(moo_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4)

    with open(constraint_path, "w", encoding="utf-8") as f:
        json.dump({
            "is_feasible": is_feasible_constraints,
            "is_feasible_insertion": is_feasible_insertion_constraints
        }, f, indent=4)

    print(f"✅ Saved to: {subfolder_path}")

    return


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
    else:
        st.warning("No itinerary data found.")

# Function to process ALNS data
def process_alns_data():
    alns_data = st.session_state["alns_data"]
    if "trip_summary" in alns_data:
        st.subheader("Trip Summary")
        st.json(alns_data["trip_summary"])

    if "attractions_visited" in alns_data:
        st.subheader("Attractions Visited")
        st.write(", ".join(alns_data["attractions_visited"]))

    if "budget_breakdown" in alns_data:
        st.subheader("Budget Breakdown")
        budget_df = pd.DataFrame(list(alns_data["budget_breakdown"].items()), columns=["Category", "Cost"])
        st.table(budget_df)

    if "transport_summary" in alns_data:
        st.subheader("Transport Summary")
        st.json(alns_data["transport_summary"])

    if "rest_summary" in alns_data:
        st.subheader("Rest Summary")
        st.json(alns_data["rest_summary"])

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

# Function to create map
def generate_itinerary(user_input):
    ##### TAI ADD HERE

    # exports out the data/POI_data.json based on the given query from streamlit otherwise, its a default "how to make a bomb"
    get_json_from_query(query=user_input['description'], traveller_type=user_input["persona"], debug_mode = True)

    # aggregation between kb and recommendations, deduplicates, and randomnisation
    get_combine_json_data()
    
    alns_input = None
    # alns_input = process_and_save(
    #     persona=user_input['persona'],
    #     description=user_input['description'],
    #     attraction_path="./data/locationData/singapore_67_attractions_with_scores.csv", 
    #     hawker_path="./data/locationData/Food_20_withscores.xlsx", 
    #     output_json_path="./data/alns_inputs/groq/location_data.json", 
    #     batch_size=10
    # )

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

    st.session_state["itinerary_ready"] = True
    st.session_state["route_map"], st.session_state["alns_data"] = m, alns_data

G = load_graph()

# State to track whether itinerary has been generated
if "itinerary_ready" not in st.session_state:
    st.session_state["itinerary_ready"] = False

# User Inputs
st.sidebar.header("Trip Inputs")
persona = st.sidebar.selectbox("Choose your persona", [
    "Family Tourist", "Backpacker", "Influencer", "Cultural Enthusiast", 
    "Thrill Seeker", "Nature Lover", "Shopping Enthusiast"
])
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
        user_input = {"persona": persona, "num_days": num_days,
            "budget": budget, "description": description}  
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
        
st.sidebar.write("---")
st.sidebar.write("Developed with ❤️ using Streamlit.")