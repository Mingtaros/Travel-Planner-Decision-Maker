
from data.llm_batch_process import process_and_save

#==================
from pydantic import BaseModel, Field
from textwrap import dedent
from typing import List, Optional, Literal
from textwrap import dedent
import time

import numpy as np
import pandas as pd
import json
from dotenv import load_dotenv
import os
import random

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq

from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.googlesearch import GoogleSearchTools

from agno.knowledge.csv import CSVKnowledgeBase
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.tools.calculator import CalculatorTools

from agentic.multiagent_utils import get_transport_matrix, get_poi_time_bracket, get_location_types

# ========================================================
# Load environment variables & classess for Pydantic Base Models
# ========================================================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

class Activity(BaseModel):
    activity_type: str  # "attraction" or "food"
    name: str
    duration: int  # in minutes
    estimated_cost: float
    satisfaction_score: int
    duration_from_previous_point: int
    notes: Optional[str]

class DayPlan(BaseModel):
    day: int
    activities: List[Activity]

class Summary(BaseModel):
    total_cost_sgd: float
    total_transit_duration_min: int
    total_satisfaction_score: int

class ItineraryResponse(BaseModel):
    itinerary: List[DayPlan]
    summary: Summary

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


class AffectedPOIDetail(BaseModel):
    poi_affected: str = Field(..., description="Name of the place affected. Please use the exact naming that exist in the itinerary.")
    poi_type: str = Field(..., description="Type of the place affected. It can ony be 'hawker' or 'attraction'. Use the exact naming that exist in the itinerary.")
    day_affected: str = Field(..., description="Day that this place is affected. Please use the exact day that exist in the itinerary.")
    time_affected: str = Field(..., description="Time that this place is affected. Please use the exact time that exist in the itinerary.")

class AffectedPOIResponse(BaseModel):
    AFFECTED_POI_DETAILS: List[AffectedPOIDetail] = Field(..., description="List of Affected Points of Interest.")

class UpdatedTripSummary(BaseModel):
    duration: int
    total_budget: float
    actual_expenditure: float
    total_travel_time: int
    total_satisfaction: float
    objective_value: float
    is_feasible: bool
    starting_hotel: str

class UpdatedLocation(BaseModel):
    name: str
    type: Literal['hotel', 'attraction', 'hawker']
    arrival_time: str
    departure_time: str
    lat: float
    lng: float
    transit_from_prev: Optional[Literal['transit', 'drive']] = None
    transit_duration: int
    transit_cost: float
    duration: Optional[int] = None
    satisfaction: Optional[float] = 0
    cost: Optional[float] = 0
    rest_duration: Optional[int] = 0
    actual_arrival_time: Optional[str] = None
    description: Optional[str] = None
    position: Optional[Literal['end']] = None
    meal_type: Optional[Literal['lunch', 'dinner']] = None

class UpdatedDayPlan(BaseModel):
    day: int
    date: str
    locations: List[UpdatedLocation]

class UpdatedBudgetBreakdown(BaseModel):
    attractions: float
    meals: float
    transportation: float

class UpdatedTransportSummary(BaseModel):
    total_duration: int
    total_cost: float

class UpdatedRestSummary(BaseModel):
    total_rest_duration: int

class UpdatedItineraryResponse(BaseModel):
    trip_summary: UpdatedTripSummary
    days: List[UpdatedDayPlan]
    attractions_visited: List[str]
    budget_breakdown: UpdatedBudgetBreakdown
    transport_summary: UpdatedTransportSummary
    rest_summary: UpdatedRestSummary
    
class UpdatedDayResponse(BaseModel):
    updated_locations: List[UpdatedLocation]

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

def create_hawker_agent(model_id="gpt-4o", batch_no=0, debug_mode=True):
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
        tools=[
            # DuckDuckGoTools(
            #     search=True,
            #     fixed_max_results=3,
            # ),
            GoogleSearchTools()
        ],
        show_tool_calls=True,
        debug_mode=debug_mode,  # Comment if not needed - added to see the granularity for debug like retrieved context from vectodb
        markdown=True,
        # add_references=True, # enable RAG by adding references from AgentKnowledge to the user prompt.
    )
    return hawker_agent

def create_attraction_agent(model_id="gpt-4o", batch_no=0, debug_mode=True):
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
        tools=[
            # DuckDuckGoTools(
            #     search=True,
            #     fixed_max_results=3,
            # ),
            GoogleSearchTools()
        ],
        show_tool_calls=True,
        debug_mode=debug_mode,  # Comment if not needed - added to see the granularity for debug like retrieved context from vectodb
        markdown=True,
        # add_references=True, # enable RAG by adding references from AgentKnowledge to the user prompt.
    )
    return attraction_agent

calculator_tool = CalculatorTools(
    add=True,
    subtract=True,
    multiply=True,
    divide=True,
    exponentiate=True,
    factorial=True,
    is_prime=True,
    square_root=True,
    )

def create_itinerary_agent(hawkers: list, attractions: list, model_id="gpt-4o", debug_mode=True):
    """
    This is the agent that creates the itinerary; closest competitor would be the optimizer method (e.g. alns) given a list of POIs.

    requires query as string
    requires list of hawkers and list of attractions
    """
    itinerary_agent = Agent(
        name="Itinerary Generator Agent",
        agent_id="itinerary_agent",
        model=OpenAIChat(
            id=model_id,
            response_format="json",
            temperature=0.2,
            top_p=0.2
        ),
        structured_outputs=True,
        response_model=ItineraryResponse,
        description="Expert in building multi-day travel itineraries within a given budget.",
        role="Create a detailed daily plan with time slots, alternating attractions and food stops.",
        instructions = dedent(f"""
        You are a professional travel itinerary planner for tourists visiting Singapore.

        You are provided with the following points of interest:
        hawkers: {hawkers}
        attractions: {attractions}

        You will receive a user query containing a travel description, number of days, and budget.

        Your task is to create a detailed, realistic, and budget-feasible multi-day itinerary.

        Constraints:
        - Each day must begin and end at Marina Bay Sands Hotel.
        - Each day should include 2–3 attractions and 2 hawker center visits (for lunch and dinner).
        - Distribute high-cost attractions across different days.
        - Mix free and paid attractions to help stay within the total budget.
        - DO NOT fabricate or guess any cost — everything must be calculated explicitly.

        ✅ You MUST use the CalculatorTool to:
        - Compute `total_cost_sgd` (sum of all POI estimated_cost values across all days)

        Return ONLY a valid JSON object with the following structure:

        {{
        "itinerary": [
            {{
            "day": 1,
            "activities": [
                {{
                "activity_type": "attraction",
                "name": "Gardens by the Bay",
                "duration": 180,
                "estimated_cost": 28.0,
                "duration_from_previous_point": 0,
                "notes": "Explore the Cloud Forest and Flower Dome."
                }},
                {{
                "activity_type": "food",
                "name": "Maxwell Food Centre",
                "duration": 60,
                "estimated_cost": 5.0,
                "duration_from_previous_point": 20,
                "notes": "Try Tian Tian Hainanese Chicken Rice."
                }}
            ]
            }}
        ],
        "summary": {{
            "total_cost_sgd": 33.0  // Must be computed using CalculatorTool
        }}
        }}

        ⚠️ DO NOT include satisfaction or transit duration in the summary.
        ⚠️ DO NOT return markdown, bullet points, or explanations.
        ✅ Ensure JSON is valid and parsable.

        💡 Step-by-step for cost computation:
        Step 1: Extract all `estimated_cost` values from the itinerary.
        Step 2: Use the CalculatorTool with a formula like:
            "28.0 + 5.0 + 12.0 + ..." to compute total cost.
        Step 3: Insert the result into `summary.total_cost_sgd`.

        ❗ Never guess the total — always compute using the CalculatorTool.
    """),
        show_tool_calls=debug_mode,
        markdown=True,
        tools=[calculator_tool, GoogleSearchTools()]
    )

    return itinerary_agent


def create_code_agent(model_id="gpt-4o", debug_mode=True):
    code_kb = get_vrp_code_kb()
    code_kb.load(recreate=True)
    code_agent = Agent(
        name="Query to Code Agent",
        agent_id="query_to_code_agent",
        model=OpenAIChat(id=model_id, 
                         response_format="json",
                         temperature=0.0,
                         top_p=0.2,
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
            "Based on the traveller's persona and description, make new constraints to add to these functions only if you think it's necessary.",
            "ONLY add the code pieces if it's necessary. If it's not needed, return an empty list for both `is_feasible` and `is_feasible_insertion` function.",
            "For variable `problem`, they have a few class attributes, NUM_DAYS, MAX_ATTRACTION_PER_DAY, START_TIME, HARD_LIMIT_END_TIME,",
            "LUNCH_START, LUNCH_END, DINNER_START, DINNER_END, TIME_BRACKETS, budget, locations, transport_types, transport_matrix",
            "Each location in locations attribute is a dictionary, the keys are 'type', 'name', 'loc', 'lat'.",
            "Available Time Brackets in TIME_BRACKETS attributes are [8, 12, 16, 20].",
            "There are 2 transport types, ['transit', 'drive'].",
            "For Transport Matrix, it is a dictionary, with the key (source location, destination location, time bracket)",
            "and the values are another dictionary, for the transport types 'transit' and 'drive'.",
            "For each dictionary, the keys are 'duration' and 'price'.",
            "Example of accessing: transport_matrix[('Marina Bay Sands', 'Tangs Market', 8)]['transit']['duration']. This will get duration of using public transport from",
            "Marina Bay Sands to Tangs Market from around 8-11 AM",
            "For `is_feasible` function, look for a comment line `# <ADD NEW FEASIBILITY CHECK HERE>`. That's where to put the code.",
            "For `is_feasible_insertion` function, look for a comment line `# <ADD NEW INSERTION FEASIBILITY CHECK HERE>`. That's where to put the code.",
            "The code MUST WORK, no error, must be integrated nicely to the code."
            "Return the output in List of string format. Do not provide any summaries, analysis, or other additional content."
            "The hawkers only have the following attributes: name, avg_food_price, rating (max 5).",
            "The attractions only have the following attributes: name, entrance_fee, satisfaction (max 5), duration.",
            "Add constraints that may limit the budget as a percentage of the total budget.",
            "Add constraints that may limit the travel time for each location.",
            "Add constraints that may place a minimum limit on satisfaction score (max 5) for each location.",
        ],
        knowledge=code_kb,
        search_knowledge=True,
        debug_mode=debug_mode,
        markdown=True
    )

    return code_agent

def create_update_day_agent(model_id="gpt-4o", debug_mode=True):
    update_day_agent = Agent(
        name="Day-wise Itinerary Update Agent",
        agent_id="update_day_agent",
        # model=OpenAIChat(
        #     id=model_id,
        #     response_format="json",
        #     temperature=0.2,
        #     top_p=0.2
        # ),
        model=Groq(id="llama-3.3-70b-versatile",
                   response_format={ "type": "json_object" }, 
                   temperature=0.2),
        # structured_outputs=True,
        response_model=UpdatedDayResponse,  # <-- this is the simplified response
        description="Update a single day's list of locations based on user feedback.",
        instructions=dedent(f"""
        You are a travel itinerary day editor for tourists visiting Singapore.

        You are given:
        - A list of locations for one day (arrival time, departure time, location names, etc.).
        - Feedback from the user about what they want changed.

        Your task:
        - Modify the locations based on the feedback.
        - Adjust arrival/departure times, transit types, durations, rest times as necessary.
        - Maintain logical, feasible, realistic schedules (no teleporting or magical times).

        Constraints:
        - Each day MUST start and end at the hotel.
        - Prefer replacing attractions with nearby alternatives if an attraction is removed.
        - Try to keep about the same number of activities (unless feedback says otherwise).
        - Only modify what is necessary to satisfy the feedback.
        - One attraction can only be picked once across all days.
        - One hawker center can only be picked once every day.
        - Do not fabricate location names — use existing attractions or hawkers.

        ✅ You MUST output ONLY the following JSON structure:

        {{
          "updated_locations": [
            {{
              "name": "...",
              "type": "attraction | hawker | hotel",
              "arrival_time": "HH:MM",
              "departure_time": "HH:MM",
              "lat": float,
              "lng": float,
              "transit_from_prev": "transit | drive | null",
              "transit_duration": int,
              "transit_cost": float,
              "duration": int,
              "satisfaction": float,
              "cost": float,
              "rest_duration": int,
              "actual_arrival_time": "HH:MM | null",
              "description": "optional",
              "position": "optional",
              "meal_type": "optional"
            }}
          ]
        }}

        ⚠️ DO NOT return full trip summaries or budgets. Only this day's updated locations.
        ⚠️ DO NOT invent names, costs, or satisfaction scores without logical basis.
        ✅ Output must be valid parsable JSON.
        """),
        debug_mode=debug_mode,
        # markdown=True,
        use_json_mode=True,
    )

    return update_day_agent

def create_feedback_affected_poi_agent(model_id="gpt-4o", debug_mode=True):
    affected_poi_agent = Agent(
        name="Finding Affected POIs based on feedback",
        agent_id="affected_poi_agent",
        # model=OpenAIChat(id=model_id, 
        #                  response_format="json",
        #                  temperature=0.0,
        #                  top_p=0.2,
        #                  ), 
        model=Groq(id="llama-3.3-70b-versatile",
                   response_format={ "type": "json_object" }, 
                   temperature=0.2),
        response_model=AffectedPOIResponse,
        structured_outputs=True,
        description="You are an expert on building travel itineraries for Singapore. You just received feedback from the user. You need to find the points of interest that the users would like to change from the user's feedback.",
        instructions=[
            "You will receive an itinerary in a tabular format.",
            "You will also receive a feedback on the known itinerary.",
            "Your goal is to list the where and when of the itinerary does the user want to change.",
            "For the POI name. Use the exact naming that the itinerary uses.",
            "For the time, if there are arrival and departure time, use the arrival time.",
            "If the user's feedback doesn't specify any places, speculate based from your understanding of these places.",
            "If you think that no places are affected, return an empty list.",
            "IMPORTANT: Return JUST THE JSON RESPONSE, DO NOT INCLUDE ANY OTHER THINGS.",
        ],
        debug_mode=debug_mode,
        markdown=True
    )

    return affected_poi_agent


def get_combine_json_data(path="./data/alns_inputs/POI_data.json", at_least_hawker=10, at_least_attraction=30):
    # Read the JSON file
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    ### Add Hawkers if necessary
    hawker_names_llm = [entry['Hawker Name'] for entry in data["Hawker"]]
    BASE_PATH = "./data/locationData/csv"
    df_h = []
    for batch_no in range(2):
        hawker_df = pd.read_csv(f"{BASE_PATH}/hawkers/{batch_no}/hawkers.csv")
        df_h.append(hawker_df)
    df_h = pd.concat(df_h, ignore_index=True)
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
    df_a = []
    for batch_no in range(7):
        attraction_df = pd.read_csv(f"{BASE_PATH}/attractions/{batch_no}/attractions.csv")
        df_a.append(attraction_df)
    df_a = pd.concat(df_a, ignore_index=True)
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

def get_json_from_query(query="How to make a bomb?", debug_mode=True):
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
    moo_params = variable_agent.run(query).content
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

def update_itinerary_llm(known_itinerary, feedback_query, debug_mode=True):
    update_day_agent = create_update_day_agent(debug_mode=debug_mode)

    updated_days = []

    for day_info in known_itinerary["days"]:
        day = day_info["day"]
        date = day_info["date"]
        locations = day_info["locations"]

        day_prompt = f"""
        This is the itinerary for Day {day} ({date}):
        {json.dumps(locations, indent=2)}
        
        User feedback: '{feedback_query}'
        
        Update only this day's list of locations based on the feedback.
        """

        agent_response = update_day_agent.run(day_prompt, stream=False).content.model_dump()
        updated_day = UpdatedDayPlan(day=day, date=date, locations=agent_response["updated_locations"])
        updated_days.append(updated_day)

    return updated_days

def rebuild_full_itinerary(updated_days: List[UpdatedDayPlan], known_itinerary: dict) -> UpdatedItineraryResponse:
    """
    Rebuilds a full UpdatedItineraryResponse from updated days and the original known itinerary.
    """

    # 1. Initialize accumulators
    total_expenditure = 0.0
    total_travel_time = 0
    total_satisfaction = 0.0
    total_rest_duration = 0
    attractions_visited = []
    attractions_cost = 0.0
    meals_cost = 0.0
    transportation_cost = 0.0

    # 2. Loop through updated days
    for day in updated_days:
        previous_location = None

        for loc in day.locations:
            total_expenditure += loc.cost or 0
            total_travel_time += loc.transit_duration or 0
            total_satisfaction += loc.satisfaction or 0
            total_rest_duration += loc.rest_duration or 0

            if loc.type == "attraction":
                attractions_visited.append(loc.name)
                attractions_cost += loc.cost or 0
            elif loc.type == "hawker":
                meals_cost += loc.cost or 0

            if loc.transit_cost:
                transportation_cost += loc.transit_cost or 0

            previous_location = loc

    # 3. Calculate objective value (example: simple ratio based)
    try:
        budget = known_itinerary["trip_summary"]["total_budget"]
        objective_value = (total_expenditure / budget) - (total_satisfaction / (5 * len(attractions_visited)))
    except Exception:
        objective_value = 0

    # 4. Create UpdatedItineraryResponse
    itinerary_response = UpdatedItineraryResponse(
        trip_summary=UpdatedTripSummary(
            duration=len(updated_days),
            total_budget=budget,
            actual_expenditure=round(total_expenditure, 2),
            total_travel_time=total_travel_time,
            total_satisfaction=round(total_satisfaction, 2),
            objective_value=round(objective_value, 6),
            is_feasible=True,  # Assume feasible if rebuilt correctly; later can validate
            starting_hotel=known_itinerary["trip_summary"]["starting_hotel"],
        ),
        days=updated_days,
        attractions_visited=list(sorted(set(attractions_visited))),
        budget_breakdown=UpdatedBudgetBreakdown(
            attractions=round(attractions_cost, 2),
            meals=round(meals_cost, 2),
            transportation=round(transportation_cost, 2),
        ),
        transport_summary=UpdatedTransportSummary(
            total_duration=total_travel_time,
            total_cost=round(transportation_cost, 2),
        ),
        rest_summary=UpdatedRestSummary(
            total_rest_duration=total_rest_duration
        )
    )

    return itinerary_response


def find_alternative_of_affected_pois(itinerary_table, feedback_prompt, top_n=5, debug_mode=True):
    affected_poi_agent = create_feedback_affected_poi_agent(debug_mode=debug_mode)
    main_prompt = f"You have this itinerary currently in a tabular format:\n{itinerary_table.to_string()}\n" \
        f"But this itinerary is not to the user's liking. In which their feedback is: '{feedback_prompt}'\n" \
        "Find the affected points of interest (POIs)!"

    agent_response = affected_poi_agent.run(main_prompt, stream=False).content.model_dump()
    affected_pois = agent_response["AFFECTED_POI_DETAILS"]

    # get tranport matrix & location types
    transport_matrix = get_transport_matrix()
    location_types = get_location_types()
    # cannot pick the places who has been taken before
    blacklist_places = set(itinerary_table["Location"].values)
    poi_suggestions = []

    transport_keys = transport_matrix.keys()
    # from the affected pois, find top N closest places from the transport matrix
    for affected_poi in affected_pois:
        poi = affected_poi["poi_affected"]
        poi_loc_type = affected_poi["poi_type"]
        if poi_loc_type == "hotel":
            # WE CANNOT CHANGE THE HOTEL, so just skip. There's no alternative to this.
            continue
        poi_day = affected_poi["day_affected"]
        poi_time = affected_poi["time_affected"]
        poi_time_bracket = get_poi_time_bracket(poi_time)
        possible_alternative_pois = [
            (
                transport_key[1], # destination name
                location_types[transport_key[1]],
                transport_matrix[transport_key]
            )
            for transport_key in transport_keys
            if (transport_key[0] == poi) and \
               (transport_key[1] not in blacklist_places) and \
               (transport_key[2] == poi_time_bracket) and \
               (location_types[transport_key[1]] == poi_loc_type)
        ] # find possible routes from this place at this time for the exactly the same place type
        # - find the 1st key that is affected poi
        # - alternative poi must be not taken before
        # - time bracket must be in the correct time bracket
        # - location type of alternative poi must be the same as affected poi
        # sort and pick only top 5, by shortest duration, let's say using driving
        possible_alternative_pois.sort(key=lambda x: x[2]["drive"]["duration"])
        poi_suggestions.append({
            "affected_poi": poi,
            "affected_day": poi_day,
            "affected_time": poi_time,
            "alternatives": possible_alternative_pois[:top_n]
        })
    
    return poi_suggestions


def update_itinerary_closest_alternatives(known_itinerary, feedback_prompt, poi_suggestions, debug_mode=True):
    update_day_agent = create_update_day_agent(debug_mode=debug_mode)
    updated_days = []

    for day_info in known_itinerary["days"]:
        # get poi suggestions that is for this specific days
        day = day_info["day"]
        date = day_info["date"]
        locations = day_info["locations"]
        this_days_affected_alternative = [
            f"`{suggestion['affected_poi']}` at {suggestion['affected_time']}, alternatives: {json.dumps(suggestion['alternatives'], indent=2)}"
            for suggestion in poi_suggestions
            if suggestion["affected_day"] == str(day)
        ]

        poi_alternatives = '\n\nAlternatives for: '.join(this_days_affected_alternative)

        day_prompt = f"""
        This is the itinerary for Day {day} ({date}):
        {json.dumps(locations, indent=2)}
        
        User feedback: '{feedback_prompt}'
        """

        if len(this_days_affected_alternative) > 0:
            # if there's nothing to change from this day's itinerary,
            # just use the exact same itinerary
            day_prompt += dedent(f"""
            Update only this day's list of locations based on the feedback using these possible alternatives:

            {poi_alternatives}

            Use the exact naming of the places that's provided in these lists.
            Choose between driving or transit. The data is already provided. Driving is faster, but transit is cheaper.
            """)
        else:
            day_prompt += dedent(f"""
            Because there are no affected Points of Interests for Day {day}, Return the exact same itinerary in the given format.
            """)

        agent_response = update_day_agent.run(day_prompt, stream=False).content.model_dump()
        updated_day = UpdatedDayPlan(day=day, date=date, locations=agent_response["updated_locations"])
        updated_days.append(updated_day)

    return updated_days

def update_invalid_itinerary(updated_itinerary, feedback, debug_mode=True):
    update_day_agent = create_update_day_agent(debug_mode=debug_mode)
    updated_days = []
    for day_info in updated_itinerary["days"]:
        feasibility_prompt = dedent(f"""
        This itinerary on day {day_info['day']} is not a feasible solution:
        
        {day_info['locations']}

        The feedback is: '{feedback}'.

        Can you modify it so that it matches the feedback, please?
        Don't forget to follow the necessary format.

        IMPORTANT: If you think the feedback doesn't affect the itinerary of this day, skip and return the exact same itinerary.
        """)

        agent_response = update_day_agent.run(feasibility_prompt, stream=False).content.model_dump()
        updated_day = UpdatedDayPlan(day=day_info['day'], date=day_info['date'], locations=agent_response["updated_locations"])
        updated_days.append(updated_day)
    
    return updated_days


#==================
user_queries = {
    "01": {
        "query": "We’re a family of four visiting Singapore for 3 days. We’d love to explore kid-friendly attractions and try some affordable local food. Budget is around 300 SGD.",
        "days": 3,
        "budget": 300,
    },
    "02": {
        "query": "I'm a solo backpacker staying for 3 days. My budget is tight (~150 SGD total), and I'm mainly here to try spicy food and explore free attractions.",
        "days": 3,
        "budget": 150,
    },
    "03": {
        "query": "I’ll be spending 3 days in Singapore and I'm really interested in cultural attractions and sampling traditional hawker food on a modest budget. Budget is 180 SGD.",
        "days": 3,
        "budget": 180,
    },
    "04": {
        "query": "I'm visiting Singapore for 3 days as a content creator. I'm looking for Instagrammable attractions and stylish food spots. Budget is 600 SGD.",
        "days": 3,
        "budget": 600,
    },
    "05": {
        "query": "I love adventure and spicy food! Spending 3 days in Singapore. What attractions and hawker stalls should I visit? Budget is 200 SGD.",
        "days": 3,
        "budget": 200,
    },
    "06": {
        "query": "Looking to relax and enjoy greenery and peaceful spots in Singapore. I’ll be there for 3 days and have 190 SGD to spend. I enjoy light snacks over heavy meals.",
        "days": 3,
        "budget": 190,
    },
    "07": {
        "query": "What can I do in Singapore in 3 days if I love shopping and modern city vibes? I’d also like to eat at famous food centres. Budget is 270 SGD.",
        "days": 3,
        "budget": 270,
    },
    "08": {
        "query": "My spouse and I are retired and visiting Singapore for 3 days. We love cultural sites and relaxing parks. Prefer to avoid loud or overly touristy spots. Budget is 210 SGD.",
        "days": 3,
        "budget": 210,
    },
    "09": {
        "query": "We’re a group of university students spending 3 days in Singapore on a budget of 180 SGD total. Recommend cheap eats and fun, free things to do.",
        "days": 3,
        "budget": 180,
    },
    "10": {
        "query": "This is my first time in Singapore and I’ll be here for 3 days. I’d like a mix of sightseeing, must-try foods, and some local experiences. Budget is 250 SGD.",
        "days": 3,
        "budget": 250,
    }
}
#==================
from tqdm import tqdm

if __name__ == "__main__":
    print()

    debug_mode=True
    intent_agent = create_intent_agent()
    variable_agent = create_variable_agent()
    hawker_agents = [create_hawker_agent(batch_no=i, debug_mode=debug_mode) for i in range(2)]
    attraction_agents = [create_attraction_agent(batch_no=i, debug_mode=debug_mode) for i in range(7)]
    
    for scenario, query_item in tqdm(user_queries.items(), desc="Generating itineraries", unit="Itinerary") :
        print(scenario, query_item)

        # get_json_from_query(query=query_item["query"], debug_mode=True)
        # get_combine_json_data()
        # itinerary = generate_llm_itinerary(
        #     budget=query_item["budget"],
        #     days=query_item["days"],
        #     query=query_item["query"]
        # )
        # # ✅ Validate cost
        # is_valid = validate_total_cost(itinerary)
        # if not is_valid:
        #     print("⚠️ Warning: LLM-generated itinerary has incorrect total cost.")

        intent_response = intent_agent.run(query_item["query"], stream=False)
        intent = intent_response.content.intent

        print(f"\n🔍 Processing Query: {query_item['query']}")

        if intent == "malicious":
            print("⚠️ Query flagged as malicious. Skipping...")
            continue

        responses = {
            "Query": query_item,
            "Hawker": [],
            "Attraction": []
        }

        # For alns variables
        moo_params = variable_agent.run(query_item["query"]).content
        print(f'🔍 MOO Parameters: {moo_params}')
        moo_params_list = moo_params.alns_weights
        params = {"params":moo_params_list}
        # print(moo_params_list)
        print("Budget priority", moo_params_list[0])
        print("Time priority",moo_params_list[1])
        print("Satisfaction priority",moo_params_list[2] )
        
            # itinerary = generate_llm_itinerary(
            #     budget=query_item["budget"],
            #     days=query_item["days"],
            #     query=query_item["query"]
            # )
            # # ✅ Validate cost
            # is_valid = validate_total_cost(itinerary)
            # if not is_valid:
            #     print("⚠️ Warning: LLM-generated itinerary has incorrect total cost.")

            # # Save the result
            # out_path = f"./results/llm/{scenario}/itinerary_{query_item['persona'].lower().replace(' ', '_')}.json"
            # os.makedirs(os.path.dirname(out_path), exist_ok=True)
            # with open(out_path, "w", encoding="utf-8") as f:
            #     json.dump(itinerary, f, indent=4)

            # print(f"\n✅ LLM Itinerary saved to {out_path}")
            # print(f"🕒 Generation time: {itinerary['meta']['generation_time_seconds']}s")

        print()
        break
