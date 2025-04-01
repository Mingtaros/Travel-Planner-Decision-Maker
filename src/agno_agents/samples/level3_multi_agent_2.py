from pydantic import BaseModel, Field
from typing import List
import time

import json
import datetime
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
# Load environment variables
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

def create_intent_agent(model_id = "deepseek-r1-distill-llama-70b"):
    # Create the Intent Classification Agent
    intent_agent = Agent(
        name="Intent Classification Agent",
        agent_id="intent_classification_agent",
        model=Groq(id=model_id, 
                   response_format="json", 
                   temperature=0.0),  
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
    hawker_kb.load(recreate=False)
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
    attraction_kb.load(recreate=False)
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

def save_as_json(responses, output_dir = "data/combined_outputs"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{timestamp}.json")
    print("\n✅ Final JSON Response:")
    print(json.dumps(responses, indent=4))
    with open(output_filename, "w", encoding="utf-8") as json_file:
        json.dump(responses, json_file, indent=4, ensure_ascii=False)

    print(f"\n📁 JSON response successfully saved to: {output_filename}")
    return


queries = {
    "01": {
        "query": "We’re a family of four visiting Singapore for 3 days. We’d love to explore kid-friendly attractions and try some affordable local food. Budget is around 300 SGD.",
        "days": 3,
        "budget": 300
    },
    "02": {
        "query": "I'm a solo backpacker staying for 2 days. My budget is tight (~50 SGD total), and I'm mainly here to try spicy hawker food and explore free attractions.",
        "days": 2,
        "budget": 50
    },
    "03": {
        "query": "I only have one full day in Singapore. Can you suggest cultural attractions and a local hawker spot that fits a 60 SGD day budget?",
        "days": 1,
        "budget": 60
    },
    "04": {
        "query": "I'm visiting Singapore for 2 days as a content creator. Looking for Instagrammable attractions and stylish food spots. Budget: 400 SGD.",
        "days": 2,
        "budget": 400
    },
    "05": {
        "query": "I love adventure and spicy food! Spending 3 days in Singapore. What attractions and hawker stalls should I visit? Budget is 200 SGD.",
        "days": 3,
        "budget": 200
    },
    "06": {
        "query": "Looking to relax and enjoy greenery and peaceful spots in Singapore. I’m here for 4 days and have 250 SGD to spend. I enjoy light snacks over heavy meals.",
        "days": 4,
        "budget": 250
    },
    "07": {
        "query": "What can I do in Singapore in 2 days if I love shopping and modern city vibes? I’d also like to eat at famous food centres. Budget is 180 SGD.",
        "days": 2,
        "budget": 180
    },
    "08": {
        "query": "My spouse and I are retired and visiting Singapore for 5 days. We love cultural sites and relaxing parks. Prefer to avoid loud or overly touristy spots. Budget is 350 SGD.",
        "days": 5,
        "budget": 350
    },
    "09": {
        "query": "We’re a group of university students on a short trip (2 days) with a budget of 60 SGD each. Recommend cheap eats and fun, free things to do.",
        "days": 2,
        "budget": 60
    },
    "10": {
        "query": "This is my first time in Singapore and I’ll be here for 3 days. I’d like a mix of sightseeing, must-try foods, and some local experiences. Budget is 250 SGD.",
        "days": 3,
        "budget": 250
    }
}

queries = {
        "10": {
        "query": "This is my first time in Singapore and I’ll be here for 3 days. I’d like a mix of sightseeing, must-try spicy foods, and some local experiences. Budget is 250 SGD.",
        "days": 3,
        "budget": 250
    }
}

# user_queries = {
    
#     "011": "I'm a solo backpacker staying for 2 days. My budget is tight (~250 SGD total), and I'm mainly here to try spicy hawker food and explore free attractions.",

#     "12": "My spouse and I are retired and visiting Singapore for 5 days. We love cultural sites and relaxing parks. Prefer to avoid loud or overly touristy spots. Prefer to have less oily food in general too. Budget is 350 SGD.",
    
#     "13": "We’re a group of university students on a short trip (2 days) with a budget of 60 SGD each. Recommend cheap eats and fun, free things to do.",
    
#     "14": "This is my first time in Singapore and I’ll be here for 3 days. I’d like a mix of sightseeing, must-try foods, and some local experiences. Budget is 250 SGD."
# }

if __name__ == "__main__":
    # Step 0: Create Agents
    debug_mode = True
    intent_agent = create_intent_agent()
    hawker_agent = create_hawker_agent(debug_mode=debug_mode)
    attraction_agent = create_attraction_agent(debug_mode=debug_mode)
    # preference_agent = create_preference_agent()

    # Step 1: Loop through all user queries
    for query_num, query_item in queries.items():
        print(f"\n🔍 Processing Query {query_num}: {query_item}")

        # Step 2a: Use Intent Agent to classify the query
        intent_response = intent_agent.run(query_item["query"], stream=False)
        intent = intent_response.content.intent

        # #Step 2b: Use Preference Agnent to ccheck what the query wants score of 1-10
        # preference_response = preference_agent.run(query, stream=False)
        # preference_score_json = preference_response
        # # print(preference_score_json)
        # # print(type(preference_score_json))
        
        if intent == "malicious":
            print("⚠️ Query flagged as malicious. Skipping...")
            continue

        # Initialize response dictionary
        responses = {
            "Query": query_item["query"],
            "Hawker": [],
            "Attraction": []
        }

        # print(f"The user query is: {query_item["query"]}, with the following intent: {intent}")

        # Step 3: Route to hawker agent
        if intent in ["food", "both"]:
            start_time = time.time()
            hawker_output = hawker_agent.run(query_item["query"], stream=False).content.model_dump()
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
            attraction_output = attraction_agent.run(query_item["query"], stream=False).content.model_dump()
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
                responses["Metrics"] = {
                                        # "Intent Agent Time (s)": round(intent_time, 2),
                                        "Hawker Agent Time (s)": round(hawker_time or 0, 2),
                                        "Attraction Agent Time (s)": round(attraction_time or 0, 2),
                                        # "Intent Agent Tokens": intent_usage if intent_usage else {},
                                        # "Hawker Agent Tokens": hawker_usage if hawker_usage else {},
                                        # "Attraction Agent Tokens": attraction_usage if attraction_usage else {}
                                    }

        # Step 5: Prepare hardcoded MOO parameters
        moo_params = {
            "Budget": 100,
            "Number of days": 3,
            "params": [0.3, 0.3, 0.4]
        }

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


        ### aggregate and check for unique for downstream task