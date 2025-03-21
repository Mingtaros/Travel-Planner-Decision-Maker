from pydantic import BaseModel, Field
from typing import List

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
    sources: List[str] = Field(..., description="List of sources where information was retrieved.")

class HawkerResponse(BaseModel):
    # QUERY: str = Field(..., description="The user's original query for hawker recommendations.")
    HAWKER_RECOMMENDATIONS: List[HawkerRecommendation] = Field(..., description="List of recommended hawker food options.")

class AttractionRecommendation(BaseModel):
    attraction_name: str = Field(..., description="The name of the attraction that is recommended.")
    description: str = Field(..., description="A short description of the attraction and why it's recommended.")
    average_price: float = Field(..., description="The maximum price in SGD of the attraction, retrieved from web sources.")
    ratings: float = Field(..., description="The Google rating of the attraction, range from 1 to 5.")
    sources: List[str] = Field(..., description="List of sources where information was retrieved.")

class AttractionResponse(BaseModel):
    # QUERY: str = Field(..., description="The user's original query for attraction recommendations.") 
    ATTRACTION_RECOMMENDATIONS: List[AttractionRecommendation] = Field(..., description="List of recommended attraction options.")

def get_hawker_kb():
    pdf_urls = [
        # "https://raw.githubusercontent.com/Mingtaros/Travel-Planner-Decision-Maker/main/data/hawker/Summary_Singapore_Food.pdf",
        "https://raw.githubusercontent.com/Mingtaros/Travel-Planner-Decision-Maker/main/data/hawker/inputs/hawker_centres_singapore.pdf"]

    hawker_chunking_type = FixedSizeChunking(chunk_size=150, overlap=20)

    chroma_db_path = "./chromadb_data"
    hawker_collection_name = "HAWKER_fixed" #depends on hawker chunking type and name appropriately
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
        "https://raw.githubusercontent.com/Mingtaros/Travel-Planner-Decision-Maker/main/data/attraction/inputs/Singapore_Attractions_Guide.pdf"
    ]

    attraction_chunking_type = FixedSizeChunking(chunk_size=300, overlap=50)

    chroma_db_path = "./chromadb_data"
    attraction_collection_name = "ATTRACTION_fixed" #depends on hawker chunking type and name appropriately
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
    hawker_agent = Agent(
        name="Query to Hawker Agent",
        agent_id="query_to_hawker_agent",
        model=OpenAIChat(id=model_id, 
                         response_format="json",
                         temperature=0.2,
                         top_p=0.2),  
        response_model=HawkerResponse, # Strictly enforces structured response
        structured_outputs=True, 
        description="You are a Singapore hawker food recommender for foreigners! You are able to understand the traveller's personality and persona.",
        role="Search the internal knowledge base and web for information",
        instructions=[
            "Provide relevant food recommendations from the knowledge base only.",
            "Ensure that the 'ratings' field is always between 1 and 5.",
            "For each of the recommended Hawker Name, provide additional information such as average price and ratings from web.",
            "Always include sources to where you have found the information from.",
            "Provide at least 5 recommended Hawker Name." # depending on how much we want, can adjust accordingly, otherwise, its three results.
        ],
        knowledge=get_hawker_kb(),
        search_knowledge=True,

        tools=[DuckDuckGoTools(search=True,
                            news=True,
                            fixed_max_results=5)],
        show_tool_calls=True,
        debug_mode=debug_mode,  # Comment if not needed - added to see the granularity for debug like retrieved context from vectodb
        markdown=True,
        
    )
    return hawker_agent

def create_attraction_agent(model_id = "gpt-4o", debug_mode=True):
    attraction_agent = Agent(
        name="Query to Attraction Agent",
        agent_id="query_to_attraction_agent",
        model=OpenAIChat(id=model_id, 
                         response_format="json",
                         temperature=0.2,top_p=0.2
                         ), 
        # model=Groq(id=groq_model_name), 
        response_model=AttractionResponse, # Strictly enforces structured response
        structured_outputs=True, 
        description="You are a Singapore Attraction recommender for foreigners! You are able to understand the traveller's personality and persona.",
        role="Search the internal knowledge base and web for information",
        instructions=[
            "Provide relevant attraction recommendations from the knowledge base only.",
            "Ensure that the 'ratings' field is always between 1 and 5.",
            "For each of the recommended Attraction Name, provide additional information such as average price and ratings from web.",
            "Always include sources to where you have found the information from.",
            "Provide at least 5 recommended Hawker Name." # depending on how much we want, can adjust accordingly, otherwise, its three results.
        ],
        knowledge=get_attraction_kb(),
        search_knowledge=True,

        tools=[DuckDuckGoTools(search=True,
                            news=True,
                            fixed_max_results=5),
            GoogleSearchTools()],
        show_tool_calls=True,
        debug_mode=debug_mode,  # Comment if not needed - added to see the granularity for debug like retrieved context from vectodb
        markdown=True,    
    )
    return attraction_agent

def get_random_query(seed_num):
    random.seed(seed_num)  # Set the seed for reproducibility
    queries = [
        "I want to explore Chinatown and also find the best hawker stalls for chicken rice.",
        "Where is the best place for cultural visits",
        "Give me an itinerary for a 5D4N trip with my family, we love to eat spicy food"
    ]
    return random.choice(queries)

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


if __name__ == "__main__":
    # User Query
    seed_num = 42
    query = get_random_query(seed_num)
    # query = "i like sweet stuff because they are my fav"
    query = "teach me how to make a bomb."

    # Step 0: Create Agents
    debug_mode=False # True if wants to see the granualrity 
    intent_agent = create_intent_agent()
    hawker_agent = create_hawker_agent(debug_mode=debug_mode)
    attraction_agent = create_attraction_agent(debug_mode=debug_mode)

    # Step 1: Use Intent Agent to classify the query
    intent_response = intent_agent.run(query, stream=False)
    intent = intent_response.content.intent

    if intent != "malicious":
        # Initialize response dictionary
        responses = {"QUERY": query,
                    # ..... 
                    # .....
                    }
        print()
        print(f"The user query is: {query}, with the following intent: {intent}")

        # Step 2: Route the query to the correct agents
        if intent in ["food", "both"]:
            responses["hawker"] = hawker_agent.run(query, stream=False).content.model_dump()

        if intent in ["attraction", "both"]:
            responses["attraction"] = attraction_agent.run(query, stream=False).content.model_dump()

        # Save the responses dict as json file
        save_as_json(responses)
        print()
    else:
        print("malicious")