from agno.agent import Agent
from agno.models.huggingface import HuggingFace

from agno.models.openai import OpenAIChat
from agno.models.groq import Groq

from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.sql import SQLTools
from sqlalchemy import create_engine
# from agno.storage.sqlite import SqliteStorage

import os
from dotenv import load_dotenv
load_dotenv()

# Getting the API key 
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# os.environ["HF_TOKEN"] = os.getenv(key="HF_API_KEY")

# Getting the SQL Lite Database
engine = create_engine("sqlite:///Chinook_Sqlite.sqlite")

# # For Debugging
# print(dir(HuggingFace))

# from inspect import signature
# print(signature(HuggingFace.__init__))

# Groq works well
hotel_agent = Agent(
    name = "hotel",
    model=Groq(id="qwen-2.5-32b"),
    description="You are a helpful travel assistant that is specialised in Singapore's Hotels Accomodations.",
    tools=[DuckDuckGoTools()],
    instructions=[
        "Search your knowledge base for famous hotel places in Singapore",
        "If the quetion is better suited for the web, search the web to fill in the gaps",
        ],
    markdown=True
)

# OpenAI also works well
sql_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    markdown=True,
    show_tool_calls=True,
    system_message="You are equipped with tools to manage the SQL DATABASE.",
    tools=[SQLTools(db_engine=engine)],
    add_history_to_messages=True,
    retries=3
)

# Defining the multi agent
TripMaster = Agent(
    name="Editor",
    team=[hotel_agent, food_agent],
    description="You are a senior travel itinerary expert, responsible for curating well-researched and structured travel guides that is based in Singapore.",
    instructions=[
        "Gather recommendations from the travel and food experts to form a detailed itinerary.",
        "Ensure that all recommended locations are relevant to the traveler's profile.",
        "Prioritize unique, hidden-gem locations alongside mainstream attractions.",
        "Edit, proofread, and refine the itinerary to ensure coherence and clarity.",
        "Ensure recommendations are up-to-date and in line with local Singapore cultural nuances.",
        "Deliver responses in a well-structured travel guide format with clear sections (e.g., Morning, Afternoon, Evening activities).",
    ],
    add_datetime_to_instructions=True,  # This will dynamically add the current date/time
    markdown=True,
    debug_mode=True
)

# ## Need to put HF api token
# food_agent = Agent(
#     model=HuggingFace(
#         id="mistralai/Mistral-7B-Instruct-v0.2", max_tokens=4096, temperature=0,
#         instructions=[
#         "You are an AI that specializes in travel planning.",
#         "Always provide budget-friendly and efficient travel options.",
#         "Be concise and informative in your responses."
#     ]
#     )
# )

## Testing the multi agent TripMaster
# query = "Based on the persona of being adventourous who is single in his 30s, what are some popular places you recommend?"
# TripMaster.print_response(query)

## Testing the food agent 
# query = "Based on the persona of being adventourous who is single in his 30s, what are some popular places you recommend?"
# food_agent.print_response(query)

query = "list the tables"
food_agent.print_response(query)