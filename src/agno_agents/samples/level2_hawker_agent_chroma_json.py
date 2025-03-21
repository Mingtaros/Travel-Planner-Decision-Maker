"""
Level 2: Intelligent agents with memory and reasoning capabilities. (SINGLE AGENT EXAMPLE)
A Hybrid Retrieval agent that does both internal rag and external web search.
A JSON compliant agent indeed ;) 

The agents here utilizes a vector database that has stored knowledge, enabling them to perform Retrieval-Augmented Generation (RAG).

By default, Agno agents employ Agentic RAG, where they query their knowledge base for specific details required to complete tasks. 
For this example, the RAG process uses a connected PDF URL as the data source and specified chunking strategy.

Alongside the agent's high-level description (system prompt), you can provide detailed instructions to fine-tune its behavior.
A single Agentic RAG agent is equipped with two key components: tools and knowledge/memory capabilities.
This allows for a dynamic workflow where queries are refined as needed, knowledge is retrieved, and responses are generated accordingly.

This example focuses on recommending food.
But what if you wanted to find hotels instead?
To handle such diverse tasks, you would need something more advanced than Level 2 agents.

SOURCE ORIGINAL:
https://singaporeverified.com/food/hawker-centres-in-singapore/
"""

"""
Expected output for Hawker
[
    {
        "name": (string) name of the hawker,
        "rating": (float) rating of the hawker in general. Range 1-5,
        "avg_food_price": (float) average price of eating there in SGD
    },
    {...}
]

export CHUNKING_TYPE="fixed" or 
export CHUNKING_TYPE="agentic" or 
export CHUNKING_TYPE="semantic"

"""

import json
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq

from agno.tools.duckduckgo import DuckDuckGoTools

from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from agno.embedder.openai import OpenAIEmbedder

from agno.document.chunking.fixed import FixedSizeChunking
from agno.document.chunking.agentic import AgenticChunking
from agno.document.chunking.semantic import SemanticChunking

from dotenv import load_dotenv
import os

from typing import List
from pydantic import BaseModel, Field

import datetime

# ========================================================
# PyDantic Model
# ========================================================

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

# ========================================================
# Load environment variables
# ========================================================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

pdf_urls = [
    # "https://raw.githubusercontent.com/Mingtaros/Travel-Planner-Decision-Maker/main/data/hawker/Summary_Singapore_Food.pdf",
    "https://raw.githubusercontent.com/Mingtaros/Travel-Planner-Decision-Maker/main/data/hawker/inputs/hawker_centres_singapore.pdf"
]
# ========================================================
# Dynamically Select Chunking Strategy
# ========================================================
chunking_options = {
    "fixed": FixedSizeChunking(chunk_size=150, overlap=20),  # Example: Adjust chunk size if needed
    "agentic": AgenticChunking(model=OpenAIChat, max_chunk_size=500),
    "semantic": SemanticChunking(),
}

# Get chunking type from environment variable (default to "fixed")
chunking_type_name = os.getenv("CHUNKING_TYPE", "fixed").lower()
if chunking_type_name not in chunking_options:
    print(f"⚠️ Warning: '{chunking_type_name}' is not a valid chunking type. Falling back to 'fixed'.")
    chunking_type_name = "fixed"

chunking_type = chunking_options[chunking_type_name]

# Print selected chunking type
print(f"✅ Using Chunking Strategy: {chunking_type_name.upper()}")

# ========================================================
# Persistent Storage Configuration
# ========================================================
# Define persistent storage paths and table names dynamically
chroma_db_path = "./chromadb_data"
collection_name = f"HAWKER_{chunking_type_name}"  # Example: "HAWKER_fixed"

vector_db = ChromaDb(
    collection=collection_name, 
    path=chroma_db_path,
    persistent_client=True   # Enable persistence
)

# ========================================================
# Print Debugging Information on Terminal
# ========================================================
# Debugging Information
print("\n=======================================")
print(f">>Chunking Strategy: {chunking_type_name.upper()}")
print(f">>ChromaDB Collection Name: {collection_name}")
print(f">>ChromaDB Storage Path: {chroma_db_path}")
print(f">>PDFs Being Processed:")
for url in pdf_urls:
    print(f"   - {url}")
print("=======================================\n")
# ========================================================
# Modify Instructions to Require JSON Output
# ========================================================
hawker_agent = Agent(
    name="Query to Hawker Agent",
    agent_id="query_to_hawker_agent",
    model=OpenAIChat(id="gpt-4o", response_format="json",temperature=0.2,top_p=0.2),  # Hyperparams pre-defined, Ensures OpenAI outputs valid JSON
    # model=Groq(id=groq_model_name), 
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
    knowledge=PDFUrlKnowledgeBase(
        urls=pdf_urls,
        chunking_strategy=chunking_type,
        vector_db=vector_db,
    ),
    search_knowledge=True,

    tools=[DuckDuckGoTools(search=True,
                           news=True,
                           fixed_max_results=5)],
    show_tool_calls=True,
    debug_mode=True,  # Comment if not needed - added to see the granularity for debug like retrieved context from vectodb
    markdown=True,
    
)

# Load knowledge base if needed
if hawker_agent.knowledge is not None:
    hawker_agent.knowledge.load()

# ========================================================
# Generate Response and Ensure Valid JSON
# ========================================================
query = "I love organs type of food as I'm pretty adventurous, what do you recommend me to eat on my first day in Singapore because I'm staying at town area?"
# query = "recommend me food places as i have sweet tooth."
# query = "I'm on a budget. What are the cheapest but best hawker foods in Singapore?"
# query = "I love spicy food! Which hawker dishes are a must-try?"
# query = "Where can I find the best Hainanese Chicken Rice in Singapore?"
# query = "Where can I find halal-certified hawker food in Singapore?"
# query = "Where should I go for late-night hawker food in Singapore?"

response = hawker_agent.run(query,
                     stream=False)  # Streaming disabled to capture full response

# ========================================================
# Prepare the folder to save the json into
# ========================================================

# Generate timestamp in "YYYYMMDD_HHMMSS" format
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Define the directory name using the timestamp
output_dir = f"data/hawker/outputs/"

# Create the directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Define the output file path inside the timestamped folder
output_filename = os.path.join(output_dir, f"{timestamp}.json")

print()
print("\nUser Query:", query)

try:
    # Convert response.content (Pydantic model) to a dictionary
    if hasattr(response.content, "model_dump"):
        json_response = response.content.model_dump()  
    else:
        json_response = response.content  
    
    # Inject the user query into the JSON response
    json_response["query"] = query  

    # Pretty-print JSON response in console
    print("\n✅ Successful JSON Response:")
    print(json.dumps(json_response, indent=4))

    # Save JSON response to the timestamped directory
    with open(output_filename, "w", encoding="utf-8") as json_file:
        json.dump(json_response, json_file, indent=4, ensure_ascii=False)

    print(f"\n📁 JSON response successfully saved to: {output_filename}")

except Exception as e:
    print("\n❌ Unexpected Error:", e)
    print("⚠️ Invalid response received:")
    print(response.content)