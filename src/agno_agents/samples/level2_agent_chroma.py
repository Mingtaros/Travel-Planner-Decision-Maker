"""
Level 2: Intelligent agents with memory and reasoning capabilities. (SINGLE AGENT EXAMPLE)

These agents utilize a vector database that has stored knowledge, enabling them to perform Retrieval-Augmented Generation (RAG).

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

# ========================================================
# Load environment variables
# ========================================================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

pdf_urls = [
    # "https://raw.githubusercontent.com/Mingtaros/Travel-Planner-Decision-Maker/main/data/hawker/Summary_Singapore_Food.pdf",
    "https://raw.githubusercontent.com/Mingtaros/Travel-Planner-Decision-Maker/main/data/hawker/hawker_centres_singapore.pdf"
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
agent = Agent(
    model=OpenAIChat(id="gpt-4o", response_format="json",temperature=0.2),  # Ensures OpenAI outputs valid JSON
    # model=Groq(id=groq_model_name),  # If using Groq, ensure the model understands JSON format
    description="You are a Singapore hawker food recommender for foreigners! You are able to understand the traveller's personality and persona.",
    role="Search the internal knowledge base and web for information",
    instructions=[
        "Provide relevant food recommendations from the knowledge base only.",
        "Provide additional information such as average price and ratings of the Hawker from web.",
        "Always include sources to back what you have mentioned.",
        "Respond **strictly** in JSON format with the following structure:",
        """
        {
            "HAWKER_RECOMMENDATIONS": [
                {   "hawker_name": "<Hawker Centre Name only>",
                    "dish_name": "<Dish Name>",
                    "description": "<Short description of the dish>",
                    "average_price": "<Max price in SGD from web>",
                    "ratings": "<Google rating from range of 1-5>",
                    "sources": ["<source1>", "<source2>"]
                }
            ]
        }
        """
    ],
    knowledge=PDFUrlKnowledgeBase(
        urls=pdf_urls,
        chunking_strategy=chunking_type,
        vector_db=vector_db,
    ),
    search_knowledge=True,

    tools=[DuckDuckGoTools(search=True,fixed_max_results=10)],
    show_tool_calls=True,
    debug_mode=True,  # Comment if not needed - added to see the granularity for debug like retrieved context from vectodb
    markdown=True,
    
)

# Load knowledge base if needed
if agent.knowledge is not None:
    agent.knowledge.load()

# ========================================================
# Generate Response and Ensure Valid JSON
# ========================================================
query = "I love organs type of food as I'm pretty adventurous, what do you recommend me to eat on my first day in Singapore because I'm staying at town area?"
query = "recommend me food places as i have sweet tooth."
query = "I hate bland food, but something spicy would be nice. what do you think its nice in Sg?"
response = agent.run(query,
                     stream=False)  # Streaming disabled to capture full response

# Ensure response is valid JSON
print()
print(query)
print()
try:
    json_response = json.loads(response.content)
    print()
    print(f"✅ Successful json") 
    print(json.dumps(json_response, indent=4))  # Pretty-print the JSON response
    print()
except json.JSONDecodeError:
    print()
    print("⚠️ Invalid JSON response received:")
    print()
    print(response.content)