"""
Level 2: Intelligent agents with memory and reasoning capabilities. (SINGLE AGENT EXAMPLE)

These agents utilize a vector database to store knowledge, enabling them to perform Retrieval-Augmented Generation (RAG) or dynamic few-shot learning.

By default, Agno agents employ Agentic RAG, where they query their knowledge base for specific details required to complete tasks. 
In this example, the RAG process uses a connected PDF URL as the data source and leverages hybrid search instead of purely semantic search.

Alongside the agent's high-level description (system prompt), you can provide detailed instructions to fine-tune its behavior.
A single Agentic RAG agent is equipped with two key components: tools and knowledge/memory capabilities.
This allows for a dynamic workflow where queries are refined as needed, knowledge is retrieved, and responses are generated accordingly.

This example focuses on recommending food.
But what if you wanted to find hotels instead?
To handle such diverse tasks, you would need something more advanced than Level 2 agents.
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
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.document.chunking.fixed import FixedSizeChunking
from agno.document.chunking.agentic import AgenticChunking
from agno.document.chunking.semantic import SemanticChunking

from agno.embedder.openai import OpenAIEmbedder
from dotenv import load_dotenv
import os

# ========================================================
# Load environment variables
# ========================================================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

pdf_urls = [
    "https://raw.githubusercontent.com/Mingtaros/Travel-Planner-Decision-Maker/main/data/Summary_Singapore_Food.pdf",
]

# ========================================================
# Dynamically Select Chunking Strategy
# ========================================================
chunking_options = {
    "fixed": FixedSizeChunking(chunk_size=200,overlap=20),
    "agentic": AgenticChunking(),
    "semantic": SemanticChunking(),
}

chunking_type_name = os.getenv("CHUNKING_TYPE", "fixed")  # Default to "fixed"
chunking_type = chunking_options.get(chunking_type_name, "fixed")

# ========================================================
# Persistent Storage Configuration
# ========================================================
# Include chunking strategy in storage identifiers
db_uri = "lancedb_data"
table_name = f"recipes_{chunking_type_name}"  # e.g., "recipes_fixed", "recipes_agentic"
storage_file = f"knowledge_base_{chunking_type_name}.db"

# ========================================================
# Print Debugging Information on Terminal
# ========================================================
print("\n=======================================")
print(f">> Chunking Strategy: {chunking_type_name.upper()}")
print(f">> PDF Sources: {pdf_urls}")
print(f">> Database URI: {db_uri}")
print(f">> Table Name: {table_name}")
print(f">> Persistent Storage File: {storage_file}")
print("=======================================\n")

# ========================================================
# Modify Instructions to Require JSON Output
# ========================================================
agent = Agent(
    model=OpenAIChat(id="gpt-4o", response_format="json",temperature=0.6),  # Ensures OpenAI outputs valid JSON
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
        vector_db=LanceDb(
            uri=db_uri,
            table_name=table_name,
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(id="text-embedding-3-small"),
        ),
    ),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    debug_mode=False,  # Comment if not needed - added to see the granularity for debug
    markdown=True
)

# Load knowledge base if needed
if agent.knowledge is not None:
    agent.knowledge.load()

# ========================================================
# Generate Response and Ensure Valid JSON
# ========================================================
response = agent.run("I love organs type of food as I'm pretty adventurous, what do you recommend me to eat on my first day in Singapore because I'm staying at Bedok?",
                     stream=False)  # Streaming disabled to capture full response

# Ensure response is valid JSON
try:
    json_response = json.loads(response.content)
    print(json.dumps(json_response, indent=4))  # Pretty-print the JSON response
except json.JSONDecodeError:
    print("⚠️ Invalid JSON response received:")
    print(response.content)