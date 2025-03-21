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

This example focuses on recommending Attraction.
But what if you wanted to find hotels instead?
To handle such diverse tasks, you would need something more advanced than Level 2 agents.

"""

"""
Expected output for Attractions

[
    {
        "name": (string) name of the attraction place,
        "satisfaction": (float) how satisfied will this persona be when going to this attraction. Range 1-10,
        "entrance_fee": (float) price of visiting the attractions in SGD
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
from agno.tools.googlesearch import GoogleSearchTools

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

class AttractionRecommendation(BaseModel):
    attraction_name: str = Field(..., description="The name of the attraction that is recommended.")
    description: str = Field(..., description="A short description of the attraction and why it's recommended.")
    average_price: float = Field(..., description="The maximum price in SGD of the attraction, retrieved from web sources.")
    ratings: float = Field(..., description="The Google rating of the attraction, range from 1 to 5.")
    sources: List[str] = Field(..., description="List of sources where information was retrieved.")

class AttractionResponse(BaseModel):
    # QUERY: str = Field(..., description="The user's original query for attraction recommendations.") 
    ATTRACTION_RECOMMENDATIONS: List[AttractionRecommendation] = Field(..., description="List of recommended attraction options.")

# ========================================================
# Load environment variables
# ========================================================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

pdf_urls = [
    # "https://raw.githubusercontent.com/Mingtaros/Travel-Planner-Decision-Maker/main/data/hawker/Summary_Singapore_Food.pdf",
    "https://raw.githubusercontent.com/Mingtaros/Travel-Planner-Decision-Maker/main/data/attraction/inputs/Singapore_Attractions_Guide.pdf"
]
# ========================================================
# Dynamically Select Chunking Strategy
# ========================================================
chunking_options = {
    "fixed": FixedSizeChunking(chunk_size=300, overlap=50),  # Example: Adjust chunk size if needed
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
collection_name = f"ATTRACTION_{chunking_type_name}"  # Example: "ATTRACTION_fixed"

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
attraction_agent = Agent(
    name="Query to Attraction Agent",
    agent_id="query_to_attraction_agent",
    model=OpenAIChat(id="gpt-4o", response_format="json",temperature=0.2,top_p=0.2),  # Hyperparams pre-defined, Ensures OpenAI outputs valid JSON
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
    knowledge=PDFUrlKnowledgeBase(
        urls=pdf_urls,
        chunking_strategy=chunking_type,
        vector_db=vector_db,
    ),
    search_knowledge=True,

    tools=[DuckDuckGoTools(search=True,
                           news=True,
                           fixed_max_results=5),
           GoogleSearchTools()],
    show_tool_calls=True,
    debug_mode=True,  # Comment if not needed - added to see the granularity for debug like retrieved context from vectodb
    markdown=True,
    
)

# Load knowledge base if needed
if attraction_agent.knowledge is not None:
    attraction_agent.knowledge.load()

# ========================================================
# Generate Response and Ensure Valid JSON
# ========================================================
# query = "I am visiting Singapore for 4 days with my family, including young kids. We love nature and iconic attractions. Can you suggest the best places to visit?" 
# query = "I’m a solo traveler visiting Singapore for 3 days. I love adventure and nature. What are some must-visit places for an exciting experience?"
# query = "I’m traveling with my friends for 5 days, and we are looking for some hidden gems and adventure activities. What should we explore?"
# query = "I am visiting Singapore with my partner for 4 days, and we are huge food lovers. Can you suggest attractions near the best hawker centers?"
# query = "I love exploring arts and culture. I’ll be in Singapore for a week as a solo traveler. What museums, galleries, or historical places should I visit?"
query = "I have half a day to explore before my flight. I’d like to visit some iconic attractions near the city center. Where should I go?"
# query = "I am from the states, surprise me."
response = attraction_agent.run(query,
                     stream=False)  # Streaming disabled to capture full response

# ========================================================
# Prepare the folder to save the json into
# ========================================================

# Generate timestamp in "YYYYMMDD_HHMMSS" format
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Define the directory name using the timestamp
output_dir = f"data/attraction/outputs/"

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
    json_response["QUERY"] = query  

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