"""
This is an implementation of an agentic RAG using agno library package and Groq as our LLM API provider. 
Open Source LLM provider like HuggingFace is also possible.
The assumption of this system is where the data sources are PDFs stored in GitHub Repo (for simplicity) instead of Cloud DB.

The package itself is agnostic to model and LLM service provider, feel free to make the necessary changes.
Agno also allows interfaces to various vector db providers besides ChromaDB such as Pinecone, LanceDB, etc. 
============================================================================================================================================================================

USAGE AS FOLLOWS:
0) Ensure that your .env environmental variables is created. For this case, we need GROQ_API_KEY=XXXX inside .env file
1) Set the necessary variables
2) Comment out knowledge_base.load(recreate=False) if already vectorized into the DB
3) Choose one of the chunking type, and copy paste into your terminal
export CHUNKING_TYPE="fixed" or 
export CHUNKING_TYPE="agentic" or 
export CHUNKING_TYPE="semantic"
4) run the following command to execute: 

python src/agno_agents/pdf_rag.py
"""

import typer
from rich.prompt import Prompt
from typing import Optional

from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from agno.document.chunking.fixed import FixedSizeChunking
from agno.document.chunking.agentic import AgenticChunking
from agno.document.chunking.semantic import SemanticChunking

from agno.storage.agent.sqlite import SqliteAgentStorage

from agno.agent import Agent
from agno.models.groq import Groq

from dotenv import load_dotenv
import os

# ======================================================== START: TO SET THE VARIABLES  ========================================================
# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# Define PDF RAG agent storage file & table name
agent_storage_file: str = "tmp/pdf_rag.db"

# Use persistent storage for ChromaDB
chroma_db_path = "./chromadb_data"

# PDF URLs
pdf_urls = [
    "https://raw.githubusercontent.com/Mingtaros/Travel-Planner-Decision-Maker/main/data/Summary_Singapore_Food.pdf",
]

# Allow selecting different chunking strategies dynamically
chunking_options = {
    "fixed": FixedSizeChunking(),
    "agentic": AgenticChunking(),
    "semantic": SemanticChunking(),
}

chunking_type_name = os.getenv("CHUNKING_TYPE", "fixed")  # Default to "fixed"
chunking_type = chunking_options.get(chunking_type_name, FixedSizeChunking())

# Create unique storage names for different chunking strategies | stored at agent_storage_file .db
#  TPI stands for Travel Planner Itinerary
table_name_agent = f"TPI_{chunking_type_name}_agent"
collection_name = f"TPI_{chunking_type_name}"

# Debugging Information
print("\n=======================================")
print(f">>Chunking Strategy: {chunking_type_name.upper()}")
print(f">>Agent Storage File: {agent_storage_file}")
print(f">>Agent Table Name: {table_name_agent}")
print(f">>ChromaDB Collection Name: {collection_name}")
print(f">>ChromaDB Storage Path: {chroma_db_path}")
print(f">>PDFs Being Processed:")
for url in pdf_urls:
    print(f"   - {url}")
print("=======================================\n")

# ======================================================== END: TO SET THE VARIABLES  ==========================================================

vector_db = ChromaDb(
    collection=collection_name, 
    path=chroma_db_path,
    persistent_client=True   # Enable persistence
)

# Defines the knowledge base
knowledge_base = PDFUrlKnowledgeBase(
    urls=pdf_urls,
    vector_db=vector_db, 
    chunking_strategy=chunking_type
)   

def pdf_agent(user: str = "user", groq_model: str = "deepseek-r1-distill-llama-70b"):
    groq_agent = Agent(
        model=Groq(id=groq_model),
        storage=SqliteAgentStorage(
            table_name=table_name_agent, 
            db_file=agent_storage_file
        ),
        show_tool_calls=True,
        debug_mode=True,
        knowledge=knowledge_base
    )
    
    print("\n RAG Agent Setup Complete!! ")
    print(f"\n Using Model: {groq_model}")
    print(f"\n Chunking Method: {chunking_type_name.upper()}")
    print(f"\n Storing vectors in: {collection_name} (ChromaDB)")
    print(f"\n Agent history stored in: {agent_storage_file} (SQLite Table: {table_name_agent})\n")

    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message in ("exit", "bye"):
            break
        groq_agent.print_response(message)

if __name__ == "__main__":
    ## Comment out after first run
    knowledge_base.load(recreate=False)

    # Initialise and run pdf agent
    groq_model_name = "llama3-70b-8192"  # or "deepseek-r1-distill-llama-70b", "qwen-qwq-32b", "llama3-70b-8192", "qwen-2.5-32b"
    typer.run(pdf_agent(groq_model=groq_model_name))