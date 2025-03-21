"""
To run the following command:
streamlit run pdf_rag_streamlit.py
"""
import streamlit as st
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
import time  # For simulating streaming effect

# ========================================================
# Load environment variables
# ========================================================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ========================================================
# Streamlit UI Header
# ========================================================
st.title("📄 AI-Powered PDF RAG Chatbot")
st.markdown("Upload PDFs or use existing GitHub-hosted documents to ask AI-powered questions.")

# ========================================================
# User Input: Upload PDF or Use GitHub PDFs
# ========================================================
pdf_urls = [
    "https://raw.githubusercontent.com/sciencenerd880/LLM-DBT/main/data/pdfs/hbs_opportunities.pdf",
    "https://raw.githubusercontent.com/sciencenerd880/LLM-DBT/main/data/pdfs/hbs_pitchdeck_sample.pdf",
]

uploaded_files = st.file_uploader("📂 Upload PDFs", accept_multiple_files=True, type=["pdf"])

# ========================================================
# Choose Chunking Strategy
# ========================================================
chunking_options = {
    "fixed": FixedSizeChunking(),
    "agentic": AgenticChunking(),
    "semantic": SemanticChunking(),
}
chunking_type_name = st.selectbox("🔍 Choose Chunking Strategy", list(chunking_options.keys()))
chunking_type = chunking_options[chunking_type_name]

# ========================================================
# Set Storage & Database
# ========================================================
agent_storage_file: str = "tmp/pdf_rag.db"
chroma_db_path = "./chromadb_data"
table_name_agent = f"HBS_{chunking_type_name}_agent"
collection_name = f"HBS_{chunking_type_name}"

vector_db = ChromaDb(collection=collection_name, path=chroma_db_path, persistent_client=True)

knowledge_base = PDFUrlKnowledgeBase(urls=pdf_urls, vector_db=vector_db, chunking_strategy=chunking_type)

# ========================================================
# Initialize PDF RAG Agent
# ========================================================
groq_model_name = st.selectbox("🧠 Select LLM Model", ["deepseek-r1-distill-llama-70b", "llama3-70b-8192"])
pdf_rag_agent = Agent(
    model=Groq(id=groq_model_name),
    storage=SqliteAgentStorage(table_name=table_name_agent, db_file=agent_storage_file),
    show_tool_calls=True,
    debug_mode=True,
    knowledge=knowledge_base,
)

# ========================================================
# Load Knowledge Base
# ========================================================
st.write("📚 **Loading knowledge base...** This may take a few seconds.")
knowledge_base.load(recreate=False)
st.success("✅ Knowledge base loaded!")

# ========================================================
# Chat Interface with Streaming Effect
# ========================================================
st.subheader("💬 Ask a Question")

user_input = st.text_input("Enter your question:")
if st.button("Ask"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a question.")
    else:
        st.markdown("### 🤖 AI Response:")

        # Streaming placeholder
        response_placeholder = st.empty()
        full_response = ""

        # Simulate streaming by updating text progressively
        for word in pdf_rag_agent.run(user_input).content.split():
            full_response += word + " "
            response_placeholder.markdown(full_response)
            time.sleep(0.05)  # Adjust speed of streaming

        # Ensure final response is shown
        response_placeholder.markdown(full_response)