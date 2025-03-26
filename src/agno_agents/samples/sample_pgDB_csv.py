from agno.knowledge.csv import CSVKnowledgeBase
from agno.vectordb.pgvector import PgVector

csv_kb = CSVKnowledgeBase(
    path="data/locationData/csv/",
    # Table name: ai.csv_documents
    vector_db=PgVector(
        table_name="sg_attraction_hawker",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    ),
)

from agno.agent import Agent
from knowledge_base import knowledge_base

agent = Agent(
    knowledge=csv_kb,
    search_knowledge=True,
)
agent.knowledge.load(recreate=False)

agent.print_response("Ask me about something from the knowledge base")