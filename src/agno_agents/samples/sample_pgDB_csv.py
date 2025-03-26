"""
usage:
python src/agno_agents/samples/sample_pgDB_csv.py
"""

from agno.knowledge.csv import CSVKnowledgeBase
from agno.vectordb.pgvector import PgVector

from dotenv import load_dotenv
import os
from pydantic import BaseModel

from agno.models.openai import OpenAIChat
from agno.agent import Agent

class SuitabilityResponse(BaseModel):
    score_attraction_suitability: int
    score_food_suitability: int

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

csv_kb = CSVKnowledgeBase(
    path="data/locationData/csv/",
    # Table name: ai.csv_documents
    vector_db=PgVector(
        table_name="sg_attraction_hawker",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    ),
)

# satisfaction agent takes in web rating and the preference to compute the satisfaction
### for now use attraction to test first 
satisfaction_agent = Agent(
    name="Satisfaction Suitability Agent",
    model=OpenAIChat(
        id="gpt-4o",  # or any model you prefer
        response_format="json", # depends what we want 
        temperature=0.1,
    ),
    agent_id="suitability_agent",
    description="You are an expert in understanding based on the traveller type, if you need to look up for suitability score of attraction table and food table. Returns only the suitability score (1-10) of a location & food for a specific traveler type.",
    knowledge=csv_kb,
    instructions=[
        # "WARNING: You should not mixed up both",
        "Search the knowledge base and return ONLY the following keys as a JSON:",
        "- score_attraction_suitability: value between 0 and 10 (0 if not found)",
        "- score_food_suitability: value between 0 and 10 (0 if not found)",
        "Do not return any explanation. Return only valid JSON."],
    search_knowledge=True,
)

# agent.knowledge.load(recreate=False)

attraction = "Universal Studios Singapore" #1st input 
hawker = "Hill Street Tai Hwa Pork Noodle" #2nd input
# hawker="" #2nd input
traveller_type = "backpacker" #3rd input

# query = f"Only return the numerical suitability score (1-10) of {attraction} {hawker} for {traveller_type}. Do not add any explanation."
query = "We’re a family of four visiting Singapore for 3 days. We’d love to explore kid-friendly attractions and try some affordable local food. Budget is around 300 SGD.",

# agent.print_response(
#     query
# )

response = satisfaction_agent.run(query, stream=False).content
print(response)
## expecting the following
"""
```json
{
  "score_attraction_suitability": 1,
  "score_food_suitability": 0
}
```
"""


###
"""
satisfaction score (0-50) = web rating (0-5) * suitability/preference score (0-10)
"""