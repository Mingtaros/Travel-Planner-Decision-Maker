from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq

from pydantic import BaseModel, Field
from typing import List

import json
import datetime
import os

from dotenv import load_dotenv
import os
# ========================================================
# Load environment variables
# ========================================================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Intent Classification Model
class IntentResponse(BaseModel):
    intent: str = Field(..., description="The detected intent of the query. Options: 'food', 'attraction', 'both'.")

# Create the Intent Classification Agent
intent_agent = Agent(
    name="Intent Classification Agent",
    agent_id="intent_classification_agent",
    model=Groq(id="deepseek-r1-distill-llama-70b", response_format="json", temperature=0.0),  
    response_model=IntentResponse,  # Enforce structured JSON output
    structured_outputs=True,
    description="You are an expert in understanding the user's intent from the query. Classify the user's query into 'food', 'attraction', or 'both' for routing.",
    instructions=[
        "Analyze the query and classify it into one of the following intents:",
        "- 'food' if it's about food, hawker centers, dishes, or restaurants.",
        "- 'attraction' if it's about places to visit, sightseeing, or landmarks.",
        "- 'both' if it's about both food and attractions in the same query.",
        "- 'unknown' if the query is unclear and needs clarification.",
        "Return only the detected intent as a structured JSON response."
    ],
)

#### User query (This is the single input from the user)
query = "I want to explore Chinatown and also find the best hawker stalls for chicken rice."
# query = "Where is the best place for cultural visits"
# query = "Give me an itinerary for a 5D4N trip with my family, we love to eat spicy food"


# Step 1: Use Intent Agent to classify the query
intent_response = intent_agent.run(query, stream=False)
intent = intent_response.content.intent
print(intent)

# # Initialize response dictionary
# responses = {"QUERY": query}

# # Step 2: Route the query to the correct agents
# if intent in ["food", "both"]:
#     # responses["food"] = hawker_agent.run(query, stream=False).content.model_dump()


# if intent in ["attraction", "both"]:
#     # responses["attraction"] = attraction_agent.run(query, stream=False).content.model_dump()

# # Step 3: Save Response to JSON File
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# output_dir = "data/combined_outputs"
# os.makedirs(output_dir, exist_ok=True)
# output_filename = os.path.join(output_dir, f"{timestamp}.json")

# # Pretty-print and save JSON response
# print("\n✅ Final JSON Response:")
# print(json.dumps(responses, indent=4))

# with open(output_filename, "w", encoding="utf-8") as json_file:
#     json.dump(responses, json_file, indent=4, ensure_ascii=False)

# print(f"\n📁 JSON response successfully saved to: {output_filename}")