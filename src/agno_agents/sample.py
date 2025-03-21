from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

local_agent = Agent(
    model=Groq(id="qwen-2.5-32b"),
    description="You are a helpful travel assistant that is specialised in Singapore Travel Itinerary",
    tools=[DuckDuckGoTools()],
    markdown=True
)

query = "what is the latest local food recommendation"
local_agent.print_response(query)

