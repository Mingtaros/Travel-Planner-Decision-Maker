"""
Level 0: Agents without tools (basic inference capabilities).

These agents (level 0) are not ideal for tasks requiring real-time information, as they lack access to current events. 
However, they perform well for generative tasks similar to a standard language model.
Hallucination is the biggest problem here!

"""
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools


from dotenv import load_dotenv
import os

# ======================================================== START: TO SET THE VARIABLES  ========================================================
# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# ======================================================== END: TO SET THE VARIABLES  ==========================================================

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="You are an enthusiastic news reporter with a flair for storytelling!", # this is like the system prompt
    markdown=True
)
agent.print_response("What's happening in New York?", stream=True)