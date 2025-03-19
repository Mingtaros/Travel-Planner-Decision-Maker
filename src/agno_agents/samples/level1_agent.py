"""
Level 1: Agents with tools for autonomous task execution.

These agents (level 1) are ideal for tasks requiring real-time information, as they HAVE access to current events. 
When needed, the tool call will be invoked, we can print the tool call on the terminal as shown in this tutorial.
We should see mcuh more relevant infromation (less hallucination)
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
    markdown=True,
    tools=[DuckDuckGoTools()], # added tool to externally access info when required
    show_tool_calls=True, # show tool call invocation when executed
    debug_mode=True, # added to see the granularity


)
agent.print_response("What's happening in New York?", stream=True)