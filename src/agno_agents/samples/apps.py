import gradio as gr
import time  # For simulating streaming effect
from agno.agent import Agent
from agno.models.groq import Groq
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
import os

# ========================================================
# Load environment variables
# ========================================================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ========================================================
# Define the Intent Response Model
# ========================================================
class IntentResponse(BaseModel):
    intent: str = Field(..., description="The detected intent of the query. Options: 'food', 'attraction', 'both', or 'unknown'.")
    hawker_query: Optional[str] = Field(None, description="Rewritten query for hawker recommendations, if applicable.")
    attraction_query: Optional[str] = Field(None, description="Rewritten query for attraction recommendations, if applicable.")

# ========================================================
# Create the Intent Classification & Query Rewriting Agent
# ========================================================
intent_agent = Agent(
    name="Intent Classification and Query Rewriting Agent",
    agent_id="intent_classification_agent",
    model=Groq(id="deepseek-r1-distill-llama-70b", response_format="json", temperature=0.0),  
    response_model=IntentResponse,  
    structured_outputs=True,
    description="Classify user query into 'food', 'attraction', or 'both' and rewrite it.",
    instructions=[
        "Classify the query into:",
        "- 'food' → about hawker centers, dishes, or restaurants.",
        "- 'attraction' → about places to visit, sightseeing, or landmarks.",
        "- 'both' → discussing both food and attractions.",
        "- 'unknown' → unclear query.",
        "",
        "Rewrite the query accordingly.",
        """
        {
            "intent": "<food|attraction|both|unknown>",
            "hawker_query": "<Rewritten query for hawker OR null>",
            "attraction_query": "<Rewritten query for attraction OR null>"
        }
        """
    ],
)

# ========================================================
# Streaming Function for AI Query Processing
# ========================================================
def process_query_streaming(query):
    """
    Processes a free-text query from 'Ask Anything' and streams responses.
    """
    if not query.strip():
        yield "⚠️ Please enter a valid query."
        return

    yield "🤖 Analyzing your query..."
    time.sleep(0.5)

    # 🔍 Step 1: Use Intent Agent to Classify & Rewrite Query
    intent_response = intent_agent.run(query, stream=False)

    # Extract JSON response
    if hasattr(intent_response.content, "model_dump"):
        intent_data = intent_response.content.model_dump()  
    else:
        intent_data = intent_response.content  

    # Extract relevant details
    detected_intent = intent_data.get("intent", "unknown")
    hawker_query = intent_data.get("hawker_query", None)
    attraction_query = intent_data.get("attraction_query", None)

    # Streaming Response
    yield f"🎯 Detected intent: **{detected_intent}**"
    time.sleep(0.5)

    if detected_intent == "food":
        yield f"🍽️ Finding the best food spots...\n🔍 **Refined Query:** {hawker_query}"
    elif detected_intent == "attraction":
        yield f"🏛️ Looking for amazing attractions...\n🔍 **Refined Query:** {attraction_query}"
    elif detected_intent == "both":
        yield f"🌆 Your trip includes both food & attractions!"
        yield f"🍽️ **Food Query:** {hawker_query}"
        yield f"🏛️ **Attraction Query:** {attraction_query}"
    else:
        yield "🤔 I'm not sure what you're looking for. Can you clarify?"

# ========================================================
# Gradio UI with Streaming
# ========================================================
with gr.Blocks(title="AI Travel Planner") as demo:
    gr.HTML("<h1 style='text-align: center;'>🌍 Get <span style='color:#FFD700;'>Inspired</span> for Your Next Trip ✈️</h1>")

    with gr.Tabs():
        with gr.TabItem("Ask Anything"):
            gr.Markdown("### 🔍 Type your travel query below")
            query_input = gr.Textbox(label="Your Travel Question", placeholder="Example: 'Where can I find the best spicy food in Singapore?'")
            ask_button = gr.Button("Ask AI")
            ask_output = gr.Textbox(label="AI Response", interactive=False)

            # Enable streaming
            ask_button.click(process_query_streaming, inputs=[query_input], outputs=ask_output, stream=True)

        with gr.TabItem("Personalize My Trip"):
            gr.Markdown("### 🛫 Customize Your Travel Plan")
            
            # Input Fields
            destination = gr.Textbox(label="Where are you traveling to?", value="Singapore")
            days = gr.Slider(minimum=1, maximum=14, step=1, value=4, label="How many days?")
            
            companions = gr.CheckboxGroup(
                ["Solo", "Partner", "Group", "Kids", "Elderly"], 
                label="Who are you traveling with?"
            )
            
            interests = gr.CheckboxGroup(
                ["Adventure", "Arts", "Culture", "Food", "Wellness", "Gems", "Iconic", "Nature"],
                label="What are you interested in?"
            )

            # Optional Extra Preferences
            custom_input = gr.Textbox(
                label="Anything else to make your trip more interesting? (Optional)", 
                placeholder="Example: 'I love sunset views' or 'I prefer budget-friendly options'"
            )

            # Generate Button
            generate_button = gr.Button("✨ Create Itinerary")
            itinerary_output = gr.Textbox(label="AI Itinerary", interactive=False)

            # Click event with streaming
            generate_button.click(process_query_streaming, 
                                  inputs=[destination], 
                                  outputs=itinerary_output, 
                                  stream=True)

# Launch App
if __name__ == "__main__":
    demo.launch()