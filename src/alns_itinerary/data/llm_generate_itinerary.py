"""
Usage:
    src/alns_itinerary/data/llm_batch_process_js.py
to obtain all the various persona, satisfaction score in preparation for itinerary generaton using llm
The json personas can be obtained via data/alns_inputs/groq

Then next, we run
    src/alns_itinerary/data/llm_generate_itinerary_js.py
to generate the itinerary based on user inputs like persona, days, budget, and query
"""

import json
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

import time

from tqdm import tqdm

load_dotenv()

# --------------------------------------
# Configuration
# --------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "deepseek-r1-distill-llama-70b")
MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "deepseek-r1-distill-qwen-32b")


# --------------------------------------
# Function to Load Persona JSON
# --------------------------------------
def load_persona_data(persona: str) -> dict:
    persona_key = persona.lower().replace(" ", "_")
    path = f"./data/alns_inputs/groq/{persona_key}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# --------------------------------------
# Function to Generate Itinerary using LLM
# --------------------------------------
def generate_llm_itinerary(persona: str, budget: int, days: int, query: str) -> dict:
    data = load_persona_data(persona)
    attractions = data.get("attractions", [])
    hawkers = data.get("hawkers", [])

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        # temperature=0.4
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert Singapore travel planner.

    Your goal is to generate a strictly valid JSON itinerary for a {persona} visiting Singapore for {days} days on a budget of {budget} SGD.

    🧭 USER QUERY:
    \"\"\"{query}\"\"\"

    📌 INSTRUCTIONS:
    1. ONLY use the provided attractions and hawker stalls to create the itinerary.
    2. Each day must include:
    - 3 meals (breakfast, lunch, dinner) from hawker stalls.
    - As many high-score attractions as possible without exceeding the daily time and budget.
    3. Carefully plan to **balance across days**, avoid overlapping schedules, and minimize idle time.
    4. Respect the TOTAL budget constraint.
    5. 🚫 Do NOT write any explanation or reasoning in your response.

    🎯 OUTPUT FORMAT — return **only** a JSON object, no commentary or extra text:

    {{
    "Day 1": [
        {{"type": "attraction", "name": "...", "cost": <number>, "duration": <minutes>}},
        {{"type": "hawker", "name": "...", "cost": <number>, "duration": <minutes>}},
        ...
    ],
    ...
    "Total Cost": <number>
    }}

    ⚠️ VERY IMPORTANT:
    - Output MUST be valid JSON starting and ending with curly braces.
    - Do NOT write "Here is your itinerary" or any explanation.
    - DO NOT output markdown, headings, or notes.
    - Only output valid parsable JSON.
    """
        ),
        (
            "human",
            "Attractions: {attractions}\n\nHawkers: {hawkers}"
        )
    ])

    chain = prompt | llm | JsonOutputParser()

    start_time = time.time()

    response = chain.invoke({
        "persona": persona,
        "budget": budget,
        "days": days,
        "query": query,
        "attractions": attractions,
        "hawkers": hawkers
    })

    duration = round(time.time() - start_time, 2)

    # Inject input metadata
    response["meta"] = {
        "persona": persona,
        "budget": budget,
        "days": days,
        "query": query,
        "generation_time_seconds": duration
    }

    return response


def validate_total_cost(itinerary: dict, verbose: bool = True) -> bool:
    """
    Validates whether the computed cost from the itinerary matches the 'Total Cost' field.

    Args:
        itinerary (dict): The LLM-generated itinerary JSON.
        verbose (bool): Whether to print mismatch info.

    Returns:
        bool: True if valid, False otherwise.
    """
    computed_total = 0.0

    for day, events in itinerary.items():
        if day.startswith("Day") and isinstance(events, list):
            for item in events:
                cost = item.get("cost", 0)
                if not isinstance(cost, (int, float)):
                    if verbose:
                        print(f"⚠️ Skipping non-numeric cost in {day}: {item}")
                    continue
                computed_total += float(cost)

    reported_total = itinerary.get("Total Cost", 0.0)

    if round(computed_total, 2) != round(reported_total, 2):
        if verbose:
            print(f"❌ Mismatch in total cost!")
            print(f"👉 Reported: {reported_total}, Computed: {computed_total}")
        return False

    if verbose:
        print(f"✅ Total cost validated: {reported_total}")
    return True
# --------------------------------------
# CLI Entry Point
# --------------------------------------

user_queries = {
    "01": {
        "query": "We’re a family of four visiting Singapore for 3 days. We’d love to explore kid-friendly attractions and try some affordable local food. Budget is around 300 SGD.",
        "days": 3,
        "budget": 300,
        "persona": "Family Tourist"
    },
    "02": {
        "query": "I'm a solo backpacker staying for 3 days. My budget is tight (~150 SGD total), and I'm mainly here to try spicy food and explore free attractions.",
        "days": 3,
        "budget": 150,
        "persona": "Backpacker"
    },
    "03": {
        "query": "I’ll be spending 3 days in Singapore and I'm really interested in cultural attractions and sampling traditional hawker food on a modest budget. Budget is 180 SGD.",
        "days": 3,
        "budget": 180,
        "persona": "Cultural Enthusiast"
    },
    "04": {
        "query": "I'm visiting Singapore for 3 days as a content creator. I'm looking for Instagrammable attractions and stylish food spots. Budget is 600 SGD.",
        "days": 3,
        "budget": 600,
        "persona": "Influencer"
    },
    "05": {
        "query": "I love adventure and spicy food! Spending 3 days in Singapore. What attractions and hawker stalls should I visit? Budget is 200 SGD.",
        "days": 3,
        "budget": 200,
        "persona": "Thrill Seeker"
    },
    "06": {
        "query": "Looking to relax and enjoy greenery and peaceful spots in Singapore. I’ll be there for 3 days and have 190 SGD to spend. I enjoy light snacks over heavy meals.",
        "days": 3,
        "budget": 190,
        "persona": "Nature Lover"
    },
    "07": {
        "query": "What can I do in Singapore in 3 days if I love shopping and modern city vibes? I’d also like to eat at famous food centres. Budget is 270 SGD.",
        "days": 3,
        "budget": 270,
        "persona": "Shopping Enthusiast"
    },
    "08": {
        "query": "My spouse and I are retired and visiting Singapore for 3 days. We love cultural sites and relaxing parks. Prefer to avoid loud or overly touristy spots. Budget is 210 SGD.",
        "days": 3,
        "budget": 210,
        "persona": "Cultural Enthusiast"
    },
    "09": {
        "query": "We’re a group of university students spending 3 days in Singapore on a budget of 180 SGD total. Recommend cheap eats and fun, free things to do.",
        "days": 3,
        "budget": 180,
        "persona": "Backpacker"
    },
    "10": {
        "query": "This is my first time in Singapore and I’ll be here for 3 days. I’d like a mix of sightseeing, must-try foods, and some local experiences. Budget is 250 SGD.",
        "days": 3,
        "budget": 250,
        "persona": "Family Tourist"
    }
}

if __name__ == "__main__":
    print()
    for scenario, query_item in tqdm(user_queries.items(), desc="Generating itineraries", unit="Itinerary") :
        itinerary = generate_llm_itinerary(
            persona=query_item["persona"],
            budget=query_item["budget"],
            days=query_item["days"],
            query=query_item["query"]
        )
        # ✅ Validate cost
        is_valid = validate_total_cost(itinerary)
        if not is_valid:
            print("⚠️ Warning: LLM-generated itinerary has incorrect total cost.")

        # Save the result
        out_path = f"./llm/{scenario}/itinerary_{query_item['persona'].lower().replace(' ', '_')}.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(itinerary, f, indent=4)

        print(f"\n✅ LLM Itinerary saved to {out_path}")
        print(f"🕒 Generation time: {itinerary['meta']['generation_time_seconds']}s")

        print()



