
import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv

from agentic.multiagent import create_intent_agent, create_hawker_agent, create_attraction_agent, create_itinerary_agent
# ========================================================
# Load environment variables 
# ========================================================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# ========================================================
# Experiment Examples E_i
# ========================================================
user_queries = {
    "01": {
        "query": "We’re a family of four visiting Singapore for 3 days. We’d love to explore kid-friendly attractions and try some affordable local food. Budget is around 300 SGD.",
        "days": 3,
        "budget": 300,
    },
    "02": {
        "query": "I'm a solo backpacker staying for 3 days. My budget is tight (~150 SGD total), and I'm mainly here to try spicy food and explore free attractions.",
        "days": 3,
        "budget": 150,
    },
    "03": {
        "query": "I’ll be spending 3 days in Singapore and I'm really interested in cultural attractions and sampling traditional hawker food on a modest budget. Budget is 180 SGD.",
        "days": 3,
        "budget": 180,
    },
    "04": {
        "query": "I'm visiting Singapore for 3 days as a content creator. I'm looking for Instagrammable attractions and stylish food spots. Budget is 600 SGD.",
        "days": 3,
        "budget": 600,
    },
    "05": {
        "query": "I love adventure and spicy food! Spending 3 days in Singapore. What attractions and hawker stalls should I visit? Budget is 200 SGD.",
        "days": 3,
        "budget": 200,
    },
    "06": {
        "query": "Looking to relax and enjoy greenery and peaceful spots in Singapore. I’ll be there for 3 days and have 190 SGD to spend. I enjoy light snacks over heavy meals.",
        "days": 3,
        "budget": 190,
    },
    "07": {
        "query": "What can I do in Singapore in 3 days if I love shopping and modern city vibes? I’d also like to eat at famous food centres. Budget is 270 SGD.",
        "days": 3,
        "budget": 270,
    },
    "08": {
        "query": "My spouse and I are retired and visiting Singapore for 3 days. We love cultural sites and relaxing parks. Prefer to avoid loud or overly touristy spots. Budget is 210 SGD.",
        "days": 3,
        "budget": 210,
    },
    "09": {
        "query": "We’re a group of university students spending 3 days in Singapore on a budget of 180 SGD total. Recommend cheap eats and fun, free things to do.",
        "days": 3,
        "budget": 180,
    },
    "10": {
        "query": "This is my first time in Singapore and I’ll be here for 3 days. I’d like a mix of sightseeing, must-try foods, and some local experiences. Budget is 250 SGD.",
        "days": 3,
        "budget": 250,
    }
}
#==================

def verify_and_get_cost_metrics(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)

    recomputed_cost = 0.0
    for day in data.get("itinerary", []):
        for activity in day.get("activities", []):
            recomputed_cost += activity.get("estimated_cost", 0)

    recomputed_cost = round(recomputed_cost, 2)
    reported_cost = round(data.get("summary", {}).get("total_cost_sgd", -1), 2)

    cost_match = recomputed_cost == reported_cost
    cost_abs_error = abs(reported_cost - recomputed_cost)
    cost_pct_error = cost_abs_error / max(recomputed_cost, 1e-5)

    return {
        "cost_match": cost_match,
        "abs_error": cost_abs_error,
        "pct_error": cost_pct_error
    }

def evaluate_cost_only(subfolder_path="results/agentic_rag"):
    scenarios = [f"{i:02}" for i in range(1, 11)]  # 01 to 10

    cost_correct = 0
    total_cost_error = 0
    total_cost_pct_error = 0
    count = 0

    print("🔍 Evaluating Cost Accuracy Only...\n")

    for scenario in scenarios:
        json_path = os.path.join(subfolder_path, f"{scenario}/agent_itinerary.json")

        if not os.path.exists(json_path):
            print(f"⚠️  Missing: {json_path}")
            continue

        result = verify_and_get_cost_metrics(json_path)
        cost_correct += int(result["cost_match"])
        total_cost_error += result["abs_error"]
        total_cost_pct_error += result["pct_error"]
        count += 1

        print(f"Scenario {scenario}: Cost Match: {'✅' if result['cost_match'] else '❌'} | Error: {result['abs_error']:.2f} SGD")

    if count > 0:
        print("\n📊 Final Cost Accuracy Report:")
        print(f"  Cost Accuracy:         {cost_correct}/{count}  ({cost_correct / count:.0%})")
        print(f"  Mean Absolute Error:   {total_cost_error / count:.2f} SGD")
        print(f"  Mean Percentage Error: {total_cost_pct_error / count:.2%}")
    else:
        print("⚠️ No valid scenarios found.")

if __name__ == "__main__":
    # print()

    debug_mode=True
    intent_agent = create_intent_agent()
    hawker_agents = [create_hawker_agent(batch_no=i, debug_mode=debug_mode) for i in range(2)]
    attraction_agents = [create_attraction_agent(batch_no=i, debug_mode=debug_mode) for i in range(7)]
    
    for scenario, query_item in tqdm(user_queries.items(), desc="Generating itineraries", unit="Itinerary") :
        start_time = time.time()
        print(scenario, query_item)
        intent_response = intent_agent.run(query_item["query"], stream=False)
        intent = intent_response.content.intent

        print(f"\n🔍 Processing Query: {query_item['query']}")

        if intent == "malicious":
            print("⚠️ Query flagged as malicious. Skipping...")
            continue

        responses = {
            "Query": query_item,
            "Hawker": [],
            "Attraction": []
        }

        if intent in ["food", "both"]:
            for hawker_agent in hawker_agents:
                hawker_output = hawker_agent.run(query=query_item["query"], stream=False).content.model_dump()
                # process in batches
                hawker_recs = hawker_output["HAWKER_DETAILS"]

                for hawker in hawker_recs:
                    if hawker["hawker_name"] in [x["Hawker Name"] for x in responses["Hawker"]]:
                        print(f"WARN: Duplicate Hawker {hawker['hawker_name']}")
                        continue
            
                    responses["Hawker"].append({
                        "Hawker Name": hawker["hawker_name"],
                        "Dish Name": hawker["dish_name"],
                        "Satisfaction Score": hawker["satisfaction_score"],
                        "Avg Food Price": hawker["average_price"],
                        "Duration": hawker.get("duration", 60)
                    })

        if intent in ["attraction", "both"]:
            for attraction_agent in attraction_agents:
                attraction_output = attraction_agent.run(query=query_item["query"], stream=False).content.model_dump()
                # process in batches
                attraction_recs = attraction_output["ATTRACTION_DETAILS"]

                for attraction in attraction_recs:
                    if attraction["attraction_name"] in [x["Attraction Name"] for x in responses["Attraction"]]:
                        print(f"WARN: Duplicate Attraction {attraction['attraction_name']}")
                        continue

                    responses["Attraction"].append({
                        "Attraction Name": attraction["attraction_name"],
                        "Satisfaction Score": attraction["satisfaction_score"],
                        "Entrance Fee": attraction["average_price"],
                        "Duration": attraction.get("duration", 120),
                    })
        
        subfolder_path = "results/agentic_rag"

        poi_path = os.path.join(subfolder_path, f"{scenario}/POI_data.json")
        os.makedirs(subfolder_path+f"/{scenario}", 
                    exist_ok=True)

        with open(poi_path, "w", encoding="utf-8") as f:
            json.dump(responses, f, indent=4)

        # with open(poi_path, 'r', encoding='utf-8') as f:
        #     responses = json.load(f)

        itinerary_agent = create_itinerary_agent(hawkers=responses["Hawker"], 
                                                 attractions=responses["Attraction"], 
                                                 model_id="gpt-4o", 
                                                 debug_mode=True
                                                 )
        
        itinerary_response = itinerary_agent.run(query=json.dumps(query_item), stream=False)
        # itinerary = itinerary_response.content
        # output_path = os.path.join(subfolder_path, f"{scenario}/agent_itinerary.txt")
        # with open(output_path, 'w') as f:
        #     f.write(itinerary)

        # print(itinerary)
        # print(f"Saved itinerary to: {output_path}")
        end_time = time.time()
        print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
        latency = round(end_time-start_time,2)
        latency_path = os.path.join(subfolder_path, f"{scenario}/latency.txt")
        with open(latency_path, "w") as f:
            f.write(str(latency))
        # Serialize Pydantic model to JSON string
        itinerary_json_str = itinerary_response.content.model_dump_json(indent=2)

        # Save to file
        output_path = os.path.join(subfolder_path, f"{scenario}/agent_itinerary.json")
        with open(output_path, 'w') as f:
            f.write(itinerary_json_str)

        # Optional: print or preview
        print(itinerary_json_str)
        print(f"Saved itinerary to: {output_path}")

        print("*"*100)

    evaluate_cost_only()