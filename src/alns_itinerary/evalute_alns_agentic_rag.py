import os
import logging
from datetime import datetime
import sys
import time

from alns_main import alns_main
from data.llm_batch_process import process_and_save
from multiagent import get_json_from_query, get_combine_json_data

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

def setup_logging():
    """
    Configure application logging.
    
    Sets up both file and console logging with timestamps and appropriate
    log levels. Log files are stored in the 'log' directory with filenames
    that include the current timestamp.
    """
    # Create logs directory if it doesn't exist
    os.makedirs("log", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"log/evaluate_alns_agentic_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)

def alns_evaluate(user_input):

    batch_size = 10
    max_rows = None
    alns_input = None
    
    start_time = time.time()
    
    get_json_from_query(query=user_input['description'], debug_mode=True)
    
    get_combine_json_data()
    
    multiagent_time = time.time()
    multiagent_duration = multiagent_time - start_time
    logger.info(f"Multi-Agent runs for {multiagent_duration:.2f} s")

    alns_data = alns_main(user_input=user_input,
                          llm_path="./data/alns_inputs/")
    alns_time = time.time()
    alns_duration = alns_time - multiagent_time

    logger.info(f"ALNS runs for {alns_duration:.2f} s")
    
    return alns_data

if __name__ == "__main__":
    
    for key in ["02"]: # ["06", "07", "08", "09", "10"]:
        
        logger.info(f"Processing query {key}")
        
        # Uncomment the next line to process all queries
        # alns_data = alns_evaluate(user_input=user_query)
        user_query = user_queries[key]
        
        user_input = {
            "num_days": user_query['days'], 
            "budget": user_query['budget'], 
            "description": user_query['query']
            }
        
        alns_data = alns_evaluate(
            user_input=user_input
        )