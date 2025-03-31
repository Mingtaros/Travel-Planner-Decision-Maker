import os
import logging
from datetime import datetime
import sys

from alns_main import alns_main
from data.llm_batch_process import process_and_save

load_dotenv()

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
            logging.FileHandler(f"log/evaluate_alns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)

personas = [
    "Family Tourist", "Backpacker", "Influencer", "Cultural Enthusiast", 
    "Thrill Seeker", "Nature Lover", "Shopping Enthusiast"
]

def alns_evaluate(user_input):

    batch_size = 10
    alns_input = None
    alns_input = process_and_save(
        persona=user_input['persona'],
        attraction_path="./data/locationData/singapore_67_attractions_with_scores.csv", 
        hawker_path="./data/locationData/Food_20_withscores.xlsx", 
        output_json_path="./data/alns_inputs/groq/location_data.json", 
        batch_size=batch_size,
    )

    alns_data = alns_main(user_input=user_input, alns_input=alns_input)
    logger.info("Itinerary data loaded successfully!")
    
    return alns_data

if __name__ == "__main__":
    
    persona = "Shopping Enthusiast"
    num_days = 3
    budget = 500
    description = 'I love shopping and spending money. I want to enjoy my time here in Singapore with unique local food and fun attractions.'
    
    user_input = {"persona": persona, "num_days": num_days, "budget": budget, "description": description}
    
    alns_data = alns_evaluate(
        user_input=user_input
    )