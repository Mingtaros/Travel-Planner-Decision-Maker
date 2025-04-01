import pandas as pd
import json
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Dict, Any
from dotenv import load_dotenv
import logging
from datetime import datetime
import time

load_dotenv()

PROMPT_DIR = './data/prompts/'
delay_seconds = 2  # Adjust this value based on API rate limits
    
logger = logging.getLogger(__name__)

def load_prompt(filename: str) -> str:
    """Load a prompt from a file."""
    try:
        file_path = PROMPT_DIR + filename
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        raise

def read_data(data_path: str, max_rows: int = None) -> pd.DataFrame:
    """Read and merge data from an Excel file and a CSV file."""
    if data_path.endswith(".xlsx"):
        df = pd.read_excel(data_path, nrows=max_rows)
    else:
        df = pd.read_csv(data_path, nrows=max_rows)
    return df

def batch_data(df: pd.DataFrame, batch_size: int) -> List[Dict[str, Any]]:
    """Split the dataframe into batches of a given size."""
    return [df.iloc[i:i + batch_size].to_dict(orient='records') for i in range(0, len(df), batch_size)]

def get_llm():
    """Initialize the Groq LLM instance."""
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=os.getenv("GROQ_MODEL_NAME", "deepseek-r1-distill-llama-70b"),
        temperature=float(os.getenv("GROQ_TEMPERATURE", 0.5)),
    )

def process_batch(persona: str, description: str, batch: List[Dict[str, Any]], location_type: str) -> Dict[str, Any]:
    """Send a batch of rows to the Groq API and return the JSON response."""
    llm = get_llm()
    system_prompt = load_prompt(f"enrich_{location_type}_prompt.txt")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Generate the output in JSON format")
    ])
    
    params = {
        "persona": persona,
        "description": description,
        "data": batch
    }
    
    chain = prompt | llm | JsonOutputParser()
    response = chain.invoke(params)
    
    return response

def generate_alns_weights(persona: str, description: str) -> Dict[str, Any]:
    """Generate ALNS weights for budget, travel time, and satisfaction."""
    llm = get_llm()
    system_prompt = load_prompt("generate_alns_weights_prompt.txt")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Generate the output in JSON format")
    ])

    params = {
        "persona": persona,
        "description": description
    }

    chain = prompt | llm | JsonOutputParser()
    response = chain.invoke(params)

    return response

def process_and_save(persona: str, description: str, attraction_path: str, hawker_path: str, output_json_path: str, batch_size: int, max_rows: int = None, llm_data_path: str = None) -> Dict[str, Any]:
    """Process data in batches, generate ALNS weights, and save the final JSON output."""
    
    if not llm_data_path:
    
        logger.info(f"Processing data for persona: {persona}")
        
        attraction_df = read_data(attraction_path, max_rows)
        attraction_df = attraction_df[["Attraction Name", "Typical Expenditure (SGD)", "Typical Time Spent (hours)"]]
        attraction_df.columns = ["name", "cost", "duration"]
        
        hawker_df = read_data(hawker_path, max_rows)
        hawker_df = hawker_df[["Name", "Ratings (Google Reviews)", "Highlights"]]
        hawker_df.columns = ["name", "rating", "description"]
        
        results = {
            "attractions": [],
            "hawkers": [],
            "alns_weights": {}
        }
        
        for location_type, df in [("attractions", attraction_df), ("hawkers", hawker_df)]:
            logger.info(f"Processing {location_type}...")
            batches = batch_data(df, batch_size)
            logger.info(f"Total batches: {len(batches)}")
            
            for batch in batches:
                logger.info(f"Processing batch {len(results[location_type]) + 1}...")
                response = process_batch(persona, description, batch, location_type)
                
                if isinstance(response, dict) and location_type in response:
                    results[location_type].extend(response[location_type])
                else:
                    logger.info(f"Warning: Unexpected response format for {location_type}: {response}")
                
                # Add delay before next API call
                logger.info(f"Waiting {delay_seconds} seconds before next API call...")
                time.sleep(delay_seconds)
        
        # Generate ALNS weights
        alns_weights = generate_alns_weights(persona, description)
        if isinstance(alns_weights, dict) and "alns_weights" in alns_weights:
            results["alns_weights"] = alns_weights["alns_weights"]
        else:
            logger.info(f"Warning: Unexpected response format for ALNS weights: {alns_weights}")

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Results saved to {output_json_path}")
        
        return results

    else:
        
        logger.info(f"Loading LLM Location data from {llm_data_path}...")
        
        with open(llm_data_path, 'r', encoding='utf-8') as f:
            llm_data = json.load(f)
        
        results = {
            "attractions": llm_data["attractions"],
            "hawkers": llm_data["hawkers"],
            "alns_weights": llm_data["alns_weights"]
        }
        
        return results

if __name__ == "__main__":
    result = process_and_save(
        persona="Backpacker",
        description="I am a student who wants to explore Singapore on a budget. I love local food and unique attractions.",
        attraction_path="./data/locationData/singapore_67_attractions_with_scores.csv", 
        hawker_path="./data/locationData/Food_20_withscores.xlsx", 
        output_json_path="./data/alns_inputs/groq/location_data.json", 
        batch_size=10
    )