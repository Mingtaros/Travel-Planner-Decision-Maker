import requests
import json
import os
import csv
import logging
import time
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bus_routes_fetch.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()  
API_KEY = os.getenv("DATAMALL_API_KEY")

# API configuration
BASE_URL = "https://datamall2.mytransport.sg/ltaodataservice"
ENDPOINT = "/BusRoutes"
OUTPUT_FILE = "../data/bus_routes.csv"
PROGRESS_FILE = "../log/fetch_progress.json"

def fetch_data(skip=0, max_retries=3, retry_delay=5):
    """
    Fetch data from the API endpoint with retry logic
    
    Args:
        skip (int): Number of records to skip
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Seconds to wait between retries
        
    Returns:
        dict: Parsed JSON response
    """
    
    headers = {
        "AccountKey": API_KEY,
        "Content-Type": "application/json"
    }
    
    # Set parameters according to API requirements
    params = {
        "$skip": skip
    }
    
    # Implement retry logic
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching data with $skip={skip} (Attempt {attempt+1}/{max_retries})")
            response = requests.get(
                f"{BASE_URL}{ENDPOINT}", 
                headers=headers,
                params=params,
                timeout=30  # Add timeout to prevent hanging
            )
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Parse and return the JSON response
            return response.json()
            
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            logger.error(f"Response: {response.text}")
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Error occurred: {req_err}")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON response")
        
        if attempt < max_retries - 1:
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    logger.error(f"Failed to fetch data after {max_retries} attempts")
    return None

def save_to_csv(data, filename, append=False):
    """
    Save the bus route data to a CSV file
    
    Args:
        data (list): List of bus route dictionaries
        filename (str): Name of the output CSV file
        append (bool): Whether to append to existing file
    """
    if not data:
        logger.warning("No data to save")
        return
    
    try:
        mode = 'a' if append else 'w'
        
        # Extract field names from the first item
        fieldnames = data[0].keys()
        
        # Write data to CSV file
        with open(filename, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header only if we're not appending
            if not append:
                writer.writeheader()
            
            # Write data rows
            writer.writerows(data)
            
        logger.info(f"Successfully saved {len(data)} records to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving data to CSV: {e}")

def save_progress(skip, total_fetched):
    """
    Save the current progress to a JSON file
    
    Args:
        skip (int): Current skip value
        total_fetched (int): Total number of records fetched so far
    """
    progress = {
        "last_skip": skip,
        "total_fetched": total_fetched,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f)
        logger.info(f"Progress saved: skip={skip}, total_fetched={total_fetched}")
    except Exception as e:
        logger.error(f"Error saving progress: {e}")

def load_progress():
    """
    Load the progress from a JSON file
    
    Returns:
        tuple: (skip, total_fetched) or (0, 0) if no progress file exists
    """
    try:
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                progress = json.load(f)
            skip = progress.get("last_skip", 0)
            total_fetched = progress.get("total_fetched", 0)
            logger.info(f"Loaded progress: skip={skip}, total_fetched={total_fetched}")
            
            # Confirm with user if they want to resume
            response = input(f"Found previous progress (skip={skip}, records={total_fetched}). Resume? (y/n): ")
            if response.lower() == 'y':
                return skip, total_fetched
    except Exception as e:
        logger.error(f"Error loading progress: {e}")
    
    return 0, 0

def fetch_all_data(start_skip=0, total_fetched=0):
    """
    Fetch all data by making multiple API calls with different skip values
    
    Args:
        start_skip (int): Skip value to start from
        total_fetched (int): Total records fetched so far
    
    Returns:
        list: All bus route records
    """
    all_data = []
    skip = start_skip
    batch_size = 500  # Fixed by the API
    
    # Check if CSV exists and we're resuming
    csv_exists = os.path.exists(OUTPUT_FILE) and start_skip > 0
    
    while True:
        logger.info(f"Fetching records with $skip={skip}...")
        response = fetch_data(skip)
        
        # Check if response has the expected structure
        if not response or 'value' not in response:
            logger.error("Invalid response format or empty response")
            break
            
        batch_data = response['value']
        
        # If no data returned, we've reached the end
        if not batch_data or len(batch_data) == 0:
            logger.info("No more records to fetch")
            break
            
        # Add batch data to our collected data
        all_data.extend(batch_data)
        total_fetched += len(batch_data)
        logger.info(f"Received {len(batch_data)} records (Total: {total_fetched})")
        
        # Save this batch to CSV (append if not the first batch or resuming)
        save_to_csv(batch_data, OUTPUT_FILE, csv_exists or skip > start_skip)
        
        # Save progress after each successful batch
        save_progress(skip + batch_size, total_fetched)
        
        # If we got fewer records than the batch size, we've reached the end
        if len(batch_data) < batch_size:
            break
            
        # Increment skip for the next batch
        skip += batch_size
    
    logger.info(f"Total records fetched: {total_fetched}")
    
    # Clear progress file if completed successfully
    if os.path.exists(PROGRESS_FILE) and len(all_data) > 0:
        try:
            os.remove(PROGRESS_FILE)
            logger.info("Cleared progress file after successful completion")
        except Exception as e:
            logger.error(f"Error removing progress file: {e}")
    
    return all_data

def main():
    # Check if we have a previous run to resume from
    start_skip, total_fetched = load_progress()
    
    # Fetch all data from the API and save to CSV
    all_data = fetch_all_data(start_skip, total_fetched)
    
    # Print sample record
    if all_data and len(all_data) > 0:
        logger.info("\nSample record:")
        logger.info(json.dumps(all_data[0], indent=2))
    
    logger.info("Data retrieval completed")

if __name__ == "__main__":
    main()