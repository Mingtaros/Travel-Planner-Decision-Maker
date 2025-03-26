#!/usr/bin/env python3
"""
Store waypoint data (location coordinates) in a JSON file

This script:
1. Reads attractions and hawker centers from Excel or CSV files
2. Geocodes them using the Google Maps API
3. Stores the coordinates in a JSON file
4. Provides functions to retrieve waypoints instead of re-geocoding

Run this script once to cache location coordinates, then modify
generate_route_matrix.py to use the JSON file instead of database.
"""

import os
import json
import logging
import argparse
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from google_maps_client import GoogleMapsClient

# Set up logging
os.makedirs("log", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/waypoints_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("waypoints")

# Load environment variables
load_dotenv()

def read_excel_or_csv(file_path):
    """Read data from Excel or CSV file based on extension"""
    try:
        _, file_extension = os.path.splitext(file_path)
        
        if file_extension.lower() in ['.xlsx', '.xls']:
            # Read Excel file
            df = pd.read_excel(file_path)
            logger.info(f"Successfully read Excel file: {file_path}")
        elif file_extension.lower() == '.csv':
            # Read CSV file
            df = pd.read_csv(file_path)
            logger.info(f"Successfully read CSV file: {file_path}")
        else:
            logger.error(f"Unsupported file format: {file_extension}")
            return None
        
        # Check if DataFrame is empty
        if df.empty:
            logger.error(f"File {file_path} is empty")
            return None
        
        return df
    
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None

def load_attractions(file_path):
    """Load attractions from Excel or CSV file"""
    df = read_excel_or_csv(file_path)
    if df is None:
        return []
    
    try:
        # Check for required columns
        required_columns = ['No.', 'Attraction Name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns in attractions file: {', '.join(missing_columns)}")
            return []
        
        # Convert DataFrame to list of dictionaries
        attractions = []
        for idx, row in df.iterrows():
            attraction = {
                'id': f"A{idx+1}",
                'name': row['Attraction Name'],
                'type': 'attraction',
            }
            
            attractions.append(attraction)
        
        logger.info(f"Loaded {len(attractions)} attractions from {file_path}")
        return attractions
    
    except Exception as e:
        logger.error(f"Error processing attractions file: {e}")
        return []

def load_hawkers(file_path):
    """Load hawker centers from Excel or CSV file"""
    df = read_excel_or_csv(file_path)
    if df is None:
        return []
    
    try:
        # Check for required columns
        required_columns = ['No.', 'Name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns in hawkers file: {', '.join(missing_columns)}")
            return []
        
        # Convert DataFrame to list of dictionaries
        hawkers = []
        for idx, row in df.iterrows():
            hawker = {
                'id': f"H{idx+1}",
                'name': row['Name'],
                'type': 'hawker',
            }
            
            hawkers.append(hawker)
        
        logger.info(f"Loaded {len(hawkers)} hawker centers from {file_path}")
        return hawkers
    
    except Exception as e:
        logger.error(f"Error processing hawkers file: {e}")
        return []

def geocode_locations(maps_client, locations):
    """Geocode locations to get their coordinates"""
    geocoded_locations = []
    
    for location in locations:
        try:
            # Prepare search query
            if location.get('address'):
                query = f"{location['name']}, {location['address']}"
            else:
                query = f"{location['name']}, Singapore"
            
            logger.info(f"Geocoding {location['name']}...")
            
            # Get geocode
            place_details = maps_client.get_place_details(place_name=query)
            place_data = maps_client.parse_place_details(place_details)
            
            if place_data and 'location' in place_data:
                geocoded_location = location.copy()
                geocoded_location['lat'] = place_data['location']['lat']
                geocoded_location['lng'] = place_data['location']['lng']
                
                # Add formatted address if available
                if 'address' in place_data:
                    geocoded_location['address'] = place_data['address']
                
                geocoded_locations.append(geocoded_location)
                
                logger.info(f"Successfully geocoded {location['name']}")
            else:
                logger.warning(f"Could not geocode {location['name']}")
        
        except Exception as e:
            logger.error(f"Error geocoding {location['name']}: {e}")
    
    logger.info(f"Successfully geocoded {len(geocoded_locations)} out of {len(locations)} locations")
    return geocoded_locations

def save_locations_to_json(locations, output_file):
    """Save geocoded locations to a JSON file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Prepare data structure for waypoints
        waypoints_data = {
            "metadata": {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_locations": len(locations)
            },
            "locations": locations
        }
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(waypoints_data, f, indent=2)
        
        logger.info(f"Saved {len(locations)} waypoints to {output_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving waypoints to file: {e}")
        return False

def load_waypoints_from_json(file_path):
    """Load waypoints from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract locations
        locations = data.get('locations', [])
        
        # Create waypoints list (expected by Google Maps API)
        waypoints = []
        for loc in locations:
            # Only add if it has coordinates
            if 'lat' in loc and 'lng' in loc:
                waypoint = [
                    loc['name'],
                    loc['lat'],
                    loc['lng']
                ]
                waypoints.append(waypoint)
        
        logger.info(f"Loaded {len(waypoints)} waypoints from {file_path}")
        return locations, waypoints
    
    except Exception as e:
        logger.error(f"Error loading waypoints from {file_path}: {e}")
        return [], []

def process_waypoints(attractions_file, hawkers_file, output_file, force_geocode=False):
    """
    Process waypoints from Excel/CSV files and geocode them
    
    Args:
        attractions_file: Path to attractions Excel/CSV file
        hawkers_file: Path to hawkers Excel/CSV file
        output_file: Path to output JSON file
        force_geocode: Whether to force geocoding even if output file exists
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Starting waypoints processing...")
    
    # Check if output file already exists and we're not forcing geocoding
    if os.path.exists(output_file) and not force_geocode:
        logger.info(f"Output file {output_file} already exists. Loading from file...")
        locations, waypoints = load_waypoints_from_json(output_file)
        if locations and waypoints:
            logger.info(f"Successfully loaded {len(locations)} locations from existing file")
            return True
        else:
            logger.warning("Failed to load from existing file, proceeding with geocoding")
    
    try:
        # Load attractions and hawkers
        attractions = load_attractions(attractions_file)
        hawkers = load_hawkers(hawkers_file)
        
        # Combine locations
        all_locations = attractions + hawkers
        
        if not all_locations:
            logger.error("No locations loaded from input files")
            return False
        
        # Initialize Google Maps client
        maps_client = GoogleMapsClient()
        logger.info("Google Maps client initialized")
        
        # Geocode locations
        geocoded_locations = geocode_locations(maps_client, all_locations)
        
        if not geocoded_locations:
            logger.error("No locations could be geocoded")
            return False
        
        # Save geocoded locations to JSON
        success = save_locations_to_json(geocoded_locations, output_file)
        
        return success
    
    except Exception as e:
        logger.error(f"Error processing waypoints: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to process command line arguments"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process waypoints from Excel/CSV files and geocode them')
    parser.add_argument('--attractions', required=True, help='Path to attractions Excel/CSV file')
    parser.add_argument('--hawkers', required=True, help='Path to hawkers Excel/CSV file')
    parser.add_argument('--output', default='data/waypointData/waypoints.json', help='Path to output JSON file')
    parser.add_argument('--force', action='store_true', help='Force geocoding even if output file exists')
    
    args = parser.parse_args()
    
    if process_waypoints(args.attractions, args.hawkers, args.output, args.force):
        logger.info("Successfully processed waypoints")
        
        # Test loading the waypoints
        locations, waypoints = load_waypoints_from_json(args.output)
        
        if locations and waypoints:
            logger.info(f"Successfully loaded {len(locations)} waypoints from output file")
            
            # Print a sample waypoint
            sample = locations[0]
            logger.info(f"Sample waypoint: {sample['name']} ({sample['id']})")
            logger.info(f"  Coordinates: {sample['lat']}, {sample['lng']}")
            logger.info(f"  Type: {sample['type']}")
            if 'address' in sample:
                logger.info(f"  Address: {sample['address']}")
        else:
            logger.error("Failed to load waypoints from output file")
    else:
        logger.error("Failed to process waypoints")

if __name__ == "__main__":
    main()