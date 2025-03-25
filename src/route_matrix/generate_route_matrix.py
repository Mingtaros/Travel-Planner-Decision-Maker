#!/usr/bin/env python3
"""
Generate a comprehensive route matrix for Singapore attractions and food centers
using waypoints from a JSON file

This script:
1. Loads waypoints from a JSON file instead of a database
2. Processes waypoints in batches to handle the 100-element limitation of the Google Maps API
3. Computes route matrices for both transit and driving modes
4. Calculates fares based on transit data or driving distance formula
5. Combines all information into a single comprehensive matrix
6. Stores the results in JSON files for different time periods
7. Supports multiple departure dates/times for more accurate transit planning
"""

import os
import json
import logging
import math
import time
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google_maps_client import GoogleMapsClient

# Set up logging
os.makedirs("log", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/route_matrix_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("route_matrix")

# Load environment variables
load_dotenv()

# Define departure dates/times for route calculations
# Format: (description, datetime_object)
DEPARTURE_TIMES = [
    ("Morning", datetime(2025, 5, 17, 8, 0, 0)), 
    ("Midday", datetime(2025, 5, 17, 12, 0, 0)), 
    ("Evening", datetime(2025, 5, 17, 16, 0, 0)), 
    ("Night", datetime(2025, 5, 17, 20, 0, 0)),  
]

# Transit fare table for Singapore (distance in km, fare in SGD)
DEFAULT_TRANSIT_FARE_TABLE = [
    {"lower_distance": 0.0, "upper_distance": 3.2, "basic_fare": 1.19},
    {"lower_distance": 3.2, "upper_distance": 4.2, "basic_fare": 1.29},
    {"lower_distance": 4.2, "upper_distance": 5.2, "basic_fare": 1.40},
    {"lower_distance": 5.2, "upper_distance": 6.2, "basic_fare": 1.50},
    {"lower_distance": 6.2, "upper_distance": 7.2, "basic_fare": 1.59},
    {"lower_distance": 7.2, "upper_distance": 8.2, "basic_fare": 1.66},
    {"lower_distance": 8.2, "upper_distance": 9.2, "basic_fare": 1.73},
    {"lower_distance": 9.2, "upper_distance": 10.2, "basic_fare": 1.77},
    {"lower_distance": 10.2, "upper_distance": 11.2, "basic_fare": 1.81},
    {"lower_distance": 11.2, "upper_distance": 12.2, "basic_fare": 1.85},
    {"lower_distance": 12.2, "upper_distance": 13.2, "basic_fare": 1.89},
    {"lower_distance": 13.2, "upper_distance": 14.2, "basic_fare": 1.93},
    {"lower_distance": 14.2, "upper_distance": 15.2, "basic_fare": 1.98},
    {"lower_distance": 15.2, "upper_distance": 16.2, "basic_fare": 2.02},
    {"lower_distance": 16.2, "upper_distance": 17.2, "basic_fare": 2.06},
    {"lower_distance": 17.2, "upper_distance": 18.2, "basic_fare": 2.10},
    {"lower_distance": 18.2, "upper_distance": 19.2, "basic_fare": 2.14},
    {"lower_distance": 19.2, "upper_distance": 20.2, "basic_fare": 2.17},
    {"lower_distance": 20.2, "upper_distance": 21.2, "basic_fare": 2.20},
    {"lower_distance": 21.2, "upper_distance": 22.2, "basic_fare": 2.23},
    {"lower_distance": 22.2, "upper_distance": 23.2, "basic_fare": 2.26},
    {"lower_distance": 23.2, "upper_distance": 24.2, "basic_fare": 2.28},
    {"lower_distance": 24.2, "upper_distance": 25.2, "basic_fare": 2.30},
    {"lower_distance": 25.2, "upper_distance": 26.2, "basic_fare": 2.32},
    {"lower_distance": 26.2, "upper_distance": 27.2, "basic_fare": 2.33},
    {"lower_distance": 27.2, "upper_distance": 28.2, "basic_fare": 2.34},
    {"lower_distance": 28.2, "upper_distance": 29.2, "basic_fare": 2.35},
    {"lower_distance": 29.2, "upper_distance": 30.2, "basic_fare": 2.36},
    {"lower_distance": 30.2, "upper_distance": 31.2, "basic_fare": 2.37},
    {"lower_distance": 31.2, "upper_distance": 32.2, "basic_fare": 2.38},
    {"lower_distance": 32.2, "upper_distance": 33.2, "basic_fare": 2.39},
    {"lower_distance": 33.2, "upper_distance": 34.2, "basic_fare": 2.40},
    {"lower_distance": 34.2, "upper_distance": 35.2, "basic_fare": 2.41},
    {"lower_distance": 35.2, "upper_distance": 36.2, "basic_fare": 2.42},
    {"lower_distance": 36.2, "upper_distance": 37.2, "basic_fare": 2.43},
    {"lower_distance": 37.2, "upper_distance": 38.2, "basic_fare": 2.44},
    {"lower_distance": 38.2, "upper_distance": 39.2, "basic_fare": 2.45},
    {"lower_distance": 39.2, "upper_distance": 40.2, "basic_fare": 2.46},
    {"lower_distance": 40.2, "upper_distance": float('inf'), "basic_fare": 2.47}
]

# API limitations
MAX_ELEMENTS_PER_REQUEST = 100  # Google Maps API limit (origins × destinations)
BATCH_SIZE = 10  # Number of origins per batch (10×10=100 elements)
API_RATE_LIMIT_DELAY = 2  # Delay between API calls (seconds)

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

def get_transit_fare(distance_km, fare_table=None):
    """Get transit fare based on distance from the fare table"""
    # Use default fare table if none provided
    if fare_table is None:
        fare_table = DEFAULT_TRANSIT_FARE_TABLE
        
    for fare_info in fare_table:
        if fare_info['lower_distance'] <= distance_km <= fare_info['upper_distance']:
            return fare_info['basic_fare']
    
    # Default fare for very long distances
    return 2.47  # Maximum fare

def calculate_car_fare(distance_m, flag_down=4.8):
    """Calculate Singapore grab/taxi fare based on distance in meters"""
    
    # Start with flag-down fare (covers first 1km)
    fare = flag_down
    
    # Calculate distance charge
    if distance_m > 1000:
        remaining_m = distance_m - 1000  # First 1 km is covered in flag-down

        if remaining_m <= 9000:
            # Charge $0.26 per 400m up to 10km total
            fare += (remaining_m // 400) * 0.26
            # Add partial unit if there's a remainder
            if remaining_m % 400 > 0:
                fare += 0.26
        else:
            # First 9km after flag-down
            fare += (9000 // 400) * 0.26
            
            # Calculate remaining distance after 10km total
            ultra_remaining = remaining_m - 9000
            
            # Beyond 10km, charge $0.26 per 350m
            fare += (ultra_remaining // 350) * 0.26
            # Add partial unit if there's a remainder
            if ultra_remaining % 350 > 0:
                fare += 0.26
    
    return round(fare, 2)

def compute_route_matrix_batch(maps_client, origin_waypoints, dest_waypoints, mode, departure_time):
    """
    Compute route matrix for a single batch of origins and destinations
    
    Args:
        maps_client: GoogleMapsClient instance
        origin_waypoints: List of origin waypoints for this batch
        dest_waypoints: List of destination waypoints for this batch
        mode: "transit" or "drive"
        departure_time: Departure datetime
        
    Returns:
        List of route elements from the API
    """
    # Log the batch size
    logger.info(f"Computing {mode} route matrix batch: {len(origin_waypoints)} origins × {len(dest_waypoints)} destinations")
    
    # Make the API request
    response = maps_client.compute_route_matrix(
        origins=origin_waypoints,
        destinations=dest_waypoints,
        mode=mode,
        departure_time=departure_time
    )
    
    if response:
        logger.info(f"Successfully computed {mode} matrix batch with {len(response)} routes")
        return response
    else:
        logger.error(f"Failed to compute {mode} route matrix batch")
        return []

def compute_route_matrices(maps_client, all_waypoints, departure_time):
    """
    Compute route matrices for both transit and driving modes in batches
    to handle the 100-element limitation
    """
    transit_matrix = []
    driving_matrix = []
    
    # Calculate how many batches we need
    num_locations = len(all_waypoints)
    num_batches = math.ceil(num_locations / BATCH_SIZE)
    
    logger.info(f"Processing {num_locations} locations in {num_batches}×{num_batches} batches")
    
    # Process in batches
    for i in range(num_batches):
        # Get origin batch
        origin_start = i * BATCH_SIZE
        origin_end = min((i + 1) * BATCH_SIZE, num_locations)
        origin_batch = all_waypoints[origin_start:origin_end]
        
        # For each origin batch, we need to process all destination batches
        for j in range(num_batches):
            # Get destination batch
            dest_start = j * BATCH_SIZE
            dest_end = min((j + 1) * BATCH_SIZE, num_locations)
            dest_batch = all_waypoints[dest_start:dest_end]
            
            # Calculate offset indices for this batch
            origin_offset = origin_start
            dest_offset = dest_start
            
            logger.info(f"Processing batch for origins {origin_start}-{origin_end-1}, destinations {dest_start}-{dest_end-1}")
            
            # Compute transit matrix for this batch
            logger.info(f"Computing transit route matrix batch for {departure_time.strftime('%Y-%m-%d %H:%M')}...")
            batch_transit = compute_route_matrix_batch(
                maps_client, origin_batch, dest_batch, "transit", departure_time
            )
            
            # Add origin/destination indices to make global indices
            for route in batch_transit:
                # Adjust the indices to be global instead of batch-local
                route['originIndex'] = route['originIndex'] + origin_offset
                route['destinationIndex'] = route['destinationIndex'] + dest_offset
            
            # Add to complete transit matrix
            transit_matrix.extend(batch_transit)
            
            # Sleep to avoid hitting rate limits
            time.sleep(API_RATE_LIMIT_DELAY)
            
            # Compute driving matrix for this batch
            logger.info(f"Computing driving route matrix batch for {departure_time.strftime('%Y-%m-%d %H:%M')}...")
            batch_driving = compute_route_matrix_batch(
                maps_client, origin_batch, dest_batch, "drive", departure_time
            )
            
            # Add origin/destination indices to make global indices
            for route in batch_driving:
                # Adjust the indices to be global instead of batch-local
                route['originIndex'] = route['originIndex'] + origin_offset
                route['destinationIndex'] = route['destinationIndex'] + dest_offset
            
            # Add to complete driving matrix
            driving_matrix.extend(batch_driving)
            
            # Sleep to avoid hitting rate limits
            time.sleep(API_RATE_LIMIT_DELAY)
    
    # Return the complete matrices
    matrices = {}
    if transit_matrix:
        matrices['transit'] = transit_matrix
        logger.info(f"Successfully computed complete transit matrix with {len(transit_matrix)} routes")
    
    if driving_matrix:
        matrices['drive'] = driving_matrix
        logger.info(f"Successfully computed complete driving matrix with {len(driving_matrix)} routes")
    
    return matrices

def process_route_matrices(matrices, locations, waypoints, fare_table, departure_time, time_description):
    """
    Process route matrices to create a comprehensive route matrix
    with both transit and driving information
    """
    # Extract waypoint names for reference
    waypoint_names = [wp[0] for wp in waypoints]
    
    # Create the comprehensive matrix structure
    comprehensive_matrix = {
        "metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "departure_time": departure_time.strftime("%Y-%m-%d %H:%M:%S"),
            "time_description": time_description,
            "num_locations": len(waypoints),
            "num_routes": 0
        },
        "locations": {location['id']: location for location in locations},
        "routes": {}
    }
    
    # Check if we have both matrices
    if 'transit' not in matrices or 'drive' not in matrices:
        logger.error("Missing transit or driving matrix data")
        return comprehensive_matrix
    
    # Process all possible routes between locations
    for i, origin in enumerate(locations):
        for j, destination in enumerate(locations):
            # Skip self-routes
            if i == j:
                continue
                
            # Create route ID
            route_id = f"{origin['id']}_to_{destination['id']}"
            
            # Initialize route data
            route_data = {
                "origin_id": origin['id'],
                "destination_id": destination['id'],
                "origin_name": origin['name'],
                "destination_name": destination['name'],
                "departure_time": departure_time.strftime("%Y-%m-%d %H:%M:%S"),
                "time_description": time_description,
                "transit": {
                    "distance_km": 0,
                    "duration_minutes": 0,
                    "fare_sgd": 0
                },
                "drive": {
                    "distance_km": 0,
                    "duration_minutes": 0,
                    "fare_sgd": 0
                }
            }
            
            # Find and process transit route
            transit_route = None
            for route in matrices['transit']:
                if (route.get('originIndex') == i and 
                    route.get('destinationIndex') == j and
                    route.get('condition') == 'ROUTE_EXISTS'):
                    transit_route = route
                    break
            
            if transit_route:
                # Extract distance and duration
                distance_meters = transit_route.get('distanceMeters', 0)
                distance_km = distance_meters / 1000
                
                duration_seconds = 0
                if 'duration' in transit_route:
                    duration_str = transit_route['duration']
                    if duration_str.endswith('s'):
                        duration_seconds = int(duration_str[:-1])
                
                # Calculate fare based on distance
                fare = get_transit_fare(distance_km, fare_table)
                
                # Update transit route data
                route_data['transit'] = {
                    "distance_km": round(distance_km, 2),
                    "duration_minutes": round(duration_seconds / 60, 2),
                    "fare_sgd": round(fare, 2)
                }
            
            # Find and process driving route
            driving_route = None
            for route in matrices['drive']:
                if (route.get('originIndex') == i and 
                    route.get('destinationIndex') == j and
                    route.get('condition') == 'ROUTE_EXISTS'):
                    driving_route = route
                    break
            
            if driving_route:
                # Extract distance and duration
                distance_meters = driving_route.get('distanceMeters', 0)
                distance_km = distance_meters / 1000
                
                duration_seconds = 0
                if 'duration' in driving_route:
                    duration_str = driving_route['duration']
                    if duration_str.endswith('s'):
                        duration_seconds = int(duration_str[:-1])
                
                # Calculate driving fare
                fare = calculate_car_fare(distance_meters)
                
                # Update driving route data
                route_data['drive'] = {
                    "distance_km": round(distance_km, 2),
                    "duration_minutes": round(duration_seconds / 60, 2),
                    "fare_sgd": fare
                }
            
            # Add route to comprehensive matrix
            comprehensive_matrix['routes'][route_id] = route_data
            comprehensive_matrix['metadata']['num_routes'] += 1
    
    return comprehensive_matrix

def save_matrix_to_file(matrix, file_path):
    """Save the comprehensive matrix to a JSON file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(matrix, f, indent=2)
        
        logger.info(f"Matrix saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving matrix to file: {e}")
        return False

def generate_comprehensive_matrix(departure_time, time_description, waypoints_file, output_file=None):
    """
    Generate a comprehensive route matrix with transit and driving information
    for attractions and hawker centers
    
    Args:
        departure_time: Departure datetime
        time_description: Description of the time period
        waypoints_file: Path to the waypoints JSON file
        output_file: Path to output JSON file
    
    Returns:
        dict or None: Comprehensive matrix if successful, None otherwise
    """
    logger.info(f"Starting comprehensive route matrix generation for {time_description}...")
    
    try:
        # Initialize Google Maps client
        maps_client = GoogleMapsClient()
        logger.info("Google Maps client initialized")
        
        # Load waypoints from JSON file
        locations, waypoints = load_waypoints_from_json(waypoints_file)
        if not locations or not waypoints:
            logger.error(f"No waypoints found in file {waypoints_file}")
            return None
        
        # Compute route matrices in batches
        matrices = compute_route_matrices(maps_client, waypoints, departure_time)
        if not matrices:
            logger.error("Failed to compute route matrices")
            return None
        
        # Process route matrices into a comprehensive matrix
        comprehensive_matrix = process_route_matrices(
            matrices, locations, waypoints, DEFAULT_TRANSIT_FARE_TABLE, 
            departure_time, time_description
        )
        
        # Save the comprehensive matrix to file if output file is specified
        if output_file:
            save_matrix_to_file(comprehensive_matrix, output_file)
        
        return comprehensive_matrix
        
    except Exception as e:
        logger.error(f"Error generating comprehensive matrix: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    """Main function to process command line arguments"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate route matrices for different time periods')
    parser.add_argument('--waypoints', default='data/waypointData/waypoints.json', help='Path to waypoints JSON file')
    parser.add_argument('--output-dir', default='data/routeData', help='Directory to save route matrices')
    parser.add_argument('--time-periods', choices=['all', 'morning', 'midday', 'evening', 'night'], 
                      default='all', help='Time periods to generate matrices for')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if waypoints file exists
    if not os.path.exists(args.waypoints):
        logger.error(f"Waypoints file not found: {args.waypoints}")
        return
    
    # Filter departure times based on selected time periods
    selected_times = DEPARTURE_TIMES
    if args.time_periods != 'all':
        selected_times = [dt for dt in DEPARTURE_TIMES if dt[0].lower() == args.time_periods.lower()]
    
    # Generate matrices for each departure time
    for time_description, departure_time in selected_times:
        # Define output file path
        output_file = os.path.join(args.output_dir, f"route_matrix_{time_description.lower()}.json")
        
        # Generate comprehensive matrix
        comprehensive_matrix = generate_comprehensive_matrix(
            departure_time=departure_time,
            time_description=time_description,
            waypoints_file=args.waypoints,
            output_file=output_file
        )
        
        if comprehensive_matrix:
            # Print a summary
            num_locations = len(comprehensive_matrix['locations'])
            num_routes = comprehensive_matrix['metadata']['num_routes']
            logger.info(f"Generated comprehensive matrix for {time_description} with {num_locations} locations and {num_routes} routes")
            
            # Print a sample route
            if comprehensive_matrix['routes']:
                sample_route = next(iter(comprehensive_matrix['routes'].values()))
                logger.info(f"Sample route for {time_description}:")
                logger.info(f"  From: {sample_route['origin_name']} ({sample_route['origin_id']})")
                logger.info(f"  To: {sample_route['destination_name']} ({sample_route['destination_id']})")
                logger.info(f"  Transit: {sample_route['transit']['distance_km']:.2f} km, " +
                           f"{sample_route['transit']['duration_minutes']:.0f} min, " +
                           f"${sample_route['transit']['fare_sgd']:.2f}")
                logger.info(f"  Driving: {sample_route['drive']['distance_km']:.2f} km, " + 
                           f"{sample_route['drive']['duration_minutes']:.0f} min, " +
                           f"${sample_route['drive']['fare_sgd']:.2f}")
        else:
            logger.error(f"Failed to generate comprehensive matrix for {time_description}")

if __name__ == "__main__":
    main()