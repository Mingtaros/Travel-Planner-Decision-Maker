#!/usr/bin/env python3
"""
Generate a comprehensive route matrix for Singapore attractions and food centers
using cached waypoints from the database

This script:
1. Fetches waypoints directly from the database instead of geocoding each time
2. Processes waypoints in batches to handle the 100-element limitation of the Google Maps API
3. Computes route matrices for both transit and driving modes
4. Calculates fares based on transit data or driving distance formula
5. Combines all information into a single comprehensive matrix
6. Stores the results in both JSON files and a database table
7. Supports multiple departure dates/times for more accurate transit planning
"""

import os
import json
import logging
import math
import time
import mysql.connector
from datetime import datetime, timedelta
from dotenv import load_dotenv
from utils.google_maps_client import GoogleMapsClient
from store_waypoints import fetch_waypoints

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

db_user = os.getenv("MYSQL_USER")
db_port = os.getenv("MYSQL_PORT")
db_pwd = os.getenv("MYSQL_PASSWORD")
db_database = os.getenv("MYSQL_DATABASE")
db_host = os.getenv("MYSQL_HOST")
# Database connection parameters
DB_CONFIG = {
    'host': db_host,
    'port': db_port,
    'user': db_user,
    'password': db_pwd,
    'database': db_database
}

# Define departure dates/times for route calculations
# Format: (description, datetime_object)
DEPARTURE_TIMES = [
    ("Morning", datetime(2025, 5, 17, 8, 0, 0)), 
    ("Midday", datetime(2025, 5, 17, 12, 0, 0)), 
    ("Evening", datetime(2025, 5, 17, 16, 0, 0)), 
    ("Night", datetime(2025, 5, 17, 20, 0, 0)),  
]

# API limitations
MAX_ELEMENTS_PER_REQUEST = 100  # Google Maps API limit (origins × destinations)
BATCH_SIZE = 10  # Number of origins per batch (10×10=100 elements)
API_RATE_LIMIT_DELAY = 2  # Delay between API calls (seconds)

waypoint_limit = None

def connect_to_database():
    """Connect to the MySQL database"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        logger.info("Connected to MySQL database")
        return connection
    except mysql.connector.Error as e:
        logger.error(f"Error connecting to MySQL database: {e}")
        return None

def fetch_transit_fare_data(connection):
    """Fetch transit fare data from the database"""
    cursor = connection.cursor(dictionary=True)
    
    # Try to fetch from transit_fare table if it exists
    try:
        cursor.execute("SELECT * FROM transit_fare")
        transit_fares = cursor.fetchall()
        if transit_fares:
            logger.info(f"Fetched {len(transit_fares)} transit fare records from transit_fare table")
            return transit_fares
    except mysql.connector.Error as e:
        logger.warning(f"Error fetching transit fare data: {e}")
        return []

def get_transit_fare(distance_km, fare_table):
    """Get transit fare based on distance from the fare table"""
    for fare_info in fare_table:
        if fare_info['lower_distance'] <= round(distance_km, 1) <= fare_info['upper_distance']:
            return fare_info['basic_fare']
    
    # Default fare for very long distances
    return None

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
        destinations=dest_waypoints,  # Enable separate destinations parameter
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
        "locations": {
            location['id']: {
                "id": location['id'],
                "name": location['name'],
                "type": location['type'],
                "lat": location['lat'],
                "lng": location['lng']
            }
            for location in locations
        },
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
        with open(file_path, 'w') as f:
            json.dump(matrix, f, indent=2)
        
        logger.info(f"Matrix saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving matrix to file: {e}")
        return False

def create_route_matrix_table(connection):
    """Create the route_matrix table in the database if it doesn't exist"""
    try:
        cursor = connection.cursor()
        
        # Create table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS route_matrix (
            id INT AUTO_INCREMENT PRIMARY KEY,
            origin_id VARCHAR(50) NOT NULL,
            destination_id VARCHAR(50) NOT NULL,
            origin_name VARCHAR(255) NOT NULL,
            destination_name VARCHAR(255) NOT NULL,
            departure_time DATETIME NOT NULL,
            time_description VARCHAR(50) NOT NULL,
            transit_distance_km FLOAT NOT NULL,
            transit_duration_minutes FLOAT NOT NULL,
            transit_fare_sgd DECIMAL(10, 2) NOT NULL,
            driving_distance_km FLOAT NOT NULL,
            driving_duration_minutes FLOAT NOT NULL,
            driving_fare_sgd DECIMAL(10, 2) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY unique_route (origin_id, destination_id, departure_time)
        )
        """
        cursor.execute(create_table_query)
        connection.commit()
        cursor.close()
        
        logger.info("route_matrix table created or already exists")
        return True
    except mysql.connector.Error as e:
        logger.error(f"Error creating route_matrix table: {e}")
        return False

def save_matrix_to_database(connection, matrix):
    """Save the comprehensive matrix to the database"""
    try:
        # Create the table if it doesn't exist
        if not create_route_matrix_table(connection):
            return False
        
        # Prepare data for insertion
        routes_data = []
        departure_time = datetime.strptime(
            matrix['metadata']['departure_time'], 
            "%Y-%m-%d %H:%M:%S"
        )
        time_description = matrix['metadata']['time_description']
        
        for route_id, route in matrix['routes'].items():
            # Prepare row data
            route_data = (
                route['origin_id'],
                route['destination_id'],
                route['origin_name'],
                route['destination_name'],
                departure_time,
                time_description,
                route['transit']['distance_km'],
                route['transit']['duration_minutes'],
                route['transit']['fare_sgd'],
                route['drive']['distance_km'],
                route['drive']['duration_minutes'],
                route['drive']['fare_sgd']
            )
            routes_data.append(route_data)
        
        # Insert data
        cursor = connection.cursor()
        
        # Use REPLACE INTO to update existing routes or insert new ones
        insert_query = """
        REPLACE INTO route_matrix (
            origin_id, destination_id, origin_name, destination_name,
            departure_time, time_description,
            transit_distance_km, transit_duration_minutes, transit_fare_sgd,
            driving_distance_km, driving_duration_minutes, driving_fare_sgd
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.executemany(insert_query, routes_data)
        connection.commit()
        
        logger.info(f"Saved {len(routes_data)} routes to database")
        return True
    except mysql.connector.Error as e:
        logger.error(f"Error saving matrix to database: {e}")
        return False

def generate_comprehensive_matrix(departure_time, time_description, output_file=None):
    """
    Generate a comprehensive route matrix with transit and driving information
    for attractions and hawker centers
    """
    logger.info(f"Starting comprehensive route matrix generation for {time_description}...")
    
    # Connect to database
    connection = connect_to_database()
    if not connection:
        return None
    
    try:
        # Initialize Google Maps client
        maps_client = GoogleMapsClient()
        logger.info("Google Maps client initialized")
        
        # Fetch waypoints from database
        locations, waypoints = fetch_waypoints(connection, waypoint_limit)
        if not locations or not waypoints:
            logger.error("No waypoints found in database")
            return None
        
        # Fetch transit fare data
        fare_table = fetch_transit_fare_data(connection)
        
        # Compute route matrices in batches
        matrices = compute_route_matrices(maps_client, waypoints, departure_time)
        if not matrices:
            logger.error("Failed to compute route matrices")
            return None
        
        # Process route matrices into a comprehensive matrix
        comprehensive_matrix = process_route_matrices(
            matrices, locations, waypoints, fare_table, 
            departure_time, time_description
        )
        
        # Save the comprehensive matrix to file if output file is specified
        if output_file:
            save_matrix_to_file(comprehensive_matrix, output_file)
        
        # Save the comprehensive matrix to database
        save_matrix_to_database(connection, comprehensive_matrix)
        
        return comprehensive_matrix
        
    except Exception as e:
        logger.error(f"Error generating comprehensive matrix: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    finally:
        # Close database connection
        if connection and connection.is_connected():
            connection.close()
            logger.info("Database connection closed")

def main():
    """Main function to demonstrate the script"""
    # Create output directory
    os.makedirs("data/routeData", exist_ok=True)
    
    # Generate matrices for each departure time
    for time_description, departure_time in DEPARTURE_TIMES:
        # Define output file path
        output_file = f"data/routeData/route_matrix_{time_description.lower().replace(' ', '_')}.json"
        
        # Generate comprehensive matrix
        comprehensive_matrix = generate_comprehensive_matrix(
            departure_time=departure_time,
            time_description=time_description,
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