"""
Generate a matrix of routes between attractions and hawker centers in Singapore

This script:
1. Connects to the MySQL database to fetch attractions and hawker centers
2. Uses Google Maps API to calculate routes between each pair for both transit and driving
3. Saves the route matrix as JSON for use in the optimization algorithm
"""

import os
import json
import time
import logging
import traceback
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from google_maps.google_maps_client import GoogleMapsClient
from get_trip_detail import get_trip_details

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/route_matrix_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("route_matrix")

# Load environment variables
load_dotenv()

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'planner',
    'password': 'plannerpassword',
    'database': 'ai_planning_project'
}

def connect_to_database():
    """Connect to the MySQL database"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        logger.info("Connected to MySQL database")
        return connection
    except mysql.connector.Error as e:
        logger.error(f"Error connecting to MySQL database: {e}")
        return None

def fetch_transit_fare(connection, limit=None):
    """Fetch transit_fare from the database"""
    cursor = connection.cursor(dictionary=True)
    query = "SELECT * FROM attractions"
    if limit:
        query += f" LIMIT {limit}"
    logger.debug(f"Executing query: {query}")
    cursor.execute(query)
    attractions = cursor.fetchall()
    cursor.close()
    logger.info(f"Fetched {len(attractions)} attractions")

def fetch_attractions(connection, limit=None):
    """Fetch all attractions from the database"""
    cursor = connection.cursor(dictionary=True)
    query = "SELECT * FROM attractions"
    if limit:
        query += f" LIMIT {limit}"
    logger.debug(f"Executing query: {query}")
    cursor.execute(query)
    attractions = cursor.fetchall()
    cursor.close()
    logger.info(f"Fetched {len(attractions)} attractions")
    
    # # Log the first attraction to verify structure
    # if attractions:
    #     logger.debug(f"Sample attraction: {attractions[0]}")
    
    return attractions

def fetch_hawker_centers(connection, limit=None):
    """Fetch all hawker centers from the database"""
    cursor = connection.cursor(dictionary=True)
    query = "SELECT * FROM foodcentre"
    if limit:
        query += f" LIMIT {limit}"
    logger.debug(f"Executing query: {query}")
    cursor.execute(query)
    hawker_centers = cursor.fetchall()
    cursor.close()
    logger.info(f"Fetched {len(hawker_centers)} hawker centers")
    
    # # Log the first hawker center to verify structure
    # if hawker_centers:
    #     logger.debug(f"Sample hawker center: {hawker_centers[0]}")
    
    return hawker_centers

def generate_location_pairs(attractions, hawker_centers):
    """Generate all pairs of locations that need routes"""
    locations = []
    
    # Add all attractions with their information
    for attraction in attractions:
        try:
            location = {
                'id': f"A{attraction['aid']}",
                'name': attraction['aname'],
                'type': 'attraction',
                'expenditure': attraction.get('expenditure', 0),
                'timespent': attraction.get('timespent', 0)
            }
            locations.append(location)
        except KeyError as e:
            logger.error(f"Missing key in attraction: {e}")
            logger.debug(f"Problematic attraction data: {attraction}")
    
    # Add all hawker centers with their information
    for hawker in hawker_centers:
        try:
            location = {
                'id': f"H{hawker['fid']}",
                'name': hawker['name'],
                'type': 'hawker',
                'expenditure': hawker.get('expenditure', 0),
                'timespent': hawker.get('timespent', 0),
                'rating': hawker.get('rating', 0),
                'food_type': hawker.get('type', ''),
                'best_for': hawker.get('bestfor', ''),
                'address': hawker.get('address', '')
            }
            locations.append(location)
        except KeyError as e:
            logger.error(f"Missing key in hawker center: {e}")
            logger.debug(f"Problematic hawker center data: {hawker}")
    
    # Generate all pairs (we need routes from every location to every other)
    pairs = []
    for i, origin in enumerate(locations):
        for destination in locations[i+1:]:  # Avoid self-loops and duplicates
            pairs.append((origin, destination))
    
    logger.info(f"Generated {len(pairs)} location pairs to route")
    if pairs:
        logger.debug(f"First pair: {pairs[0][0]['name']} to {pairs[0][1]['name']}")
    
    return locations, pairs

def get_transit_route_for_pair(maps_client, origin, destination, departure_time=None):
    """Get transit route information for a pair of locations"""
    try:
        # Prepare location names
        origin_name = f"{origin['name']}, Singapore"
        destination_name = f"{destination['name']}, Singapore"
        
        # Use address if available (for hawker centers)
        if origin['type'] == 'hawker' and origin.get('address'):
            origin_name = origin['address']
        if destination['type'] == 'hawker' and destination.get('address'):
            destination_name = destination['address']
        
        logger.debug(f"Getting transit route from '{origin_name}' to '{destination_name}'")
        
        # Get the routes with transit mode
        routes = maps_client.get_route_directions(
            origin_name, 
            destination_name,
            mode="transit",
            departure_time=departure_time
        )
        
        if not routes:
            logger.warning(f"No transit routes found from {origin_name} to {destination_name}")
            return None
        
        logger.debug(f"Found {len(routes)} transit route options")
        
        # Parse the routes
        route_data = maps_client.parse_routes_to_json(
            routes, 
            origin_name, 
            destination_name
        )
        
        logger.debug(f"Transit route data parsed with {route_data.get('num_routes', 0)} routes")
        
        # Get detailed trip information including fares
        try:
            trip_details = get_trip_details(
                route_data, 
                rider_type="Adult", 
                sort_priority="price",
                departure_time=departure_time
            )
            logger.debug(f"Successfully processed transit trip details, found {len(trip_details) if trip_details else 0} options")
        except Exception as e:
            logger.error(f"Error in get_trip_details for transit: {e}")
            logger.error(traceback.format_exc())
            return None
        
        # Take the cheapest route
        if trip_details and len(trip_details) > 0:
            cheapest_route = trip_details[0]
            
            # Create a compact route entry
            try:
                route_entry = {
                    "origin_id": origin['id'],
                    "destination_id": destination['id'],
                    "mode": "transit",
                    "distance_km": cheapest_route["distance_km"],
                    "duration_minutes": cheapest_route["duration_minutes"],
                    "price_sgd": cheapest_route["price_sgd"],
                    "departure_time": cheapest_route.get("departure_time", "N/A"),
                    "arrival_time": cheapest_route.get("arrival_time", "N/A"),
                    "route_summary": [
                        {
                            "mode": step.get("travel_mode", "UNKNOWN"),
                            "line": step.get("line", {}).get("name", "") if step.get("travel_mode") == "TRANSIT" else "",
                            "vehicle_type": step.get("line", {}).get("vehicle_type", "") if step.get("travel_mode") == "TRANSIT" else "",
                            "duration": step.get("duration", "")
                        }
                        for step in cheapest_route["steps"]
                    ]
                }
                
                logger.debug(f"Successfully created transit route entry from {origin['name']} to {destination['name']}")
                return route_entry
            except KeyError as e:
                logger.error(f"Missing key in transit cheapest_route: {e}")
                logger.debug(f"Problematic transit cheapest_route: {cheapest_route}")
                return None
        else:
            logger.warning(f"No transit trip details found for route from {origin_name} to {destination_name}")
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting transit route from {origin.get('name', 'unknown')} to {destination.get('name', 'unknown')}: {e}")
        logger.error(traceback.format_exc())
        return None

def get_driving_route_for_pair(maps_client, origin, destination, departure_time=None):
    """Get driving route information for a pair of locations"""
    try:
        # Prepare location names
        origin_name = f"{origin['name']}, Singapore"
        destination_name = f"{destination['name']}, Singapore"
        
        # Use address if available (for hawker centers)
        if origin['type'] == 'hawker' and origin.get('address'):
            origin_name = origin['address']
        if destination['type'] == 'hawker' and destination.get('address'):
            destination_name = destination['address']
        
        logger.debug(f"Getting driving route from '{origin_name}' to '{destination_name}'")
        
        # Get the routes with driving mode
        routes = maps_client.get_route_directions(
            origin_name, 
            destination_name,
            mode="driving",
            departure_time=departure_time
        )
        
        if not routes:
            logger.warning(f"No driving routes found from {origin_name} to {destination_name}")
            return None
        
        logger.debug(f"Found {len(routes)} driving route options")
        
        # Parse the routes
        route_data = maps_client.parse_routes_to_json(
            routes, 
            origin_name, 
            destination_name
        )
        
        logger.debug(f"Driving route data parsed with {route_data.get('num_routes', 0)} routes")
        
        # For driving mode, we calculate the cost based on distance
        try:
            if 'routes' in route_data and len(route_data['routes']) > 0:
                route = route_data['routes'][0]  # Take the first (fastest) driving route
                
                # Extract distance and duration
                distance_km = route['distance']['value'] / 1000
                duration_minutes = route['duration']['value'] / 60
                
                # Calculate car fare based on distance (simplified Singapore taxi fare)
                # Base fare of $3.20 + $0.22 per 400m for first 10km + $0.22 per 350m after 10km
                price_sgd = 3.20  # Base fare
                
                if distance_km > 1:
                    remaining_km = distance_km - 1  # First 1 km is included in base fare
                    
                    if remaining_km <= 9:
                        # $0.22 per 400m for first 10km
                        price_sgd += (remaining_km * 1000 / 400) * 0.22
                    else:
                        # First 9km after base fare
                        price_sgd += (9 * 1000 / 400) * 0.22
                        
                        # Remaining distance after 10km at $0.22 per 350m
                        price_sgd += ((remaining_km - 9) * 1000 / 350) * 0.22
                
                # Create route entry for driving
                route_entry = {
                    "origin_id": origin['id'],
                    "destination_id": destination['id'],
                    "mode": "driving",
                    "distance_km": distance_km,
                    "duration_minutes": duration_minutes,
                    "price_sgd": round(price_sgd, 2),
                    "departure_time": "N/A",  # Driving mode doesn't have fixed departure times
                    "arrival_time": "N/A",    # Driving mode doesn't have fixed arrival times
                    "route_summary": [
                        {
                            "mode": "DRIVING",
                            "line": "",
                            "vehicle_type": "CAR",
                            "duration": route['duration']['text']
                        }
                    ]
                }
                
                logger.debug(f"Successfully created driving route entry from {origin['name']} to {destination['name']}")
                return route_entry
            else:
                logger.warning(f"No driving route found from {origin_name} to {destination_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing driving route: {e}")
            logger.error(traceback.format_exc())
            return None
        
    except Exception as e:
        logger.error(f"Error getting driving route from {origin.get('name', 'unknown')} to {destination.get('name', 'unknown')}: {e}")
        logger.error(traceback.format_exc())
        return None

def generate_route_matrix(attractions, hawker_centers, output_file="route_matrix.json"):
    """Generate a matrix of routes between all locations for both transit and driving"""
    logger.info("Starting route matrix generation")
    
    # Initialize Google Maps client
    try:
        maps_client = GoogleMapsClient()
        logger.info("Google Maps client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Google Maps client: {e}")
        return None
    
    # Set a default departure time (e.g., 10 AM on a weekday)
    departure_time = datetime(2025, 3, 20, 10, 0, 0)  # Saturday, 10 AM
    logger.info(f"Using departure time: {departure_time}")
    
    # Get all locations and pairs
    locations, pairs = generate_location_pairs(attractions, hawker_centers)
    
    # Store location details
    location_details = {loc['id']: loc for loc in locations}
    
    # Initialize route matrix
    route_matrix = {}
    
    # Process each pair and get route information
    logger.info("Generating route matrix...")
    for i, (origin, destination) in enumerate(tqdm(pairs)):
        logger.info(f"Processing pair {i+1}/{len(pairs)}: {origin['name']} to {destination['name']}")
        
        # Get transit route from origin to destination
        transit_forward_route = get_transit_route_for_pair(
            maps_client, 
            origin, 
            destination, 
            departure_time
        )
        
        # Get transit route from destination to origin
        transit_backward_route = get_transit_route_for_pair(
            maps_client, 
            destination, 
            origin, 
            departure_time
        )
        
        # Get driving route from origin to destination
        driving_forward_route = get_driving_route_for_pair(
            maps_client, 
            origin, 
            destination, 
            departure_time
        )
        
        # Get driving route from destination to origin
        driving_backward_route = get_driving_route_for_pair(
            maps_client, 
            destination, 
            origin, 
            departure_time
        )
        
        # Store transit routes in the matrix
        if transit_forward_route:
            route_id = f"{origin['id']}_to_{destination['id']}_transit"
            route_matrix[route_id] = transit_forward_route
            logger.debug(f"Added forward transit route: {route_id}")
        else:
            logger.warning(f"Could not get forward transit route from {origin['name']} to {destination['name']}")
        
        if transit_backward_route:
            route_id = f"{destination['id']}_to_{origin['id']}_transit"
            route_matrix[route_id] = transit_backward_route
            logger.debug(f"Added backward transit route: {route_id}")
        else:
            logger.warning(f"Could not get backward transit route from {destination['name']} to {origin['name']}")
        
        # Store driving routes in the matrix
        if driving_forward_route:
            route_id = f"{origin['id']}_to_{destination['id']}_driving"
            route_matrix[route_id] = driving_forward_route
            logger.debug(f"Added forward driving route: {route_id}")
        else:
            logger.warning(f"Could not get forward driving route from {origin['name']} to {destination['name']}")
        
        if driving_backward_route:
            route_id = f"{destination['id']}_to_{origin['id']}_driving"
            route_matrix[route_id] = driving_backward_route
            logger.debug(f"Added backward driving route: {route_id}")
        else:
            logger.warning(f"Could not get backward driving route from {destination['name']} to {origin['name']}")
        
        # Sleep to avoid hitting API rate limits
        time.sleep(0.5)
    
    # Create the final matrix data structure
    matrix_data = {
        "metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_locations": len(locations),
            "num_routes": len(route_matrix)
        },
        "locations": location_details,
        "routes": route_matrix
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to file
    try:
        with open(output_file, 'w') as f:
            json.dump(matrix_data, f, indent=2)
        logger.info(f"Route matrix saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving route matrix to file: {e}")
    
    return matrix_data

def main():
    """Main function to generate the route matrix"""
    logger.info("=== Starting route matrix generation script ===")
    
    # Connect to the database
    connection = connect_to_database()
    if not connection:
        logger.error("Failed to connect to database. Exiting.")
        return
    
    try:
        # Fetch data from database
        attractions = fetch_attractions(connection, limit=1)
        hawker_centers = fetch_hawker_centers(connection, limit=1)
        
        if not attractions:
            logger.error("No attractions found in database")
            return
            
        if not hawker_centers:
            logger.error("No hawker centers found in database")
            return
        
        # Generate route matrix
        generate_route_matrix(attractions, hawker_centers, "data/route_matrix.json")
        
    except Exception as e:
        logger.error(f"Unexpected error in main function: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Close database connection
        if connection and connection.is_connected():
            connection.close()
            logger.info("Database connection closed")
    
    logger.info("=== Route matrix generation script completed ===")

if __name__ == "__main__":
    main()