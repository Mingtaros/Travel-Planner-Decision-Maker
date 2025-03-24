import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def get_transport_matrix():
    """
    Load the transport matrix from cached JSON files
    
    Returns:
        dict: Comprehensive transport matrix with routes and costs
    """
    try:
        # Determine the base path for route data
        base_path = os.path.join("data", "routeData")
        
        # Time periods for route matrices
        time_periods = ["morning", "midday", "evening", "night"]
        
        # Consolidated transport matrix
        transport_matrix = {}
        
        # Load route matrices for each time period
        for period in time_periods:
            filepath = os.path.join(base_path, f"route_matrix_{period}.json")
            
            # Check if file exists
            if not os.path.exists(filepath):
                logger.warning(f"Route matrix file not found: {filepath}")
                continue
            
            try:
                with open(filepath, 'r') as f:
                    route_matrix = json.load(f)
                
                # Map route data to standard format
                for route_key, route_data in route_matrix.get("routes", {}).items():
                    # Determine time bracket
                    time_brackets = {"morning": 8, "midday": 12, "evening": 16, "night": 20}
                    time_bracket = time_brackets.get(period, 8)
                    
                    # Create route key tuple
                    matrix_key = (
                        route_data.get("origin_name", ""),
                        route_data.get("destination_name", ""),
                        time_bracket
                    )
                    
                    # Add route to transport matrix
                    transport_matrix[matrix_key] = {
                        "transit": {
                            "duration": route_data.get("transit", {}).get("duration_minutes", 0),
                            "price": route_data.get("transit", {}).get("fare_sgd", 0)
                        },
                        "drive": {
                            "duration": route_data.get("drive", {}).get("duration_minutes", 0),
                            "price": route_data.get("drive", {}).get("fare_sgd", 0)
                        }
                    }
                
                logger.info(f"Loaded route matrix for {period}")
            
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in route matrix file: {filepath}")
            except Exception as e:
                logger.error(f"Error loading route matrix {filepath}: {e}")
        
        logger.info(f"Total routes in transport matrix: {len(transport_matrix)}")
        return transport_matrix
    
    except Exception as e:
        logger.error(f"Unexpected error loading transport matrix: {e}")
        return {}

def get_all_locations():
    """
    Retrieve all locations from the route matrix
    
    Returns:
        list: List of location dictionaries
    """
    try:
        # Determine the base path for route data
        base_path = os.path.join("data", "routeData")
        filepath = os.path.join(base_path, "route_matrix_morning.json")
        
        # Check if file exists
        if not os.path.exists(filepath):
            logger.error(f"Route matrix file not found: {filepath}")
            return []
        
        # Load locations from the first route matrix file
        with open(filepath, 'r') as f:
            route_matrix = json.load(f)
        
        # Convert locations to standard format
        locations = []
        for location_id, location_data in route_matrix.get("locations", {}).items():
            # Determine location type based on name or other heuristics
            location_type = "attraction"  # Default assumption
            
            # You might want to add more sophisticated type detection logic here
            if "hotel" in location_data.get("type", "").lower():
                location_type = "hotel"
            elif "food centre" in location_data.get("type", "").lower() or "hawker" in location_data.get("type", "").lower():
                location_type = "hawker"
            
            # Construct location dictionary
            location = {
                "id": location_id,
                "name": location_data.get("name", ""),
                "type": location_type,
                "lat": location_data.get("lat", 0),
                "lng": location_data.get("lng", 0)
            }
            
            locations.append(location)
        
        logger.info(f"Retrieved {len(locations)} locations")
        return locations
    
    except Exception as e:
        logger.error(f"Unexpected error retrieving locations: {e}")
        return []

def get_transport_hour(transport_time):
    """
    Convert transport time to the nearest transport matrix bracket
    
    Args:
        transport_time: Time in minutes since day start
    
    Returns:
        int: Transport hour bracket (8, 12, 16, or 20)
    """
    # Transport_matrix is bracketed to 4 groups, find the earliest applicable one
    brackets = [8, 12, 16, 20]
    transport_hour = transport_time // 60
    
    for bracket in reversed(brackets):
        if transport_hour >= bracket:
            return bracket
    
    return brackets[0]  # Default to first bracket if before 8 AM