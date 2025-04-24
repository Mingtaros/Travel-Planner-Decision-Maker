import json
import os
import logging

logger = logging.getLogger(__name__)


def get_poi_time_bracket(poi_time):
        TIME_BRACKETS = ["08:00", "12:00", "16:00", "20:00"]
        
        for bracket in reversed(TIME_BRACKETS):
            if poi_time >= bracket:
                return bracket
        
        return TIME_BRACKETS[0]


def get_location_types():
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
        location_types = {}
        for location_id, location_data in route_matrix.get("locations", {}).items():
            # Determine location type based on name or other heuristics
            location_type = "attraction"  # Default assumption
            
            # You might want to add more sophisticated type detection logic here
            if "hotel" in location_data.get("type", "").lower():
                location_type = "hotel"
            elif "food centre" in location_data.get("type", "").lower() or "hawker" in location_data.get("type", "").lower():
                location_type = "hawker"

            location = location_data.get("name", "")
            location_types[location] = location_type
        
        logger.info(f"Retrieved {len(location_types)} locations")
        return location_types
    
    except Exception as e:
        logger.error(f"Unexpected error retrieving locations: {e}")
        return {}


def get_transport_matrix():
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
                    time_brackets = {
                        "morning": "08:00",
                        "midday": "12:00",
                        "evening": "16:00",
                        "night": "20:00"
                    }
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
                            "duration": round(route_data.get("transit", {}).get("duration_minutes", 0)),
                            "price": round(route_data.get("transit", {}).get("fare_sgd", 0), 2)
                        },
                        "drive": {
                            "duration": round(route_data.get("drive", {}).get("duration_minutes", 0)),
                            "price": round(route_data.get("drive", {}).get("fare_sgd", 0), 2)
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
    