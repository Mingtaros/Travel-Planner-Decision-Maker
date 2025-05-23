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
    

def is_updated_itinerary_feasible(itinerary_response, budget, num_of_days, default_hotel_name="Marina Bay Sands Singapore") -> str:
    # if valid, return "valid"
    # otherwise, return the feedback, so the LLM know what's wrong with it
    # initialize constants
    START_TIME = "09:00"
    HARD_LIMIT_END_TIME = "22:00"

    LUNCH_START = "12:00"
    LUNCH_END = "15:00"
    DINNER_START = "18:00"
    DINNER_END = "21:00"
    MAX_ATTRACTION_PER_DAY = 4

    if num_of_days != len(itinerary_response["days"]):
        return f"Number of Days should be {num_of_days}, not {len(itinerary_response['days'])}."

    if budget < itinerary_response["trip_summary"]["actual_expenditure"]:
        return f"Itinerary went over budget. Budget: {budget}, Expenditure: {itinerary_response['trip_summary']['actual_expenditure']}."
    
    visited_attractions = set()
    for day in itinerary_response["days"]:
        # check that attractions are visited at most once across all days
        visited_hawkers = set()
        for location in day["locations"]:
            if location["type"] == "hawker":
                if location["name"] in visited_hawkers:
                    # for hawkers, can only visit once per day
                    return f"Day {day['day']}: {location['name']} Visited more than once."
                visited_hawkers.add(location["name"])

            if location["type"] == "attraction" and location["name"] in visited_attractions:
                # for attractions, can only visit once across all days
                return f"Day {day['day']}: {location['name']} Visited more than once."
            visited_attractions.add(location["name"])

        # check if first and last locations of every day, appear more than once
        if day["locations"][0]["name"] != default_hotel_name:
            return f"Day {day['day']} doesn't start at the hotel '{default_hotel_name}'."
        
        if day["locations"][-1]["name"] != default_hotel_name:
            return f"Day {day['day']} doesn't end at the hotel '{default_hotel_name}'."
    
        # check time of itinerary start and finish
        if day["locations"][0]["departure_time"] < START_TIME:
            # both in string
            return f"Day {day['day']} must start at {START_TIME}. Got {day['locations'][0]['departure_time']} instead."

        if day["locations"][-1]["departure_time"] > HARD_LIMIT_END_TIME:
            # both in string
            return f"Day {day['day']} must end at {HARD_LIMIT_END_TIME}. Got {day['locations'][-1]['departure_time']} instead."
        
        lunch_visits = 0
        dinner_visits = 0
        for location in day["locations"]:
            if location["type"] == "hawker":
                # check for lunch and dinner
                if LUNCH_START <= location["arrival_time"] <= LUNCH_END:
                    lunch_visits += 1
                
                if DINNER_START <= location["arrival_time"] <= DINNER_END:
                    dinner_visits += 1
        
        # number of lunches and dinners must be only 1 every day
        if lunch_visits != 1:
            return f"Day {day['day']} has {lunch_visits} lunch visits while only 1 is necessary."
        
        if dinner_visits != 1:
            return f"Day {day['day']} has {dinner_visits} dinner visits while only 1 is necessary."

        # check number of attractions visited
        num_attractions_visited = sum([
            1 if location["type"] == "attraction" else 0
            for location in day["locations"]
        ])

        if num_attractions_visited > MAX_ATTRACTION_PER_DAY:
            return f"Day {day['day']} have {num_attractions_visited} attractions visited. Maximum is {MAX_ATTRACTION_PER_DAY}."

    return "valid" # this is everything passed.
