import os
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)

def read_llm_output(base_path):
    # List all folders in the base directory that follow the naming convention
    # folders = [f for f in os.listdir(base_path) if f.isdigit() and 1 <= int(f) <= 99]
    
    # if not folders:
    #     raise ValueError("No valid numbered folders found.")

    # # Get the folder with the largest number
    # latest_folder = max(folders, key=int)
    # latest_folder_path = os.path.join(base_path, latest_folder)

    # Define file paths
    parameter_file = os.path.join(base_path, "moo_parameters.json")
    location_file = os.path.join(base_path, "final_combined_POI.json")

    # Read JSON files
    def read_json(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    parameter_data = read_json(parameter_file)
    location_data = read_json(location_file)

    return parameter_data, location_data

def load_recommendations(json_path=None, alns_input=None):
    """
    Load attraction and hawker recommendations from external JSON file
    
    Args:
        json_path: Path to the JSON file with recommendations
        
    Returns:
        dict: Dictionary with attraction and hawker recommendations
    """
    try:
        # logger.info("ALNS Input: %s", alns_input)
        parameter_data = {}
        recommendations = {
                'hawkers': [],
                'attractions': []
            }
        if alns_input:
            logger.info("Loading recommendations from ALNS input")
            alns_weights = alns_input.get('alns_weights', {})
            
            logger.info("ALNS Weights: %s", alns_weights)
            
            parameter_data["params"] = list(alns_weights.values())
            
            location_data = {k: alns_input[k] for k in ["attractions", "hawkers"]}
            # logger.info("Location Data: %s", location_data)
            
            for hawker in location_data['hawkers']:
                recommendations['hawkers'].append({
                    'name': hawker['name'],
                    'avg_food_price': round(float(hawker['avg_food_price']),2),
                    'rating': round(float(hawker['relevance_score']),1),
                    'duration': 60,
                })
                
            for attraction in location_data['attractions']:
                recommendations['attractions'].append({
                    'name': attraction['name'],
                    'entrance_fee': round(float(attraction['entrance_fee']),2),
                    'satisfaction': round(float(attraction['relevance_score']),1),
                    'duration': round(attraction.get('estimated_duration', 60))
                })
            
        elif json_path:
            logger.info(f"Loading recommendations from JSON file: {json_path}")
            parameter_data, location_data = read_llm_output(json_path)
            
            # Extract hawker recommendations
            if 'Hawker' in location_data:
                for hawker in location_data['Hawker']:
                    recommendations['hawkers'].append({
                        'name': hawker['Hawker Name'],
                        'avg_food_price': hawker['Avg Food Price'],
                        'rating': hawker['Satisfaction Score'],
                        'duration': hawker.get('Duration', 1) * 60, # Convert hours to minutes
                    })
            
            # Extract attraction recommendations
            if 'Attraction' in location_data:
                for attraction in location_data['Attraction']:
                    recommendations['attractions'].append({
                        'name': attraction['Attraction Name'],
                        'entrance_fee': attraction['Entrance Fee'],
                        'satisfaction': attraction['Satisfaction Score'],
                        'duration': attraction.get('Duration', 1) * 60, # Convert hours to minutes
                    })
        
        logger.info(f"Loaded {len(recommendations['hawkers'])} hawker recommendations and {len(recommendations['attractions'])} attraction recommendations")
        logger.info(f"Loaded parameters: {parameter_data}")
        
        return recommendations, parameter_data
    except Exception as e:
        logger.error(f"Error loading recommendations: {e}")
        return {'hawkers': [], 'attractions': []}
    
def augment_location_data(locations, recommendations=None):
    """
    Add missing data to locations with recommended data when available
    
    Args:
        locations: List of all location dictionaries
        recommendations: Dictionary with attraction and hawker recommendations
        
    Returns:
        list: Augmented location data
    """
    # Create lookup dictionaries for quick access to recommendations
    hawker_lookup = {}
    attraction_lookup = {}
    
    if recommendations:
        for hawker in recommendations['hawkers']:
            hawker_lookup[hawker['name'].lower()] = hawker
        
        for attraction in recommendations['attractions']:
            attraction_lookup[attraction['name'].lower()] = attraction
    
    # For all locations, add missing data with prioritizing recommendations
    for loc in locations:
        loc_name_lower = loc["name"].lower()
        
        if loc["type"] == "hawker":
            # Check if we have recommendation data for this hawker
            if loc_name_lower in hawker_lookup:
                rec = hawker_lookup[loc_name_lower]
                
                loc["rating"] = rec["rating"]
                loc["avg_food_price"] = rec["avg_food_price"]
            else:
                # Use randomized values for missing data
                logger.warning(f"No recommendation data found for hawker: {loc['name']}")
                if "rating" not in loc:
                    loc["rating"] = np.random.uniform(3, 5)  # More realistic ratings
                if "avg_food_price" not in loc:
                    loc["avg_food_price"] = np.random.uniform(5, 15)
            
            # Ensure duration is set
            if "duration" not in loc:
                loc["duration"] = 60  # standardize 60 mins for meals
                
        elif loc["type"] == "attraction":
            # Check if we have recommendation data for this attraction
            if loc_name_lower in attraction_lookup:
                rec = attraction_lookup[loc_name_lower]
                # Use recommendation data if available
                loc["satisfaction"] = rec["satisfaction"]
                loc["entrance_fee"] = rec["entrance_fee"]
            else:
                logger.warning(f"No recommendation data found for attraction: {loc['name']}")
                # Use randomized values for missing data
                if "satisfaction" not in loc:
                    loc["satisfaction"] = np.random.uniform(5, 10)  # More realistic ratings
                if "entrance_fee" not in loc:
                    loc["entrance_fee"] = np.random.uniform(5, 100)
            
            # Ensure duration is set
            if "duration" not in loc:
                loc["duration"] = np.random.randint(45, 120)  # More realistic durations
                
        elif loc["type"] == "hotel":
            # Set hotel duration to 0 (no time spent at hotel for activities)
            loc["duration"] = 0
    
    return locations

def filter_by_recommendations(locations, recommendations=None):
    """
    Filter locations to prioritize recommended attractions and hawkers
    
    Args:
        locations: List of all location dictionaries
        recommendations: Dictionary with attraction and hawker recommendations
        
    Returns:
        list: Filtered locations
    """
    if not recommendations or (not recommendations['attractions'] and not recommendations['hawkers']):
        return []
    
    # Create sets of recommended location names (lowercase for case-insensitive matching)
    recommended_hawkers = {hawker['name'].lower() for hawker in recommendations['hawkers']}
    recommended_attractions = {attraction['name'].lower() for attraction in recommendations['attractions']}
    
    # Separate locations by type
    hotels = [loc for loc in locations if loc["type"] == "hotel"]
    
    # List of recommendations not in the location data
    missing_hawkers = recommended_hawkers - {loc["name"].lower() for loc in locations if loc["type"] == "hawker"}
    missing_attractions = recommended_attractions - {loc["name"].lower() for loc in locations if loc["type"] == "attraction"}
    
    if missing_hawkers:
        logger.warning(f"Missing hawker recommendations: {missing_hawkers}")
    
    if missing_attractions:
        logger.warning(f"Missing attraction recommendations: {missing_attractions}")
    
    # Filter hawkers and attractions based on recommendations
    filtered_locations = [
        loc for loc in locations
        if (loc["type"] == "hawker" and loc["name"].lower() in recommended_hawkers) or
           (loc["type"] == "attraction" and loc["name"].lower() in recommended_attractions) or
           (loc["type"] == "hotel")
    ]
    
    return filtered_locations