import os
import json
import numpy as np
import logging

def read_llm_output(base_path):
    # List all folders in the base directory that follow the naming convention
    folders = [f for f in os.listdir(base_path) if f.isdigit() and 1 <= int(f) <= 99]
    
    if not folders:
        raise ValueError("No valid numbered folders found.")

    # Get the folder with the largest number
    latest_folder = max(folders, key=int)
    latest_folder_path = os.path.join(base_path, latest_folder)

    # Define file paths
    parameter_file = os.path.join(latest_folder_path, "moo_parameters.json")
    location_file = os.path.join(latest_folder_path, "POI_data.json")

    # Read JSON files
    def read_json(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    parameter_data = read_json(parameter_file)
    location_data = read_json(location_file)

    return parameter_data, location_data

def load_recommendations(json_path):
    """
    Load attraction and hawker recommendations from external JSON file
    
    Args:
        json_path: Path to the JSON file with recommendations
        
    Returns:
        dict: Dictionary with attraction and hawker recommendations
    """
    try:
        parameter_data, location_data = read_llm_output(json_path)
        
        recommendations = {
            'hawkers': [],
            'attractions': []
        }
        
        # Extract hawker recommendations
        if 'Hawker' in location_data:
            for hawker in location_data['Hawker']:
                recommendations['hawkers'].append({
                    'name': hawker['Hawker Name'],
                    'dish': hawker['Dish Name'],
                    'description': hawker['Description'],
                    'avg_price': hawker['Avg Food Price'],
                    'rating': hawker['Rating'],
                    'duration': hawker['Duration'] * 60, # Convert hours to minutes
                })
        
        # Extract attraction recommendations
        if 'Attraction' in location_data:
            for attraction in location_data['Attraction']:
                recommendations['attractions'].append({
                    'name': attraction['Attraction Name'],
                    'description': attraction['Description'],
                    'price': attraction['Entrance Fee'],
                    'rating': attraction['Rating'],
                    'duration': hawker['Duration'] * 60, # Convert hours to minutes
                })
        
        return recommendations, parameter_data
    except Exception as e:
        logging.error(f"Error loading recommendations: {e}")
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
                # Use recommendation data if available
                if "rating" not in loc:
                    loc["rating"] = rec["rating"]
                if "avg_food_price" not in loc:
                    loc["avg_food_price"] = rec["avg_price"]
                if "description" not in loc:
                    loc["description"] = rec["description"]
                if "dish" not in loc:
                    loc["dish"] = rec["dish"]
            else:
                # Use randomized values for missing data
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
                if "satisfaction" not in loc:
                    loc["satisfaction"] = rec["rating"]
                if "entrance_fee" not in loc:
                    loc["entrance_fee"] = rec["price"]
                if "description" not in loc:
                    loc["description"] = rec["description"]
            else:
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
        # If no recommendations, return original locations
        return locations
    
    # Create sets of recommended location names (lowercase for case-insensitive matching)
    recommended_hawkers = {hawker['name'].lower() for hawker in recommendations['hawkers']}
    recommended_attractions = {attraction['name'].lower() for attraction in recommendations['attractions']}
    
    # Separate locations by type
    hotels = [loc for loc in locations if loc["type"] == "hotel"]
    
    # Filter hawkers and attractions based on recommendations
    hawkers = []
    attractions = []
    
    for loc in locations:
        loc_name_lower = loc["name"].lower()
        
        if loc["type"] == "hawker":
            # Check if this hawker is in recommendations
            is_recommended = any(hawker_name in loc_name_lower or loc_name_lower in hawker_name 
                               for hawker_name in recommended_hawkers)
            
            if is_recommended:
                # Add recommended hawkers with high priority
                hawkers.append((loc, 1, loc.get("rating", 3.0)))
            else:
                # Add other hawkers with lower priority
                hawkers.append((loc, 0, loc.get("rating", 3.0)))
        
        elif loc["type"] == "attraction":
            # Check if this attraction is in recommendations
            is_recommended = any(attraction_name in loc_name_lower or loc_name_lower in attraction_name 
                               for attraction_name in recommended_attractions)
            
            if is_recommended:
                # Add recommended attractions with high priority
                attractions.append((loc, 1, loc.get("satisfaction", 5.0)))
            else:
                # Add other attractions with lower priority
                attractions.append((loc, 0, loc.get("satisfaction", 5.0)))
    
    # Sort hawkers and attractions by priority (recommended first) and then by rating
    hawkers.sort(key=lambda x: (-x[1], -x[2]))
    attractions.sort(key=lambda x: (-x[1], -x[2]))
    
    # Extract just the location dictionaries
    hawkers = [h[0] for h in hawkers]
    attractions = [a[0] for a in attractions]
    
    # Combine all locations with hotels first
    filtered_locations = hotels + attractions + hawkers
    
    return filtered_locations