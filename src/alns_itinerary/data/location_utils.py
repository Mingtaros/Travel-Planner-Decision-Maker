"""
Location Utilities
=================

This module provides functions for working with location data in travel itineraries:
- Hotel geocoding and waypoint creation
- Computing routes between hotels and attractions/food centers
- Location filtering and prioritization
- Integration of hotel data with existing location datasets

The module handles Google Maps API interactions and maintains a caching system
to reduce API calls and improve performance.

Usage examples:
    # Get hotel location from name
    hotel = get_hotel_waypoint("Marina Bay Sands")
    
    # Compute routes between hotel and attractions
    hotel_routes = compute_hotel_routes(hotel, attractions)
    
    # Filter locations to include the best-rated
    filtered = filter_locations(locations, max_attractions=10, max_hawkers=5)
"""
import logging
import numpy as np
from datetime import datetime

# Import required utilities
from utils.google_maps_client import GoogleMapsClient
from .trip_detail import calculate_public_transport_fare, calculate_car_fare
from .cache_manager import save_hotel_routes_to_cache, load_hotel_routes_from_cache

logger = logging.getLogger(__name__)

def get_hotel_waypoint(hotel_name):
    """
    Get hotel coordinates and details from a hotel name.
    
    Uses Google Maps API to geocode the hotel name and retrieve its location.
    If geocoding fails, returns a default hotel location in central Singapore.
    
    Args:
        hotel_name (str): Name of the hotel to geocode
        
    Returns:
        dict: Hotel location information with the following keys:
            - type: Always "hotel"
            - name: Hotel name (as provided or from Google)
            - lat: Latitude coordinate
            - lng: Longitude coordinate
            
    Note:
        This function gracefully handles errors and will return a default
        location if the hotel cannot be found or in case of API failure.
    """
    if not hotel_name:
        logger.warning("No hotel information provided, using default hotel")
        # Return a default hotel in central Singapore if none provided
        return {
            "type": "hotel",
            "name": "Marina Bay Sands",
            "lat": 1.2904527,  # Marina Bay Sands coordinates as default
            "lng": 103.8577566,
        }
    
    try:
        # Initialize Google Maps client with API key from environment variables
        maps_client = GoogleMapsClient()
        
        # Create search query
        search_query = f"{hotel_name}, Singapore"
        
        logger.info(f"Geocoding hotel: {search_query}")
        
        # Get place details using Google Maps API
        place_details = maps_client.get_place_details(place_name=search_query)
        place_data = maps_client.parse_place_details(place_details)
        
        if not place_data or 'location' not in place_data:
            logger.warning(f"Could not geocode hotel: {search_query}, using default location")
            return {
                "type": "hotel",
                "name": hotel_name,
                "lat": 1.2904527,  # Marina Bay Sands coordinates as default
                "lng": 103.8577566,
            }
        
        # Create and return hotel location information
        hotel = {
            "type": "hotel",
            "name": place_data.get('name', hotel_name),
            "lat": place_data['location']['lat'],
            "lng": place_data['location']['lng'],
        }
        
        logger.info(f"Successfully geocoded hotel: {hotel['name']} at {hotel['lat']}, {hotel['lng']}")
        return hotel
        
    except Exception as e:
        logger.error(f"Error geocoding hotel: {e}")
        # Return default hotel on error
        return {
            "type": "hotel",
            "name": hotel_name,
            "lat": 1.2904527,  # Default coordinates
            "lng": 103.8577566,
        }

def compute_hotel_routes(hotel, locations):
    """
    Compute travel routes between a hotel and all other locations.
    
    Calculates transit and driving routes for different times of day between
    the hotel and all attractions/hawker centers. Results are cached to avoid
    repeated API calls.
    
    Args:
        hotel (dict): Hotel location dictionary with name, lat, lng
        locations (list): List of location dictionaries for attractions/hawkers
        
    Returns:
        dict: Route matrix entries for the hotel with structure:
            {(origin_name, destination_name, hour): 
                {
                    "transit": {"duration": minutes, "price": fare},
                    "drive": {"duration": minutes, "price": fare}
                }
            }
            
    Note:
        This function makes multiple Google Maps API calls and may be rate-limited.
        Results are cached and will be reused when possible.
    """
    cached_routes = load_hotel_routes_from_cache(hotel, locations)
    if cached_routes is not None:
        return cached_routes
    
    try:
        # Initialize Google Maps client
        maps_client = GoogleMapsClient()
        
        # Time brackets used in the travel itinerary problem
        time_brackets = [8, 12, 16, 20]  # morning, midday, evening, night
        
        # Create a dictionary to store route information
        hotel_routes = {}
        
        # Create a hotel waypoint for the matrix calculation
        hotel_waypoint = [hotel["name"], hotel["lat"], hotel["lng"]]
        
        # Filter out other hotels from locations
        non_hotel_locations = [loc for loc in locations if loc["type"] != "hotel"]
        
        # Create waypoints list for the other locations
        destination_waypoints = [[loc["name"], loc["lat"], loc["lng"]] for loc in non_hotel_locations]
        
        # Log the operation
        logger.info(f"Computing route matrices for hotel '{hotel['name']}' with {len(destination_waypoints)} destinations")
        
        # For each time bracket, compute route matrix
        for hour in time_brackets:
            # Set departure time based on hour
            departure_time = datetime(2025, 5, 17, hour, 0, 0)
            
            # Compute transit route matrix (hotel to all locations)
            logger.info(f"Computing transit route matrix for hour {hour}...")
            transit_matrix = maps_client.compute_route_matrix(
                origins=[hotel_waypoint],
                destinations=destination_waypoints,
                mode="transit",
                departure_time=departure_time
            )
            
            # Compute driving route matrix (hotel to all locations)
            logger.info(f"Computing driving route matrix for hour {hour}...")
            driving_matrix = maps_client.compute_route_matrix(
                origins=[hotel_waypoint],
                destinations=destination_waypoints,
                mode="drive",
                departure_time=departure_time
            )
            
            # Compute transit route matrix (all locations to hotel)
            logger.info(f"Computing transit route matrix from locations to hotel for hour {hour}...")
            transit_matrix_return = maps_client.compute_route_matrix(
                origins=destination_waypoints,
                destinations=[hotel_waypoint],
                mode="transit",
                departure_time=departure_time
            )
            
            # Compute driving route matrix (all locations to hotel)
            logger.info(f"Computing driving route matrix from locations to hotel for hour {hour}...")
            driving_matrix_return = maps_client.compute_route_matrix(
                origins=destination_waypoints,
                destinations=[hotel_waypoint],
                mode="drive",
                departure_time=departure_time
            )
            
            # Process transit and driving routes
            for route in transit_matrix:
                if route.get('condition') == 'ROUTE_EXISTS':
                    dest_index = route.get('destinationIndex', 0)
                    if dest_index < len(non_hotel_locations):
                        dest_location = non_hotel_locations[dest_index]
                        
                        # Extract distance and duration
                        distance_meters = route.get('distanceMeters', 0)
                        distance_km = distance_meters / 1000
                        
                        duration_seconds = 0
                        if 'duration' in route:
                            duration_str = route['duration']
                            if duration_str.endswith('s'):
                                duration_seconds = int(duration_str[:-1])
                        
                        # Calculate fare
                        transit_fare = calculate_public_transport_fare(distance_km)
                        
                        # Store the route data
                        if (hotel["name"], dest_location["name"], hour) not in hotel_routes:
                            hotel_routes[(hotel["name"], dest_location["name"], hour)] = {
                                "transit": {
                                    "duration": round(duration_seconds / 60),  # Convert to minutes
                                    "price": round(transit_fare, 2) if transit_fare else 0,
                                },
                                "drive": {
                                    "duration": 0,
                                    "price": 0,
                                }
                            }
                        else:
                            hotel_routes[(hotel["name"], dest_location["name"], hour)]["transit"] = {
                                "duration": round(duration_seconds / 60),  # Convert to minutes
                                "price": round(transit_fare, 2) if transit_fare else 0,
                            }
            
            # Process hotel to locations routes (driving)
            for route in driving_matrix:
                if route.get('condition') == 'ROUTE_EXISTS':
                    dest_index = route.get('destinationIndex', 0)
                    if dest_index < len(non_hotel_locations):
                        dest_location = non_hotel_locations[dest_index]
                        
                        # Extract distance and duration
                        distance_meters = route.get('distanceMeters', 0)
                        distance_km = distance_meters / 1000
                        
                        duration_seconds = 0
                        if 'duration' in route:
                            duration_str = route['duration']
                            if duration_str.endswith('s'):
                                duration_seconds = int(duration_str[:-1])
                        
                        # Calculate fare
                        driving_fare = calculate_car_fare(distance_meters)
                        
                        # Store the route data
                        if (hotel["name"], dest_location["name"], hour) not in hotel_routes:
                            hotel_routes[(hotel["name"], dest_location["name"], hour)] = {
                                "transit": {
                                    "duration": 0,
                                    "price": 0,
                                },
                                "drive": {
                                    "duration": round(duration_seconds / 60),  # Convert to minutes
                                    "price": round(driving_fare, 2) if driving_fare else 0,
                                }
                            }
                        else:
                            hotel_routes[(hotel["name"], dest_location["name"], hour)]["drive"] = {
                                "duration": round(duration_seconds / 60),  # Convert to minutes
                                "price": round(driving_fare, 2) if driving_fare else 0,
                            }
            
            # Process locations to hotel routes (transit)
            for route in transit_matrix_return:
                if route.get('condition') == 'ROUTE_EXISTS':
                    origin_index = route.get('originIndex', 0)
                    if origin_index < len(non_hotel_locations):
                        origin_location = non_hotel_locations[origin_index]
                        
                        # Extract distance and duration
                        distance_meters = route.get('distanceMeters', 0)
                        distance_km = distance_meters / 1000
                        
                        duration_seconds = 0
                        if 'duration' in route:
                            duration_str = route['duration']
                            if duration_str.endswith('s'):
                                duration_seconds = int(duration_str[:-1])
                        
                        # Calculate fare
                        transit_fare = calculate_public_transport_fare(distance_km)
                        
                        # Store the route data
                        if (origin_location["name"], hotel["name"], hour) not in hotel_routes:
                            hotel_routes[(origin_location["name"], hotel["name"], hour)] = {
                                "transit": {
                                    "duration": round(duration_seconds / 60),  # Convert to minutes
                                    "price": round(transit_fare, 2) if transit_fare else 0,
                                },
                                "drive": {
                                    "duration": 0,
                                    "price": 0,
                                }
                            }
                        else:
                            hotel_routes[(origin_location["name"], hotel["name"], hour)]["transit"] = {
                                "duration": round(duration_seconds / 60),  # Convert to minutes
                                "price": round(transit_fare, 2) if transit_fare else 0,
                            }
            
            # Process locations to hotel routes (driving)
            for route in driving_matrix_return:
                if route.get('condition') == 'ROUTE_EXISTS':
                    origin_index = route.get('originIndex', 0)
                    if origin_index < len(non_hotel_locations):
                        origin_location = non_hotel_locations[origin_index]
                        
                        # Extract distance and duration
                        distance_meters = route.get('distanceMeters', 0)
                        distance_km = distance_meters / 1000
                        
                        duration_seconds = 0
                        if 'duration' in route:
                            duration_str = route['duration']
                            if duration_str.endswith('s'):
                                duration_seconds = int(duration_str[:-1])
                        
                        # Calculate fare
                        driving_fare = calculate_car_fare(distance_meters)
                        
                        # Store the route data
                        if (origin_location["name"], hotel["name"], hour) not in hotel_routes:
                            hotel_routes[(origin_location["name"], hotel["name"], hour)] = {
                                "transit": {
                                    "duration": 0,
                                    "price": 0,
                                },
                                "drive": {
                                    "duration": round(duration_seconds / 60),  # Convert to minutes
                                    "price": round(driving_fare, 2) if driving_fare else 0,
                                }
                            }
                        else:
                            hotel_routes[(origin_location["name"], hotel["name"], hour)]["drive"] = {
                                "duration": round(duration_seconds / 60),  # Convert to minutes
                                "price": round(driving_fare, 2) if driving_fare else 0,
                            }
        
        logger.info(f"Successfully computed {len(hotel_routes)} hotel route entries across {len(time_brackets)} time periods")
        
        return hotel_routes
        
    except Exception as e:
        import traceback
        logger.error(f"Error computing hotel routes: {e}")
        logger.error(traceback.format_exc())
        # Return empty dictionary on error
        return {}

def integrate_hotel_with_locations(hotel, locations, transport_matrix):
    """
    Integrate hotel data with existing locations and transport matrices.
    
    This function:
    1. Adds or updates the hotel in the locations list
    2. Computes routes between the hotel and all other locations
    3. Merges the hotel routes with the existing transport matrix
    
    Args:
        hotel (dict): Hotel location information
        locations (list): Existing list of location dictionaries
        transport_matrix (dict): Existing transport matrix
        
    Returns:
        tuple: (updated_locations, updated_transport_matrix)
            Returns (None, None) if hotel route computation fails
            
    Note:
        The hotel is always positioned at index 0 in the locations list
        to ensure it's treated as the starting point for itineraries.
    """
    # Check if hotel already exists in locations
    hotel_exists = False
    for i, loc in enumerate(locations):
        if loc["type"] == "hotel" and loc["name"] == hotel["name"]:
            # Hotel already exists, update its information
            locations[i] = hotel
            hotel_exists = True
            logger.info(f"Updated existing hotel: {hotel['name']}")
            break
    
    # If hotel doesn't exist, add it to locations (at index 0)
    if not hotel_exists:
        locations.insert(0, hotel)
        logger.info(f"Added new hotel: {hotel['name']}")
    
    # Compute routes between hotel and all locations
    hotel_routes = compute_hotel_routes(hotel, locations)
    
    save_hotel_routes_to_cache(hotel, hotel_routes)
    
    # Check if we have any hotel routes
    if not hotel_routes:
        logger.warning("Failed to compute hotel routes, using default values")
        # Create default routes (this is a fallback to prevent optimization failure)
        return None, None
    
    # Merge hotel routes with existing transport matrix
    updated_matrix = {**transport_matrix, **hotel_routes}
    logger.info(f"Added {len(hotel_routes)} hotel route entries to transport matrix")
    
    return locations, updated_matrix

def filter_locations(locations, max_attractions=None, max_hawkers=None, filter_criteria=None):
    """
    Filter and prioritize locations for itinerary planning.
    
    Sorts locations by criteria (e.g., rating, satisfaction) and limits
    the number of each type to include in the itinerary optimization.
    
    Args:
        locations (list): List of all location dictionaries
        max_attractions (int, optional): Maximum number of attractions to include
        max_hawkers (int, optional): Maximum number of hawkers to include
        filter_criteria (dict, optional): Criteria for sorting locations:
            {
                "attractions": field_name_for_sorting, # e.g., "satisfaction"
                "hawkers": field_name_for_sorting,     # e.g., "rating"
            }
    
    Returns:
        list: Filtered list of locations with hotels first, followed by
              the top attractions and hawkers based on criteria
    """
    # Sort hotel(s) to the front of the list
    hotels = [loc for loc in locations if loc["type"] == "hotel"]
    attractions = [loc for loc in locations if loc["type"] == "attraction"]
    hawkers = [loc for loc in locations if loc["type"] == "hawker"]
    
    logger.info(f"Filtering locations: {len(hotels)} hotels, {len(attractions)} attractions, {len(hawkers)} hawkers")
    
    # Default sorting criteria
    attraction_sort_key = "satisfaction" if not filter_criteria else filter_criteria.get("attractions", "satisfaction")
    hawker_sort_key = "rating" if not filter_criteria else filter_criteria.get("hawkers", "rating")
    
    # Sort attractions and hawkers by criteria (if present in data)
    if all(attraction_sort_key in loc for loc in attractions):
        attractions.sort(key=lambda x: x.get(attraction_sort_key, 0), reverse=True)
    
    if all(hawker_sort_key in loc for loc in hawkers):
        hawkers.sort(key=lambda x: x.get(hawker_sort_key, 0), reverse=True)
    
    # Limit the number of attractions and hawkers if specified
    if max_attractions is not None and max_attractions > 0:
        attractions = attractions[:max_attractions]
    
    if max_hawkers is not None and max_hawkers > 0:
        hawkers = hawkers[:max_hawkers]
    
    # Combine filtered locations, ensuring hotels come first
    filtered_locations = hotels + attractions + hawkers
    
    return filtered_locations