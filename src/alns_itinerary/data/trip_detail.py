"""
Trip Detail Calculator
=====================

This module provides utilities for calculating transportation costs, travel times, 
and route details between locations. It supports multiple transportation modes:
- Car/taxi fare calculation
- Public transport fare calculation
- Trip details formatting and sorting

These functions can be used to compute the cost and time components needed for
itinerary optimization and to present detailed transportation information to users.

Usage:
    # Get detailed transit options sorted by price
    sorted_trips = get_trip_details(route_data, sort_priority="price")
    
    # Calculate car fare for a 12km journey
    fare = calculate_car_fare(12000)  # 12000 meters
"""

import re
import json
import datetime

# Constants for fare calculations
FLAG_DOWN_FARE = 4.8
FIRST_TIER_RATE = 0.26  # per 400m
FIRST_TIER_DISTANCE = 400  # meters
SECOND_TIER_RATE = 0.26  # per 350m 
SECOND_TIER_DISTANCE = 350  # meters
FLAG_DOWN_COVERAGE = 1000  # meters
FIRST_TIER_MAX = 9000  # meters

def calculate_car_fare(distance_m, flag_down=FLAG_DOWN_FARE):
    """
    Calculate taxi/car fare based on Singapore's standard taxi fare structure.
    
    The fare calculation follows typical Singapore taxi pricing:
    - Flag-down fare covers first 1km
    - $0.26 per 400m for the next 9km
    - $0.26 per 350m for any distance beyond 10km
    
    Args:
        distance_m (float): Distance in meters
        flag_down (float, optional): Flag-down fare in SGD (default: 4.8)
        
    Returns:
        float: Calculated fare in SGD, rounded to 2 decimal places
        
    Example:
        >>> calculate_car_fare(5000)
        9.88  # Flag-down $4.8 + (4000m / 400m) * $0.26
    """
    # Start with flag-down fare
    fare = flag_down
    
    # Skip calculation if distance is within flag-down coverage
    if distance_m <= FLAG_DOWN_COVERAGE:
        return round(fare, 2)
    
    # Calculate distance beyond flag-down
    remaining_m = distance_m - FLAG_DOWN_COVERAGE
    
    # First 9km after flag-down (at $0.26 per 400m)
    first_tier_distance = min(remaining_m, FIRST_TIER_MAX)
    fare += (first_tier_distance // FIRST_TIER_DISTANCE) * FIRST_TIER_RATE
    if first_tier_distance % FIRST_TIER_DISTANCE > 0:
        fare += FIRST_TIER_RATE
    
    # Distance beyond 10km total (at $0.26 per 350m)
    if remaining_m > FIRST_TIER_MAX:
        second_tier_distance = remaining_m - FIRST_TIER_MAX
        fare += (second_tier_distance // SECOND_TIER_DISTANCE) * SECOND_TIER_RATE
        if second_tier_distance % SECOND_TIER_DISTANCE > 0:
            fare += SECOND_TIER_RATE
    
    return round(fare, 2)

def calculate_public_transport_fare(distance_km):
    """
    Calculate public transportation fare based on distance traveled.
    
    Uses Singapore's distance-based fare system with a table of fare brackets.
    The fare increases with distance traveled up to a maximum of $2.47.
    
    Args:
        distance_km (float): Distance in kilometers
        
    Returns:
        float: Calculated fare in SGD
        
    Example:
        >>> calculate_public_transport_fare(5.0)
        1.40
    """
    fare_table_brackets = [
        (3.2, 1.19), (4.2, 1.29), (5.2, 1.40), (6.2, 1.50), (7.2, 1.59),
        (8.2, 1.66), (9.2, 1.73), (10.2, 1.77), (11.2, 1.81), (12.2, 1.85),
        (13.2, 1.89), (14.2, 1.93), (15.2, 1.98), (16.2, 2.02), (17.2, 2.06),
        (18.2, 2.10), (19.2, 2.14), (20.2, 2.17), (21.2, 2.20), (22.2, 2.23),
        (23.2, 2.26), (24.2, 2.28), (25.2, 2.30), (26.2, 2.32), (27.2, 2.33),
        (28.2, 2.34), (29.2, 2.35), (30.2, 2.36), (31.2, 2.37), (32.2, 2.38),
        (33.2, 2.39), (34.2, 2.40), (35.2, 2.41), (36.2, 2.42), (37.2, 2.43),
        (38.2, 2.44), (39.2, 2.45), (40.2, 2.46), (float('inf'), 2.47)
    ]
    
    # Binary search for the appropriate fare bracket
    low, high = 0, len(fare_table_brackets) - 1
    
    while low <= high:
        mid = (low + high) // 2
        limit, fare = fare_table_brackets[mid]
        
        if distance_km <= limit:
            # Look for a potentially lower bracket
            high = mid - 1
        else:
            low = mid + 1
    
    # After binary search, 'low' is the index of the smallest bracket
    # that is greater than or equal to the distance
    if low < len(fare_table_brackets):
        return fare_table_brackets[low][1]
    else:
        return fare_table_brackets[-1][1]  # Return highest fare

def process_driving_route(route, departure_time):
    """Process a driving route and calculate fare."""
    price = calculate_car_fare(route["distance"]["value"], flag_down=4.8)
    arriving_time = departure_time + datetime.timedelta(seconds=route["duration"]["value"])
    
    return {
        "price_sgd": price,
        "distance_km": route["distance"]["value"] / 1000,
        "duration_minutes": round(route["duration"]["value"] / 60),
        "departure_time": departure_time.strftime("%-I:%M %p"),
        "arrival_time": arriving_time.strftime("%-I:%M %p"),
        "steps": route["steps"],
    }
    
def process_transit_route(route):
    """Process a transit route and calculate fare."""
    accumulated_price = 0
    for step in route["steps"]:
        if step["travel_mode"] == "TRANSIT":
            distance_km = float(step["distance"].split()[0])
            accumulated_price += calculate_public_transport_fare(distance_km)
    
    return {
        "price_sgd": accumulated_price,
        "distance_km": route["distance"]["value"] / 1000,
        "duration_minutes": round(route["duration"]["value"] / 60),
        "departure_time": route["departure_time"].replace("\u202f", " "),
        "arrival_time": route["arrival_time"].replace("\u202f", " "),
        "steps": route["steps"],
    }

def get_trip_details(route_data, sort_priority="price", departure_time=datetime.datetime.now()):
    """
    Extract and calculate detailed trip information from route data.
    
    Processes route data (typically from Google Maps API) to extract:
    - Price estimations for both driving and public transport
    - Distance and duration
    - Departure and arrival times
    - Step-by-step navigation details
    
    Results are sorted according to the priority (price or arrival time).
    
    Args:
        route_data (dict): Route information typically from Google Maps API
        sort_priority (str): Sorting criterion - "price" or "time" (default: "price")
        departure_time (datetime): Starting time of the trip (default: current time)
        
    Returns:
        list: List of dictionaries containing detailed trip information, sorted by
              the specified priority
              
    Example:
        >>> route_data = get_directions("Orchard Road", "Marina Bay Sands")
        >>> trip_details = get_trip_details(route_data, sort_priority="time")
        >>> print(f"Fastest trip: {trip_details[0]['duration_minutes']} minutes")
    """
    trip_details = []
    
    for route in route_data["routes"]:
        if route["steps"][0]["travel_mode"] == "DRIVING":
            details = process_driving_route(route, departure_time)
        else:
            details = process_transit_route(route)
        
        trip_details.append(details)
    
    # Sort by appropriate criteria
    key_func = lambda x: (x["price_sgd"], x["arrival_time"]) if sort_priority == "price" else x["arrival_time"]
    return sorted(trip_details, key=key_func)
