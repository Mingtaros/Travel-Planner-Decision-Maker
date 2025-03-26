"""
Transport Matrix Utilities
========================

This module provides functions for loading and managing transportation data:
- Loading route matrices from JSON files 
- Converting between time formats and transport brackets
- Extracting location data from route matrices

The transport matrix is a key component for itinerary optimization,
providing travel times and costs between locations at different times of day.

Usage:
    # Load the full transport matrix
    transport_matrix = get_transport_matrix()
    
    # Get all locations from the transport data
    locations = get_all_locations()
    
    # Convert a time to the appropriate transport bracket
    hour_bracket = get_transport_hour(540)  # 9:00 AM -> 8
"""

import os
import json
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)

# Constants for fare calculations
FLAG_DOWN_FARE = 4.8
FIRST_TIER_RATE = 0.26  # per 400m
FIRST_TIER_DISTANCE = 400  # meters
SECOND_TIER_RATE = 0.26  # per 350m 
SECOND_TIER_DISTANCE = 350  # meters
FLAG_DOWN_COVERAGE = 1000  # meters
FIRST_TIER_MAX = 9000  # meters

def get_transport_matrix():
    """
    Load the complete transport matrix from cached JSON files.
    
    Reads route data for different time periods (morning, midday, evening, night)
    and consolidates them into a single transport matrix dictionary.
    
    Returns:
        dict: Transport matrix with the following structure:
            {(origin_name, destination_name, hour_bracket): 
                {
                    "transit": {"duration": minutes, "price": fare},
                    "drive": {"duration": minutes, "price": fare}
                }
            }
    
    Note:
        - hour_bracket is an integer (8, 12, 16, or 20) representing time of day
        - If files are missing or corrupted, partial data will be returned
        - Empty dictionary is returned in case of critical errors
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

def get_transport_hour(transport_time):
    """
    Convert a time value to the appropriate transport matrix time bracket.
    
    The transport matrix uses discrete time brackets (8, 12, 16, 20) to reduce
    data storage requirements. This function maps any time to the correct bracket.
    
    Args:
        transport_time (int): Time in minutes since start of day
                             (e.g., 540 = 9:00 AM, 720 = 12:00 PM)
    
    Returns:
        int: Transport hour bracket (8, 12, 16, or 20)
        
    Examples:
        >>> get_transport_hour(540)   # 9:00 AM
        8
        >>> get_transport_hour(750)   # 12:30 PM
        12
        >>> get_transport_hour(1020)  # 5:00 PM
        16
    """
    # Transport_matrix is bracketed to 4 groups, find the earliest applicable one
    brackets = [8, 12, 16, 20]
    transport_hour = transport_time // 60
    
    for bracket in reversed(brackets):
        if transport_hour >= bracket:
            return bracket
    
    return brackets[0]  # Default to first bracket if before 8 AM

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