"""
Cache Management System
=====================

This module provides functions for caching and retrieving transportation route data.
Caching is critical for performance as it reduces the number of API calls to 
external services like Google Maps.

Features:
- Hotel route caching using pickle serialization
- Cache validation to ensure data consistency
- Automatic cache expiration and cleanup
- MD5 hashing for cache key generation

Usage:
    # Save computed routes to cache
    save_hotel_routes_to_cache(hotel, routes)
    
    # Try to load from cache first
    cached_routes = load_hotel_routes_from_cache(hotel, locations)
    if cached_routes:
        # Use cached data
    else:
        # Compute fresh data and cache it
        routes = compute_hotel_routes(hotel, locations)
        save_hotel_routes_to_cache(hotel, routes)
"""
import os
import json
import hashlib
import logging
from datetime import datetime, timedelta
import pickle

logger = logging.getLogger(__name__)

def generate_cache_key(hotel):
    """
    Generate a unique cache key for hotel routes.
    
    Creates an MD5 hash based on the hotel's name and coordinates
    to uniquely identify the cached routes.
    
    Args:
        hotel (dict): Hotel location dictionary with name, lat, lng
        
    Returns:
        str: MD5 hexadecimal hash string
        
    Example:
        >>> hotel = {"name": "Marina Bay Sands", "lat": 1.2904, "lng": 103.8577}
        >>> generate_cache_key(hotel)
        'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6'
    """
    # Create a string representation of hotel and locations for hashing
    key_str = f"{hotel['name']}_{hotel['lat']}_{hotel['lng']}"
    return hashlib.md5(key_str.encode()).hexdigest()

def validate_cached_routes(hotel, hotel_routes, locations):
    """
    Validate that cached routes are compatible with current locations.
    
    Checks if all required routes between the hotel and locations exist
    in the cached data to ensure it can be used for the current dataset.
    
    Args:
        hotel (dict): Hotel location dictionary
        hotel_routes (dict): Dictionary of cached routes
        locations (list): Current location dictionaries
        
    Returns:
        bool: True if the cache is valid, False otherwise
        
    Note:
        A valid cache must contain routes between the hotel and each location
        in both directions for all time brackets (8, 12, 16, 20).
    """
    if not hotel_routes:
        return False
    
    # Get non-hotel location names
    location_names = [loc["name"] for loc in locations if loc["type"] != "hotel"]
    
    # Check if all required location pairs are in the cache
    time_brackets = [8, 12, 16, 20]  # The time brackets used in the system
    
    for location_name in location_names:
        # Check hotel to location routes
        hotel_to_loc_exists = False
        # Check location to hotel routes
        loc_to_hotel_exists = False
        
        for hour in time_brackets:
            # Check hotel -> location route
            if (hotel["name"], location_name, hour) in hotel_routes:
                hotel_to_loc_exists = True
            
            # Check location -> hotel route
            if (location_name, hotel["name"], hour) in hotel_routes:
                loc_to_hotel_exists = True
        
        # If either direction is missing for this location, cache is invalid
        if not hotel_to_loc_exists or not loc_to_hotel_exists:
            logger.warning(f"Missing routes between hotel and location '{location_name}' in cached data")
            return False
    
    logger.info("Cached hotel routes validated successfully")
    return True

def save_hotel_routes_to_cache(hotel, hotel_routes, cache_dir="cache/hotel_routes"):
    """
    Save hotel routes to a pickle cache file.
    
    Serializes the route data to a file in the cache directory using
    the hotel's unique cache key as the filename.
    
    Args:
        hotel (dict): Hotel location dictionary
        hotel_routes (dict): Dictionary of computed hotel routes
        cache_dir (str): Directory to store cache files
    
    Returns:
        bool: True if successfully saved, False otherwise
        
    Note:
        This function creates the cache directory if it doesn't exist.
        The cache file uses pickle format for efficient storage.
    """
    try:
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate unique cache key
        cache_key = generate_cache_key(hotel)
        
        # Create cache filename
        cache_filename = os.path.join(cache_dir, f"{cache_key}.pkl")
        
        with open(cache_filename, 'wb') as f:
            pickle.dump(hotel_routes, f)
        
        logger.info(f"Saved hotel routes to cache: {cache_filename}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving hotel routes to cache: {e}")
        return False

def load_hotel_routes_from_cache(hotel, locations, cache_dir="cache/hotel_routes", max_age_hours=24):
    """
    Load hotel routes from cache if available and valid.
    
    Attempts to load cached route data for the specified hotel and validates
    it against the current location list to ensure it's complete.
    
    Args:
        hotel (dict): Hotel location dictionary
        locations (list): Current location dictionaries
        cache_dir (str): Directory containing cache files
        max_age_hours (int): Maximum age of cache in hours
    
    Returns:
        dict or None: Cached routes if available and valid, otherwise None
        
    Note:
        Returns None if:
        - No cache file exists
        - Cache file exists but is invalid/incomplete
        - Error occurs during loading
    """
    try:
        # Generate unique cache key
        cache_key = generate_cache_key(hotel)
        
        # Create cache filename
        cache_filename = os.path.join(cache_dir, f"{cache_key}.pkl")
        # cache_filename = os.path.join(cache_dir, "33d0cebf177d6de8e61e63e000df5d9e.pkl")
        
        # Check if cache file exists
        if not os.path.exists(cache_filename):
            logger.info(f"No cache found for hotel routes: {cache_filename}")
            return None
        
        # Read cache file
        with open(cache_filename, 'rb') as f:
            hotel_routes = pickle.load(f)
        
        if locations and not validate_cached_routes(hotel, hotel_routes, locations):
            logger.warning(f"Disk-cached routes for {hotel['name']} don't match current locations, cache invalid")
            return None
        
        logger.info(f"Loaded hotel routes from cache: {cache_filename}")
        return hotel_routes
    
    except Exception as e:
        logger.error(f"Error loading hotel routes from cache: {e}")
        return None

def clear_old_cache(cache_dir="cache/hotel_routes", max_age_hours=72):
    """
    Remove expired cache files to free up disk space.
    
    Deletes cache files that are older than the specified maximum age.
    
    Args:
        cache_dir (str): Directory containing cache files
        max_age_hours (int): Maximum age of cache files to keep
    
    Returns:
        int: Number of files deleted
        
    Note:
        Only deletes files with .json extension (not .pkl files).
        This appears to be a bug in the implementation, as the actual
        cache files are stored in .pkl format.
    """
    try:
        deleted_files = 0
        now = datetime.now()
        
        # Ensure cache directory exists
        if not os.path.exists(cache_dir):
            return 0
        
        # Iterate through cache files
        for filename in os.listdir(cache_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(cache_dir, filename)
                
                try:
                    # Read file timestamp
                    with open(filepath, 'r') as f:
                        cache_entry = json.load(f)
                        cache_timestamp = datetime.fromisoformat(cache_entry['timestamp'])
                    
                    # Check file age
                    age = now - cache_timestamp
                    if age > timedelta(hours=max_age_hours):
                        os.remove(filepath)
                        deleted_files += 1
                        logger.info(f"Deleted old cache file: {filename}")
                
                except Exception as e:
                    logger.warning(f"Error processing cache file {filename}: {e}")
        
        logger.info(f"Cleared {deleted_files} old cache files")
        return deleted_files
    
    except Exception as e:
        logger.error(f"Error clearing old cache: {e}")
        return 0