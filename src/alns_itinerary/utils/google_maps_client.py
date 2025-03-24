"""
Google Maps API Client
=====================

A comprehensive interface for Google Maps API interactions, supporting:
- Place details retrieval and parsing
- Route matrix calculation for travel planning
- Multiple travel modes (transit, driving)

Requirements:
- Google Maps API key (set in environment or passed directly)
- googlemaps and requests Python packages

Usage Example:
    client = GoogleMapsClient()
    place_details = client.get_place_details(place_name="Marina Bay Sands")
    parsed_details = client.parse_place_details(place_details)
"""

import os
import json
import logging
from datetime import datetime

import googlemaps
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class GoogleMapsClient:
    """
    Client for Google Maps API interactions.
    
    This class provides a unified interface for various Google Maps API endpoints,
    including Places API for location details and Routes API for travel planning.
    All methods handle errors gracefully and provide detailed logging.
    
    Attributes:
        gmaps: The Google Maps Python client instance
        api_key: The Google Maps API key being used
    """
    def __init__(self, api_key=None):
        """
        Initialize the Google Maps client
        
        Args:
            api_key (str, optional): Google Maps API key
        """
        # Load environment variables
        load_dotenv()
        
        # Determine API key
        if api_key is None:
            api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        
        # Validate API key
        if not api_key:
            raise ValueError("Google Maps API key is required. "
                             "Set GOOGLE_MAPS_API_KEY in .env file or provide directly.")
        
        # Initialize clients
        self.gmaps = googlemaps.Client(key=api_key)
        self.api_key = api_key
        
        # Logging
        logger.info("Google Maps client initialized successfully")
    
    def get_place_details(self, place_id=None, place_name=None, language="en"):
        """
        Retrieve detailed information about a place from Google Maps.
        
        This method can work with either a place_id or place_name. If only place_name
        is provided, it will first search for the place and then retrieve its details.
        
        Args:
            place_id (str, optional): Google Maps place ID for direct lookup
            place_name (str, optional): Name of the place to search for
            language (str): Language code for localized results (default: "en")
        
        Returns:
            dict: Detailed place information or None if not found/error
        
        Note:
            The returned data includes name, address, coordinates, rating, 
            opening hours, and other available details.
        """
        try:
            # If only place name is provided, first search for the place
            if place_id is None and place_name:
                # Search for the place
                search_results = self.gmaps.places(query=place_name, language=language)
                
                if not search_results.get('results'):
                    logger.warning(f"No places found for query: {place_name}")
                    return None
                
                # Use the first result's place ID
                place_id = search_results['results'][0]['place_id']
            
            # Retrieve detailed place information
            place_details = self.gmaps.place(
                place_id=place_id,
                fields=[
                    'name', 'formatted_address', 'geometry', 'formatted_phone_number', 
                    'website', 'rating', 'user_ratings_total', 'opening_hours', 
                    'price_level', 'types'
                ],
                language=language
            )
            
            return place_details
        
        except Exception as e:
            logger.error(f"Error retrieving place details: {e}")
            return None
    
    def parse_place_details(self, place_details):
        """
        Parse Google Maps place details into a simplified format
        
        Args:
            place_details (dict): Raw place details from Google Maps API
        
        Returns:
            dict: Simplified place information
        """
        if not place_details or 'result' not in place_details:
            return None
        
        result = place_details['result']
        
        # Extract location coordinates
        location = result.get('geometry', {}).get('location', {})
        
        # Prepare parsed data
        parsed_place = {
            'name': result.get('name', ''),
            'address': result.get('formatted_address', ''),
            'location': {
                'lat': location.get('lat', 0),
                'lng': location.get('lng', 0)
            },
            'phone': result.get('formatted_phone_number', ''),
            'website': result.get('website', ''),
            'rating': result.get('rating', 0),
            'total_ratings': result.get('user_ratings_total', 0),
            'price_level': result.get('price_level', 0),
            'types': result.get('types', [])
        }
        
        # Parse opening hours if available
        if 'opening_hours' in result:
            hours = result['opening_hours']
            parsed_place['opening_hours'] = {
                'open_now': hours.get('open_now', False),
                'weekday_text': hours.get('weekday_text', [])
            }
        
        return parsed_place
    
    def compute_route_matrix(self, origins, destinations=None, mode="transit", departure_time=None):
        """
        Calculate routes between multiple origins and destinations.
        
        Uses the Google Routes Matrix API to compute travel times, distances, and routes
        between multiple points. Particularly useful for trip planning applications.
        
        Args:
            origins: List of origin points in one of these formats:
                    - [lat, lng]
                    - [name, lat, lng]
            destinations: List of destination points (same format as origins)
                        If None, uses origins as destinations
            mode: Travel mode, either "transit" or "drive"
            departure_time: Departure datetime object (default: current time)
        
        Returns:
            List of dictionary objects with route information including:
            - duration
            - distance
            - route details
            
        Note:
            For transit mode, departure_time is used for public transit scheduling.
        """
        # Use current time if no departure time specified
        if departure_time is None:
            departure_time = datetime.now()
        
        # Ensure destinations matches origins if not provided
        if destinations is None:
            destinations = origins
        
        try:
            # Prepare API request
            url = 'https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix'
            headers = {
                'Content-Type': 'application/json',
                'X-Goog-Api-Key': self.api_key,
                'X-Goog-FieldMask': '*'  # Get all fields
            }
            
            # Prepare waypoints
            def format_waypoint(point):
                """Format a single waypoint"""
                if isinstance(point, (list, tuple)):
                    lat = point[1] if len(point) >= 3 else point[0]
                    lng = point[2] if len(point) >= 3 else point[1]
                    
                    return {
                        "waypoint": {
                            "location": {
                                "latLng": {
                                    "latitude": lat,
                                    "longitude": lng
                                }
                            }
                        }
                    }
                else:
                    raise ValueError(f"Invalid waypoint format: {point}")
            
            # Format origins and destinations
            formatted_origins = [format_waypoint(origin) for origin in origins]
            formatted_destinations = [format_waypoint(dest) for dest in destinations]
            
            # Prepare payload
            payload = {
                "origins": formatted_origins,
                "destinations": formatted_destinations,
                "travelMode": mode.upper()
            }
            
            # Add departure time for transit mode
            if mode.lower() == "transit":
                payload["departureTime"] = departure_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Make the API request
            response = requests.post(url, headers=headers, json=payload)
            
            # Check response
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Route matrix request failed: {response.text}")
                return []
        
        except Exception as e:
            logger.error(f"Error computing route matrix: {e}")
            return []