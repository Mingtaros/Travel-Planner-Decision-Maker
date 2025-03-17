"""
Generate optimized travel itineraries for Singapore

This script:
1. Takes user preferences, hotel location, and constraints
2. Uses the route matrix data to optimize a multi-day itinerary
3. Generates a day-by-day plan that maximizes satisfaction within constraints
"""

import os
import json
import random
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from google_maps.google_maps_client import GoogleMapsClient
import folium
from folium.plugins import MarkerCluster


class ItineraryGenerator:
    def __init__(self, route_matrix_file="data/route_matrix.json"):
        """Initialize the itinerary generator"""
        # Load route matrix
        self.load_route_matrix(route_matrix_file)
        
        # Initialize maps client for getting hotel coordinates
        self.maps_client = GoogleMapsClient()
        
        # Set default constraints
        self.constraints = {
            'max_budget_per_day': 200,  # SGD
            'max_hours_per_day': 10,    # hours
            'start_time': '09:00',      # 24-hour format
            'end_time': '21:00'         # 24-hour format
        }
        
        # Set default preferences (0-1 scale)
        self.preferences = {
            'nature': 0.5,
            'culture': 0.5,
            'shopping': 0.5,
            'food': 0.5
        }
        
        # Location coordinates cache
        self.coordinates = {}
    
    def load_route_matrix(self, file_path):
        """Load the route matrix from a file"""
        try:
            with open(file_path, 'r') as f:
                self.matrix_data = json.load(f)
            
            # Extract locations and routes for easier access
            self.locations = self.matrix_data['locations']
            self.routes = self.matrix_data['routes']
            
            # Create a list of location IDs
            self.location_ids = list(self.locations.keys())
            
            print(f"Loaded route matrix with {len(self.locations)} locations and {len(self.routes)} routes")
            return True
        except Exception as e:
            print(f"Error loading route matrix: {e}")
            return False
    
    def set_hotel(self, hotel_name):
        """Set the hotel as the starting point for each day"""
        self.hotel_name = hotel_name
        
        # Get hotel details from Google Maps
        place_details = self.maps_client.get_place_details(place_name=f"{hotel_name}, Singapore")
        place_data = self.maps_client.parse_place_details(place_details)
        
        if place_data and place_data.get('location'):
            self.hotel_location = {
                'lat': place_data['location']['lat'],
                'lng': place_data['location']['lng']
            }
            
            # Add hotel to our locations
            self.hotel_id = "H0"  # Special ID for hotel
            self.locations[self.hotel_id] = {
                'id': self.hotel_id,
                'name': hotel_name,
                'type': 'hotel',
                'expenditure': 0,
                'timespent': 0
            }
            
            print(f"Set hotel to {hotel_name} at coordinates {self.hotel_location}")
            
            # Calculate and add routes between hotel and all other locations
            self._calculate_hotel_routes()
            
            return True
        else:
            print(f"Could not find coordinates for hotel: {hotel_name}")
            return False
    
    def _calculate_hotel_routes(self):
        """Calculate routes between hotel and all other locations"""
        print("Calculating routes between hotel and all attractions/hawker centers...")
        
        # For each location, calculate routes to/from hotel
        for loc_id, details in list(self.locations.items()):
            if loc_id == self.hotel_id:
                continue
            
            # Estimate route from hotel to location
            route_to = {
                "origin_id": self.hotel_id,
                "destination_id": loc_id,
                "distance_km": 5.0,  # Default estimate
                "duration_minutes": 30.0,  # Default estimate
                "price_sgd": 2.0,  # Default estimate
                "route_summary": [
                    {
                        "mode": "TRANSIT",
                        "line": "Estimated",
                        "vehicle_type": "SUBWAY",
                        "duration": "30 mins"
                    }
                ]
            }
            
            # Estimate route from location to hotel
            route_from = {
                "origin_id": loc_id,
                "destination_id": self.hotel_id,
                "distance_km": 5.0,  # Default estimate
                "duration_minutes": 30.0,  # Default estimate
                "price_sgd": 2.0,  # Default estimate
                "route_summary": [
                    {
                        "mode": "TRANSIT",
                        "line": "Estimated",
                        "vehicle_type": "SUBWAY",
                        "duration": "30 mins"
                    }
                ]
            }
            
            # Add routes to matrix
            self.routes[f"{self.hotel_id}_to_{loc_id}"] = route_to
            self.routes[f"{loc_id}_to_{self.hotel_id}"] = route_from
    
    def set_user_preferences(self, preferences):
        """Set user preferences for different attraction types"""
        # Validate input preferences
        valid_preferences = {}
        for pref_type, value in preferences.items():
            if pref_type in ['nature', 'culture', 'shopping', 'food']:
                # Normalize to 0-1 scale
                valid_preferences[pref_type] = max(0, min(1, float(value)))
        
        # Update preferences
        self.preferences.update(valid_preferences)
        print(f"Updated user preferences: {self.preferences}")
    
    def set_constraints(self, constraints):
        """Set constraints for the itinerary"""
        # Validate and update constraints
        if 'max_budget_per_day' in constraints:
            self.constraints['max_budget_per_day'] = float(constraints['max_budget_per_day'])
        
        if 'max_hours_per_day' in constraints:
            self.constraints['max_hours_per_day'] = float(constraints['max_hours_per_day'])
        
        if 'start_time' in constraints:
            self.constraints['start_time'] = constraints['start_time']
        
        if 'end_time' in constraints:
            self.constraints['end_time'] = constraints['end_time']
        
        if 'num_days' in constraints:
            self.constraints['num_days'] = int(constraints['num_days'])
        
        print(f"Updated constraints: {self.constraints}")
    
    def calculate_location_scores(self):
        """Calculate a score for each location based on user preferences"""
        # Define attraction types and weights
        attraction_types = {
            'Gardens by the Bay': {'nature': 0.9, 'culture': 0.5},
            'National Museum': {'culture': 0.9, 'nature': 0.1},
            'Orchard Road': {'shopping': 0.9, 'food': 0.4},
            'Sentosa': {'nature': 0.7, 'culture': 0.4, 'shopping': 0.5},
            'Marina Bay Sands': {'shopping': 0.8, 'culture': 0.5},
            'Singapore Zoo': {'nature': 0.9, 'culture': 0.3},
            'Chinatown': {'culture': 0.8, 'food': 0.8, 'shopping': 0.6},
            'Little India': {'culture': 0.8, 'food': 0.8, 'shopping': 0.6},
            'Kampong Glam': {'culture': 0.8, 'food': 0.7, 'shopping': 0.6}
        }
        
        # Calculate a score for each location
        scores = {}
        for loc_id, details in self.locations.items():
            if loc_id == self.hotel_id:
                scores[loc_id] = 0  # Hotel has zero attraction score
                continue
            
            if details['type'] == 'attraction':
                # Find the best matching attraction type based on name
                best_match = None
                best_similarity = 0
                
                for attr_type in attraction_types:
                    if attr_type.lower() in details['name'].lower():
                        best_match = attr_type
                        best_similarity = 1
                        break
                
                # If no exact match, assign a default balanced score
                if best_match:
                    type_weights = attraction_types[best_match]
                    score = 0
                    for pref_type, pref_value in self.preferences.items():
                        if pref_type in type_weights:
                            score += pref_value * type_weights[pref_type]
                else:
                    # Default balanced score
                    score = (
                        self.preferences['nature'] * 0.3 + 
                        self.preferences['culture'] * 0.3 + 
                        self.preferences['shopping'] * 0.2 + 
                        self.preferences['food'] * 0.2
                    )
            
            elif details['type'] == 'hawker':
                # Hawker centers are primarily scored based on food preference
                score = self.preferences['food'] * 0.8
                
                # Add a bonus