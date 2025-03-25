"""
Travel Itinerary Problem Definition
==================================

This module defines the TravelItineraryProblem class, which encapsulates all the
parameters, constraints, and validation logic for multi-day travel itinerary planning.

The problem includes:
- Budget constraints
- Time window constraints for activities and meals
- Location data (attractions, food centers, hotels)
- Transportation options between locations
- Feasibility validation

Usage:
    problem = TravelItineraryProblem(
        budget=1000,
        locations=location_data,
        transport_matrix=transport_data,
        num_days=3
    )
    
    if problem.is_feasible:
        solution = optimizer.solve(problem)
    else:
        print("No feasible solutions exist")
"""

import numpy as np
import logging
import os
from utils.config import load_config

logger = logging.getLogger(__name__)

class TravelItineraryProblem:
    """
    Travel Itinerary Problem definition.
    
    This class defines the problem structure and constraints for optimizing a
    multi-day travel itinerary. It handles all aspects of the problem including
    locations, transport options, time windows, and budget limitations.
    
    The class also provides validation to ensure the problem is well-formed and
    potentially feasible before optimization begins.
    
    Attributes:
        budget (float): Maximum budget in SGD
        NUM_DAYS (int): Number of days for the itinerary
        MAX_ATTRACTION_PER_DAY (int): Maximum attractions to visit per day
        START_TIME (int): Daily start time in minutes from midnight
        HARD_LIMIT_END_TIME (int): Latest possible end time per day
        LUNCH_START, LUNCH_END (int): Lunch time window in minutes
        DINNER_START, DINNER_END (int): Dinner time window in minutes
        locations (list): List of location dictionaries with details
        transport_matrix (dict): Transportation options between locations
        is_feasible (bool): Whether the problem has potential feasible solutions
    """

    def __init__(self, budget, locations, transport_matrix, num_days, config_path="./src/alns_itinerary/config.json"):
        """
        Initialize a travel itinerary optimization problem.
        
        Args:
            budget (float): Maximum budget in SGD for the entire trip
            locations (list): List of location dictionaries containing:
                - name (str): Location name
                - type (str): "attraction", "hawker", or "hotel"
                - entrance_fee (float): Cost for attractions
                - satisfaction (float): Rating for attractions (0-10)
                - rating (float): Rating for hawkers (0-5)
                - duration (int): Time spent at location in minutes
            transport_matrix (dict): Dictionary mapping (origin, destination, hour) tuples
                to dictionaries of transport modes with duration and price
            num_days (int): Number of days for the itinerary
            config_path (str): Path to configuration file with time constraints
        
        Note:
            The first location in the locations list is assumed to be the hotel
            where the trip starts and ends each day.
        """
        config = load_config(config_path)
        # Constants from config
        self.NUM_DAYS = num_days
        self.MAX_ATTRACTION_PER_DAY = config["MAX_ATTRACTION_PER_DAY"]
        self.START_TIME = config["START_TIME"]
        self.HARD_LIMIT_END_TIME = config["HARD_LIMIT_END_TIME"]
        
        # Define lunch and dinner time windows (in minutes since start of day)
        self.LUNCH_START = config["LUNCH_START"]
        self.LUNCH_END = config["LUNCH_END"]
        self.DINNER_START = config["DINNER_START"]
        self.DINNER_END = config["DINNER_END"]
        self.AVG_HAWKER_COST = config["AVG_HAWKER_COST"]
        self.TIME_BRACKETS = [8, 12, 16, 20]
        
        self.budget = budget
        self.locations = locations
        self.num_locations = len(locations)
        
        location_counts = {
            location_type: sum(1 for loc in self.locations if loc["type"] == location_type)
            for location_type in ["attraction", "hawker", "hotel"]
        }
        self.num_attractions = location_counts["attraction"]
        self.num_hawkers = location_counts["hawker"]
        self.num_hotels = location_counts["hotel"]
        
        # Transportation options
        self.transport_types = ["transit", "drive"]
        self.num_transport_types = len(self.transport_types)
        self.transport_matrix = transport_matrix
        
        # Validate the inputs
        self.validate_inputs(budget, locations, transport_matrix, num_days)
        
        # Check if the problem has any feasible solutions
        self.is_feasible = self.test_feasibility()
    
    def validate_location(self, loc):
        """Validate a single location's required fields."""
        missing = []
        required_fields = {
            "attraction": ["entrance_fee", "satisfaction", "duration"],
            "hawker": ["rating", "duration"],
            "hotel": []  # No specific requirements for hotels
        }
        
        if loc["type"] in required_fields:
            for field in required_fields[loc["type"]]:
                if field not in loc or loc[field] is None:
                    missing.append(field)
        
        if missing:
            logger.warning(f"Location '{loc['name']}' is missing required fields: {', '.join(missing)}")
            return False
        return True
    
    def validate_transport_matrix(self, locations, transport_matrix):
        """Validate completeness of transport matrix."""
        required_routes = 0
        missing_routes = []
        
        for i, src in enumerate(locations):
            for j, dest in enumerate(locations):
                if i != j:  # Skip self-routes
                    for hour in self.TIME_BRACKETS:
                        key = (src["name"], dest["name"], hour)
                        required_routes += 1
                        
                        if key not in transport_matrix:
                            missing_routes.append(key)
        
        # Log results
        if missing_routes:
            for i, route in enumerate(missing_routes[:5]):  # Log first 5
                logger.error(f"Missing transport data: {route}")
            
            if len(missing_routes) > 5:
                logger.error(f"... and {len(missing_routes) - 5} more missing routes")
            
            logger.error(f"Missing {len(missing_routes)} of {required_routes} routes in transport matrix")
            return False
        else:
            logger.info(f"Transport matrix complete with all {required_routes} required routes")
            return True
    
    def validate_inputs(self, budget, locations, transport_matrix, num_days):
        """
        Validate input data and log warnings or errors.
        
        Performs comprehensive validation of the problem inputs:
        - Checks for required location types (hotels, hawkers, attractions)
        - Validates completeness of location data
        - Verifies transport matrix coverage
        - Ensures budget is sufficient for basic needs
        
        Args:
            budget (float): Maximum budget
            locations (list): List of location dictionaries
            transport_matrix (dict): Dictionary of transportation options
            num_days (int): Number of days for the itinerary
            
        Note:
            This method logs warnings and errors but does not raise exceptions,
            allowing the optimization to attempt a solution even with imperfect data.
        """
        logger.info("Validating optimization inputs...")
        
        # Check if we have at least one hotel
        hotels = [loc for loc in locations if loc["type"] == "hotel"]
        if not hotels:
            logger.error("No hotels found in locations data!")
        else:
            logger.info(f"Found {len(hotels)} hotels in the data")
        
        # Check if we have hawkers for meals
        hawkers = [loc for loc in locations if loc["type"] == "hawker"]
        if not hawkers:
            logger.error("No hawker centers found - meal constraints cannot be satisfied!")
        else:
            logger.info(f"Found {len(hawkers)} hawker centers in the data")
        
        # Check for attractions
        attractions = [loc for loc in locations if loc["type"] == "attraction"]
        logger.info(f"Found {len(attractions)} attractions in the data")
        
        # Validate location data completeness
        valid_locations = sum(1 for loc in locations if self.validate_location(loc))
        logger.info(f"{valid_locations} of {len(locations)} locations have complete data")
        
        # Check transport matrix completeness
        self.validate_transport_matrix(locations, transport_matrix)
        
        # Check budget feasibility
        min_food_cost = num_days * 2 * self.AVG_HAWKER_COST # 2 meals per day
        
        if budget < min_food_cost:
            logger.error(f"Budget (${budget}) is too low! Minimum needed is ${min_food_cost} for food alone")
        else:
            logger.info(f"Budget check passed: ${budget} >= minimum ${min_food_cost} for food")
    
    def test_feasibility(self):
        """
        Test if the problem has any feasible solutions.
        
        Performs basic feasibility checks including:
        - Availability of required location types
        - Sufficient time windows for activities and meals
        - Temporal constraints compatibility
        
        Returns:
            bool: True if the problem is potentially feasible, False otherwise
            
        Note:
            This is a preliminary check and does not guarantee that an optimal
            solution exists or can be found by the optimizer.
        """
        logger.info("Testing problem feasibility...")
        
        # Check if we have enough hawkers for lunch and dinner every day
        if self.num_hawkers == 0:
            logger.error("Infeasible: No hawker centers available for meals")
            return False
        
        # Check if there's enough time in the day for the minimum itinerary
        available_time = self.HARD_LIMIT_END_TIME - self.START_TIME  # Minutes available
        logger.info(f"Available time per day: {available_time} minutes")
        
        # Check time windows 
        lunch_window = self.LUNCH_END - self.LUNCH_START
        dinner_window = self.DINNER_END - self.DINNER_START
        logger.info(f"Lunch window: {lunch_window} minutes, Dinner window: {dinner_window} minutes")
        
        # Check if we can satisfy hawker constraints 
        if lunch_window < 60:
            logger.error(f"Infeasible: Lunch window ({lunch_window} min) too short for a 60 min meal")
            return False
        
        if dinner_window < 60:
            logger.error(f"Infeasible: Dinner window ({dinner_window} min) too short for a 60 min meal")
            return False
        
        # Check if we have enough hawkers for lunch and dinner
        if self.num_hawkers < 2:
            logger.warning("Only one hawker available - this limits meal options")
        
        return True
    
    def get_transport_hour(self, transport_time):
        """
        Convert a time to the appropriate transport time bracket.
        
        The transport matrix uses discrete time brackets (8, 12, 16, 20)
        to simplify data collection. This method maps any given time to
        the appropriate bracket for lookup.
        
        Args:
            transport_time (int): Time in minutes since start of day (0 = midnight)
                
        Returns:
            int: Transport hour bracket (8, 12, 16, or 20)
            
        Example:
            >>> problem.get_transport_hour(540)  # 9:00 AM
            8
            >>> problem.get_transport_hour(840)  # 2:00 PM
            12
        """
        # Transport_matrix is bracketed to 4 groups, find the earliest applicable one
        transport_hour = transport_time // 60
        
        for bracket in reversed(self.TIME_BRACKETS):
            if transport_hour >= bracket:
                return bracket
        
        return self.TIME_BRACKETS[0]  # Default to first bracket if before 8 AM