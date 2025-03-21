import numpy as np
import logging
import os

logger = logging.getLogger("itinerary_problem")

class TravelItineraryProblem:
    """
    Travel Itinerary Problem definition
    This class defines the problem structure and constraints for a multi-day travel itinerary
    """
    def __init__(self, budget, locations, transport_matrix, num_days=3):
        """
        Initialize the travel itinerary problem
        
        Args:
            budget: Maximum budget in SGD
            locations: List of location dictionaries
            transport_matrix: Dictionary of transportation options between locations
            num_days: Number of days for the itinerary
        """
        # Constants
        self.NUM_DAYS = num_days
        self.HOTEL_COST = 50
        self.START_TIME = 9 * 60  # Every day starts at 9 AM
        self.HARD_LIMIT_END_TIME = 22 * 60  # Return to hotel by 10 PM
        
        # Define lunch and dinner time windows (in minutes since start of day)
        self.LUNCH_START = 11 * 60  # 11 AM
        self.LUNCH_END = 15 * 60    # 3 PM
        self.DINNER_START = 17 * 60  # 5 PM
        self.DINNER_END = 21 * 60   # 9 PM
        
        # Problem parameters
        self.budget = budget
        self.locations = locations
        self.num_locations = len(locations)
        self.num_attractions = len([loc for loc in self.locations if loc["type"] == "attraction"])
        self.num_hawkers = len([loc for loc in self.locations if loc["type"] == "hawker"])
        self.num_hotels = len([loc for loc in self.locations if loc["type"] == "hotel"])
        
        # Transportation options
        self.transport_types = ["transit", "drive"]
        self.num_transport_types = len(self.transport_types)
        self.transport_matrix = transport_matrix
        
        # Validate the inputs
        self.validate_inputs(budget, locations, transport_matrix, num_days)
        
        # Check if the problem has any feasible solutions
        self.is_feasible = self.test_feasibility()
    
    def validate_inputs(self, budget, locations, transport_matrix, num_days):
        """
        Validate input data and print warnings/errors
        
        Args:
            budget: Maximum budget
            locations: List of location dictionaries
            transport_matrix: Dictionary of transportation options
            num_days: Number of days for the itinerary
        """
        logging.info("Validating optimization inputs...")
        
        # Check if we have at least one hotel
        hotels = [loc for loc in locations if loc["type"] == "hotel"]
        if not hotels:
            logging.error("No hotels found in locations data!")
        else:
            logging.info(f"Found {len(hotels)} hotels in the data")
        
        # Check if we have hawkers for meals
        hawkers = [loc for loc in locations if loc["type"] == "hawker"]
        if not hawkers:
            logging.error("No hawker centers found - meal constraints cannot be satisfied!")
        else:
            logging.info(f"Found {len(hawkers)} hawker centers in the data")
        
        # Check for attractions
        attractions = [loc for loc in locations if loc["type"] == "attraction"]
        logging.info(f"Found {len(attractions)} attractions in the data")
        
        # Validate location data completeness
        for i, loc in enumerate(locations):
            missing = []
            if loc["type"] == "attraction":
                if "entrance_fee" not in loc or loc["entrance_fee"] is None:
                    missing.append("entrance_fee")
                if "satisfaction" not in loc or loc["satisfaction"] is None:
                    missing.append("satisfaction")
                if "duration" not in loc or loc["duration"] is None:
                    missing.append("duration")
            elif loc["type"] == "hawker":
                if "rating" not in loc or loc["rating"] is None:
                    missing.append("rating")
                if "duration" not in loc or loc["duration"] is None:
                    missing.append("duration")
            
            if missing:
                logging.warning(f"Location '{loc['name']}' is missing required fields: {', '.join(missing)}")
        
        # Check transport matrix completeness
        sample_routes = 0
        missing_routes = 0
        for i, src in enumerate(locations):
            for j, dest in enumerate(locations):
                if i != j:  # Skip self-routes
                    for hour in [8, 12, 16, 20]:  # Time brackets
                        key = (src["name"], dest["name"], hour)
                        if key not in transport_matrix:
                            missing_routes += 1
                            if missing_routes <= 5:  # Only log the first few missing routes
                                logging.error(f"Missing transport data: {key}")
                        else:
                            sample_routes += 1
        
        if missing_routes > 0:
            logging.error(f"Missing {missing_routes} routes in transport matrix!")
        else:
            logging.info(f"Transport matrix contains all required routes ({sample_routes} total)")
        
        # Check budget feasibility
        hotel_cost = num_days * self.HOTEL_COST
        min_food_cost = num_days * 2 * 10  # Minimum 2 meals per day at $10 each
        
        min_cost = hotel_cost + min_food_cost
        if budget < min_cost:
            logging.error(f"Budget (${budget}) is too low! Minimum needed is ${min_cost} for hotel and food alone")
        else:
            logging.info(f"Budget check passed: ${budget} >= minimum ${min_cost} for hotel and food")
    
    def test_feasibility(self):
        """
        Test if the problem has any feasible solutions
        
        Returns:
            bool: True if the problem is feasible, False otherwise
        """
        logging.info("Testing problem feasibility...")
        
        # Check if we have enough hawkers for lunch and dinner every day
        if self.num_hawkers == 0:
            logging.error("Infeasible: No hawker centers available for meals")
            return False
        
        # Check if there's enough time in the day for the minimum itinerary
        available_time = self.HARD_LIMIT_END_TIME - self.START_TIME  # Minutes available
        logging.info(f"Available time per day: {available_time} minutes")
        
        # Check time windows 
        lunch_window = self.LUNCH_END - self.LUNCH_START
        dinner_window = self.DINNER_END - self.DINNER_START
        logging.info(f"Lunch window: {lunch_window} minutes, Dinner window: {dinner_window} minutes")
        
        # Check if we can satisfy hawker constraints 
        if lunch_window < 60:
            logging.error(f"Infeasible: Lunch window ({lunch_window} min) too short for a 60 min meal")
            return False
        
        if dinner_window < 60:
            logging.error(f"Infeasible: Dinner window ({dinner_window} min) too short for a 60 min meal")
            return False
        
        # Check if we have enough hawkers for lunch and dinner
        if self.num_hawkers < 2:
            logging.warning("Only one hawker available - this limits meal options")
        
        return True
    
    def get_transport_hour(self, transport_time):
        """
        Convert a time (in minutes since day start) to the appropriate transport time bracket
        
        Args:
            transport_time: Time in minutes since day start
            
        Returns:
            int: Transport hour bracket (8, 12, 16, or 20)
        """
        # Transport_matrix is bracketed to 4 groups, find the earliest applicable one
        brackets = [8, 12, 16, 20]
        transport_hour = transport_time // 60
        
        for bracket in reversed(brackets):
            if transport_hour >= bracket:
                return bracket
        
        return brackets[0]  # Default to first bracket if before 8 AM
    
    def evaluate_solution(self, vrp_solution):
        """
        Evaluate a VRPSolution and calculate objectives and constraint violations
        
        Args:
            vrp_solution: VRPSolution instance
            
        Returns:
            dict: Results including objectives and constraint violations
        """
        # Use the VRPSolution's evaluate method which already handles all the checks
        # and calculations we need
        evaluation = vrp_solution.evaluate()
        
        # The VRPSolution.evaluate() method returns a complete evaluation that includes:
        # - is_feasible
        # - total_cost
        # - total_travel_time
        # - total_satisfaction
        # - daily_routes
        # - visited_attractions
        # - constraint_violations (which includes both inequality and equality violations)
        
        return evaluation