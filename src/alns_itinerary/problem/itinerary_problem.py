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
        
        # Solution shape parameters
        # x_ijkl = BINARY VAR, if the route goes in day i, use transport type j, from location k, to location l
        self.x_shape = self.NUM_DAYS * self.num_transport_types * self.num_locations * self.num_locations
        # u_ik  = CONTINUOUS VAR, tracking FINISH time of the person in day i, location k
        self.u_shape = self.NUM_DAYS * self.num_locations
        
        # Total solution vector length
        self.n_var = self.x_shape + self.u_shape
        
        # Bounds for the variables
        self.xl = np.concatenate([np.zeros(self.x_shape), np.full(self.u_shape, 0)])
        self.xu = np.concatenate([np.ones(self.x_shape), np.full(self.u_shape, self.HARD_LIMIT_END_TIME)])
        
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
    
    def evaluate_solution(self, solution):
        """
        Evaluate a solution vector and calculate objectives and constraint violations
        
        Args:
            solution: Solution vector (x_var and u_var concatenated)
            
        Returns:
            dict: Results including objectives and constraint violations
        """
        # Reshape solution vector into decision variables (x_var) and time variables (u_var)
        x_var = solution[:self.x_shape].reshape(self.NUM_DAYS, self.num_transport_types, 
                                             self.num_locations, self.num_locations)
        u_var = solution[self.x_shape:].reshape(self.NUM_DAYS, self.num_locations)
        
        # Initialize results
        results = {
            "is_feasible": True,
            "inequality_violations": [],
            "equality_violations": [],
            "total_cost": 0,
            "total_travel_time": 0,
            "total_satisfaction": 0,
            "visited_attractions": [],
            "daily_routes": []
        }
        
        # Start with hotel costs
        total_cost = self.NUM_DAYS * self.HOTEL_COST
        total_travel_time = 0
        total_satisfaction = 0
        
        # Check all constraints and calculate objectives
        
        # 1. For attractions, visit at most once (both as source and destination)
        for k in range(self.num_locations):
            if self.locations[k]["type"] == "attraction":
                visits_as_source = np.sum(x_var[:, :, k, :])
                visits_as_dest = np.sum(x_var[:, :, :, k])
                
                if visits_as_source > 1:
                    results["is_feasible"] = False
                    results["inequality_violations"].append({
                        "type": "attraction_max_once_source",
                        "location": k,
                        "name": self.locations[k]["name"],
                        "value": float(visits_as_source),
                        "limit": 1
                    })
                
                if visits_as_dest > 1:
                    results["is_feasible"] = False
                    results["inequality_violations"].append({
                        "type": "attraction_max_once_dest",
                        "location": k,
                        "name": self.locations[k]["name"],
                        "value": float(visits_as_dest),
                        "limit": 1
                    })
                
                # If an attraction is visited, record it
                if visits_as_dest > 0:
                    results["visited_attractions"].append(self.locations[k]["name"])
        
        # 2. For hawkers, visit at most once per day
        for day in range(self.NUM_DAYS):
            for k in range(self.num_locations):
                if self.locations[k]["type"] == "hawker":
                    daily_visits_source = np.sum(x_var[day, :, k, :])
                    daily_visits_dest = np.sum(x_var[day, :, :, k])
                    
                    if daily_visits_source > 1:
                        results["is_feasible"] = False
                        results["inequality_violations"].append({
                            "type": "hawker_max_once_daily_source",
                            "day": day,
                            "location": k,
                            "name": self.locations[k]["name"],
                            "value": float(daily_visits_source),
                            "limit": 1
                        })
                    
                    if daily_visits_dest > 1:
                        results["is_feasible"] = False
                        results["inequality_violations"].append({
                            "type": "hawker_max_once_daily_dest",
                            "day": day,
                            "location": k,
                            "name": self.locations[k]["name"],
                            "value": float(daily_visits_dest),
                            "limit": 1
                        })
        
        # 3. Check start time constraint (must start from hotel at START_TIME)
        for day in range(self.NUM_DAYS):
            if u_var[day, 0] < self.START_TIME:
                results["is_feasible"] = False
                results["inequality_violations"].append({
                    "type": "start_time_before_minimum",
                    "day": day,
                    "value": float(u_var[day, 0]),
                    "limit": self.START_TIME
                })
        
        # 4. Time constraints and calculation of costs/travel time
        for day in range(self.NUM_DAYS):
            for j in range(self.num_transport_types):
                for k in range(self.num_locations):
                    for l in range(self.num_locations):
                        if k == l:
                            continue
                        
                        # Skip if route not chosen
                        if x_var[day, j, k, l] == 0:
                            continue
                        
                        # Check time constraints
                        time_finish_l = u_var[day, l]
                        transport_hour = self.get_transport_hour(u_var[day, k])
                        
                        try:
                            transport_key = (self.locations[k]["name"], self.locations[l]["name"], transport_hour)
                            transport_value = self.transport_matrix[transport_key][self.transport_types[j]]
                            
                            # Calculate time to finish at destination including duration at destination
                            time_should_finish_l = u_var[day, k] + transport_value["duration"] + self.locations[l]["duration"]
                            
                            if time_finish_l < time_should_finish_l:
                                results["is_feasible"] = False
                                results["inequality_violations"].append({
                                    "type": "time_constraint_violated",
                                    "day": day,
                                    "transport": j,
                                    "from": k,
                                    "to": l,
                                    "from_name": self.locations[k]["name"],
                                    "to_name": self.locations[l]["name"],
                                    "actual_finish": float(time_finish_l),
                                    "required_finish": float(time_should_finish_l)
                                })
                            
                            # Accumulate travel time and costs
                            total_travel_time += transport_value["duration"]
                            total_cost += transport_value["price"]
                            
                            # Add location costs and satisfaction
                            if self.locations[l]["type"] == "attraction":
                                total_cost += self.locations[l]["entrance_fee"]
                                total_satisfaction += self.locations[l]["satisfaction"]
                            elif self.locations[l]["type"] == "hawker":
                                total_cost += 10  # Assumed meal cost
                                total_satisfaction += self.locations[l]["rating"]
                                
                        except KeyError:
                            results["inequality_violations"].append({
                                "type": "missing_transport_data",
                                "day": day,
                                "transport": j,
                                "from": k,
                                "to": l,
                                "from_name": self.locations[k]["name"],
                                "to_name": self.locations[l]["name"],
                                "hour": transport_hour
                            })
        
        # 5. Check transport type constraints (can't use multiple transport types for same route)
        for day in range(self.NUM_DAYS):
            for k in range(self.num_locations):
                for l in range(self.num_locations):
                    transport_sum = np.sum(x_var[day, :, k, l])
                    
                    if transport_sum > 1:
                        results["is_feasible"] = False
                        results["inequality_violations"].append({
                            "type": "multiple_transport_types",
                            "day": day,
                            "from": k,
                            "to": l,
                            "from_name": self.locations[k]["name"],
                            "to_name": self.locations[l]["name"],
                            "value": float(transport_sum),
                            "limit": 1
                        })
        
        # 6. Check lunch and dinner constraints
        for day in range(self.NUM_DAYS):
            lunch_visits = 0
            dinner_visits = 0
            hawker_sum = 0
            
            for k in range(self.num_locations):
                if self.locations[k]["type"] == "hawker":
                    hawker_sum += np.sum(x_var[day, :, :, k])  # Count visits as destination
                    
                    # Check if this hawker visit is during lunch or dinner
                    for src in range(self.num_locations):
                        if src == k:
                            continue
                        
                        for j_transport in range(self.num_transport_types):
                            if x_var[day, j_transport, src, k] == 1:
                                arrival_time = u_var[day, k]
                                if arrival_time >= self.LUNCH_START and arrival_time <= self.LUNCH_END:
                                    lunch_visits += 1
                                if arrival_time >= self.DINNER_START and arrival_time <= self.DINNER_END:
                                    dinner_visits += 1
            
            # Every day, must go to hawkers at least twice
            if hawker_sum < 2:
                results["is_feasible"] = False
                results["inequality_violations"].append({
                    "type": "insufficient_hawker_visits",
                    "day": day,
                    "value": float(hawker_sum),
                    "limit": 2
                })
            
            # Must visit a hawker during lunch time
            if lunch_visits < 1:
                results["is_feasible"] = False
                results["inequality_violations"].append({
                    "type": "no_lunch_hawker_visit",
                    "day": day,
                    "value": lunch_visits,
                    "limit": 1
                })
            
            # Must visit a hawker during dinner time
            if dinner_visits < 1:
                results["is_feasible"] = False
                results["inequality_violations"].append({
                    "type": "no_dinner_hawker_visit",
                    "day": day,
                    "value": dinner_visits,
                    "limit": 1
                })
        
        # 7. Check budget constraint
        if total_cost > self.budget:
            results["is_feasible"] = False
            results["inequality_violations"].append({
                "type": "budget_exceeded",
                "value": float(total_cost),
                "limit": self.budget
            })
        
        # 8. Check minimum and maximum visits
        min_visits = self.NUM_DAYS * 2  # Minimum 2 visits per day (lunch & dinner)
        max_visits = self.NUM_DAYS * 6  # Maximum reasonable visits per day
        total_visits = np.sum(x_var)
        
        if total_visits < min_visits:
            results["is_feasible"] = False
            results["inequality_violations"].append({
                "type": "insufficient_total_visits",
                "value": float(total_visits),
                "limit": min_visits
            })
        
        if total_visits > max_visits:
            results["is_feasible"] = False
            results["inequality_violations"].append({
                "type": "excessive_total_visits",
                "value": float(total_visits),
                "limit": max_visits
            })
        
        # 9. For each day, hotel must be the starting point
        for day in range(self.NUM_DAYS):
            # Hotel index is 0
            hotel_outgoing = np.sum(x_var[day, :, 0, :])
            if hotel_outgoing != 1:
                results["is_feasible"] = False
                results["equality_violations"].append({
                    "type": "hotel_not_starting_point",
                    "day": day,
                    "value": float(hotel_outgoing),
                    "required": 1
                })
        
        # 10. For each day, must return to hotel
        for day in range(self.NUM_DAYS):
            # Find last location
            last_time = np.max(u_var[day, :])
            last_locations = np.where(u_var[day, :] == last_time)[0]
            
            # If last location is not hotel (0), check if there's a route back to hotel
            if len(last_locations) > 0 and 0 not in last_locations:
                last_loc = last_locations[0]
                returns_to_hotel = np.sum(x_var[day, :, last_loc, 0])
                
                if returns_to_hotel != 1:
                    results["is_feasible"] = False
                    results["equality_violations"].append({
                        "type": "not_returning_to_hotel",
                        "day": day,
                        "last_location": last_loc,
                        "last_location_name": self.locations[last_loc]["name"],
                        "value": float(returns_to_hotel),
                        "required": 1
                    })
        
        # 11. Flow conservation (in = out) for each location
        for day in range(self.NUM_DAYS):
            for k in range(self.num_locations):
                incoming = np.sum(x_var[day, :, :, k])
                outgoing = np.sum(x_var[day, :, k, :])
                
                if incoming != outgoing:
                    results["is_feasible"] = False
                    results["equality_violations"].append({
                        "type": "flow_conservation_violated",
                        "day": day,
                        "location": k,
                        "location_name": self.locations[k]["name"],
                        "incoming": float(incoming),
                        "outgoing": float(outgoing)
                    })
        
        # 12. If an attraction is visited as source, it must be visited as destination
        for k in range(self.num_locations):
            if self.locations[k]["type"] == "attraction":
                as_source = np.sum(x_var[:, :, k, :])
                as_dest = np.sum(x_var[:, :, :, k])
                
                if as_source != as_dest:
                    results["is_feasible"] = False
                    results["equality_violations"].append({
                        "type": "attraction_source_dest_mismatch",
                        "location": k,
                        "name": self.locations[k]["name"],
                        "as_source": float(as_source),
                        "as_dest": float(as_dest)
                    })
        
        # Store objectives in results
        results["total_cost"] = float(total_cost)
        results["total_travel_time"] = float(total_travel_time)
        results["total_satisfaction"] = float(total_satisfaction)
        
        # Generate daily routes for the solution
        for day in range(self.NUM_DAYS):
            route = self.trace_daily_route(x_var, u_var, day)
            results["daily_routes"].append(route)
        
        return results
    
    def trace_daily_route(self, x_var, u_var, day):
        """
        Trace the route for a specific day, avoiding cycles and repetition
        
        Args:
            x_var: The binary decision variables (4D array)
            u_var: The time variables (2D array)
            day: The day to trace
            
        Returns:
            list: Ordered sequence of locations visited
        """
        route = []
        visited = set()  # Track visited locations to avoid cycles
        current_loc = 0  # Start at hotel (assumed to be index 0)
        
        # Add starting point
        route.append({
            "location": current_loc,
            "name": self.locations[current_loc]["name"],
            "type": self.locations[current_loc]["type"],
            "time": float(u_var[day, current_loc]),
            "transport_from_prev": None
        })
        visited.add(current_loc)
        
        # Follow the route based on time ordering
        while True:
            next_loc = None
            next_transport = None
            min_next_time = float('inf')
            
            # Find the next location in time sequence
            for j in range(self.num_transport_types):
                for l in range(self.num_locations):
                    # Check if there's a route from current location to location l
                    if x_var[day, j, current_loc, l] > 0 and l not in visited:
                        # Check if this is the earliest next location
                        if u_var[day, l] < min_next_time:
                            min_next_time = u_var[day, l]
                            next_loc = l
                            next_transport = j
            
            # If no next location found, we've completed the route
            if next_loc is None:
                break
            
            # Add next location to route
            route.append({
                "location": next_loc,
                "name": self.locations[next_loc]["name"],
                "type": self.locations[next_loc]["type"],
                "time": float(u_var[day, next_loc]),
                "transport_from_prev": self.transport_types[next_transport]
            })
            
            # Mark location as visited and move to it
            visited.add(next_loc)
            current_loc = next_loc
            
            # Safety check to prevent infinite loops
            if len(route) > self.num_locations:
                break
        
        return route