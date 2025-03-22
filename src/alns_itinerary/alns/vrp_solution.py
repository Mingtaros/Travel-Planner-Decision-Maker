"""
Solution representation for the VRP-based travel itinerary problem.
This file defines the core data structure for representing solutions.
"""

import copy
import numpy as np
import logging

logger = logging.getLogger(__name__)

class VRPSolution:
    """
    Position-based solution representation for the Travel Itinerary Problem.
    Each solution consists of routes for each day, where each route is a sequence
    of locations with arrival/departure times and transport modes.
    """
    def __init__(self, problem):
        """
        Initialize a VRP solution with empty routes.
        
        Args:
            problem: The TravelItineraryProblem instance
        """
        self.problem = problem
        self.num_days = problem.NUM_DAYS
        
        # For each day, store a sequence of locations and their visit times
        # Format: (location_idx, arrival_time, departure_time, transport_mode)
        self.routes = [[] for _ in range(self.num_days)]
        
        # Initialize all days to start at hotel
        hotel_idx = 0  # Assuming hotel is at index 0
        for day in range(self.num_days):
            # Add hotel as first location (no transport for first location)
            self.routes[day].append((hotel_idx, problem.START_TIME, problem.START_TIME, None))
    
    def clone(self):
        """Create a deep copy of this solution"""
        new_solution = VRPSolution(self.problem)
        new_solution.routes = copy.deepcopy(self.routes)
        return new_solution
    
    def get_cost_duration(self, day, location_idx, position, transport_mode="transit"):
        """
        Calculate the cost and duration of travel to a new location.
        
        Args:
            location_idx: Index of the new location
            prev_loc: Index of the previous location
            prev_departure: Departure time from the previous location
            transport_mode: Transport mode ("transit" or "drive")
            
        Returns:
            tuple: (cost, duration) of travel
        """
        route = self.routes[day]
        # Get previous location and departure time
        if position == 0:
            prev_loc, prev_departure = 0, self.problem.START_TIME
        else:
            prev_loc, _, prev_departure, _ = route[position-1]
        # Calculate travel time
        transport_hour = self.problem.get_transport_hour(prev_departure)
        try:
            transport_key = (self.problem.locations[prev_loc]["name"], 
                           self.problem.locations[location_idx]["name"], 
                           transport_hour)
            transport_data = self.problem.transport_matrix[transport_key][transport_mode]
            
            loc_cost = 0
            if self.problem.locations[location_idx]['type'] == 'attraction':
                loc_cost = self.problem.locations[location_idx].get('entrance_fee', 0)
            elif self.problem.locations[location_idx]['type'] == 'hawker':
                loc_cost = self.problem.locations[location_idx].get('avg_food_price', 0)
            else:
                loc_cost = 0
                
            return transport_data["price"] + loc_cost, transport_data["duration"] + self.problem.locations[location_idx].get('duration', 60)
        except KeyError:
            # Missing transport data, use defaults
            logger.warning(f"Missing transport data for {self.problem.locations[location_idx]}")
            logger.warning(f"Previous Location: {self.problem.locations[prev_loc]}")
    
    def insert_location(self, day, position, location_idx, transport_mode="transit", meal=None):
        """
        Insert a location into a route at the specified position.
        Does not check feasibility - use is_feasible_insertion for that.
        
        Args:
            day: Day index
            position: Position to insert (1 = after hotel, etc.)
            location_idx: Location index to insert
            transport_mode: Transport mode ("transit" or "drive")
            
        Returns:
            bool: True if insert was successful
        """
        route = self.routes[day]
        
        # Ensure position is valid
        if position < 1 or position > len(route):
            logger.warning(f"Invalid position {position} for insertion")
            return False
        
        # Get preceding location
        prev_loc, _, prev_departure, _ = route[position-1]
        
        # Calculate travel time to new location
        transport_hour = self.problem.get_transport_hour(prev_departure)
        try:
            transport_key = (self.problem.locations[prev_loc]["name"], 
                             self.problem.locations[location_idx]["name"], 
                             transport_hour)
            transport_data = self.problem.transport_matrix[transport_key][transport_mode]
            
            # Calculate arrival time
            arrival_time = prev_departure + transport_data["duration"]
            
            # Special handling for hawkers (enforce meal windows)
            if self.problem.locations[location_idx]["type"] == "hawker":
                # Check if it's lunch or dinner time
                if meal == 'Lunch':
                    # Lunch visit - ensure it's within lunch window
                    arrival_time = max(arrival_time, self.problem.LUNCH_START)
                elif meal == 'Dinner':
                    # Dinner visit - ensure it's within dinner window
                    arrival_time = max(arrival_time, self.problem.DINNER_START)
            
            # Calculate departure time
            location_duration = self.problem.locations[location_idx]["duration"]
            departure_time = arrival_time + location_duration
            
            # Insert the new location
            route.insert(position, (location_idx, arrival_time, departure_time, transport_mode))
            
            return True, departure_time
            
        except KeyError:
            # Missing transport data
            return False
    
    def remove_location(self, day, position):
        """
        Remove a location from a route.
        
        Args:
            day: Day index
            position: Position to remove (0 = hotel start, etc.)
            
        Returns:
            bool: True if removal was successful
        """
        route = self.routes[day]
        
        # Don't remove hotel start position
        if position == 0:
            return False
        
        # Ensure position is valid
        if position >= len(route):
            return False
        
        # Remove the location
        route.pop(position)
        
        # Recalculate times for the rest of the route
        self.recalculate_route_times(day, position)
        
        return True
    
    def recalculate_route_times(self, day, start_position=1):
        """
        Recalculate arrival and departure times for a route from start_position onward.
        
        Args:
            day: Day index
            start_position: Position to start recalculation from
            
        Returns:
            bool: True if recalculation was successful
        """
        route = self.routes[day]
        
        # Ensure start position is valid
        if start_position >= len(route):
            return True  # Nothing to recalculate
        
        # Process each location in sequence
        for i in range(start_position, len(route)):
            prev_loc, _, prev_departure, _ = route[i-1]
            curr_loc, _, _, transport_mode = route[i]
            
            # Calculate travel time
            transport_hour = self.problem.get_transport_hour(prev_departure)
            try:
                transport_key = (self.problem.locations[prev_loc]["name"], 
                               self.problem.locations[curr_loc]["name"], 
                               transport_hour)
                transport_data = self.problem.transport_matrix[transport_key][transport_mode]
                
                # Calculate arrival time
                arrival_time = prev_departure + transport_data["duration"]
                
                # Special handling for hawkers (enforce meal windows)
                if self.problem.locations[curr_loc]["type"] == "hawker":
                    # Check if it's lunch or dinner time
                    if arrival_time < self.problem.LUNCH_END and arrival_time > self.problem.LUNCH_START - 60:
                        # Lunch visit - ensure it's within lunch window
                        arrival_time = max(arrival_time, self.problem.LUNCH_START)
                    elif arrival_time < self.problem.DINNER_END and arrival_time > self.problem.DINNER_START - 60:
                        # Dinner visit - ensure it's within dinner window
                        arrival_time = max(arrival_time, self.problem.DINNER_START)
                
                # Calculate departure time
                location_duration = self.problem.locations[curr_loc]["duration"]
                departure_time = arrival_time + location_duration
                
                # Update the route with new times
                route[i] = (curr_loc, arrival_time, departure_time, transport_mode)
                
            except KeyError:
                # Missing transport data, use defaults
                arrival_time = prev_departure + 30  # Default 30 min travel
                location_duration = self.problem.locations[curr_loc]["duration"]
                departure_time = arrival_time + location_duration
                route[i] = (curr_loc, arrival_time, departure_time, transport_mode)
        
        return True

    def is_feasible_insertion(self, day, position, location_idx, transport_mode="transit"):
        """
        Enhanced check if inserting a location at a position is feasible.
        Includes additional checks to prevent duplicate visits and unrealistic itineraries.
        With improved dinner scheduling flexibility.
        
        Args:
            day: Day index
            position: Position to insert
            location_idx: Location index to insert
            transport_mode: Transport mode
            
        Returns:
            bool: True if insertion is feasible
        """
        route = self.routes[day]
        
        # Ensure position is valid
        if position < 1 or position > len(route):
            return False
        
        # Location-type specific checks
        location_type = self.problem.locations[location_idx]["type"]
        
        # Check uniqueness constraints based on location type
        if location_type == "hotel":
            # Prevent unnecessary hotel visits during the day
            # Only allow hotel at start and end positions
            if position > 1 and position < len(route):
                return False
            
            # Check if we already have a hotel at this position
            if position < len(route) and self.problem.locations[route[position][0]]["type"] == "hotel":
                return False
        
        elif location_type == "attraction":
            # Check if this attraction is already in any route (attractions can only be visited once)
            for d in range(self.num_days):
                for loc, _, _, _ in self.routes[d]:
                    if loc == location_idx:
                        return False
        
        elif location_type == "hawker":
            # Check if this hawker has already been visited today
            # This prevents multiple visits to the same hawker in one day
            for loc, _, _, _ in route:
                if loc == location_idx:
                    return False
            
            # Get preceding location
            prev_loc, _, prev_departure, _ = route[position-1]
            
            # Calculate arrival time
            transport_hour = self.problem.get_transport_hour(prev_departure)
            try:
                transport_key = (self.problem.locations[prev_loc]["name"], 
                            self.problem.locations[location_idx]["name"], 
                            transport_hour)
                transport_data = self.problem.transport_matrix[transport_key][transport_mode]
                arrival_time = prev_departure + transport_data["duration"]
                
                # Check if this would be a lunch or dinner visit
                is_lunch = (arrival_time >= self.problem.LUNCH_START and arrival_time <= self.problem.LUNCH_END)
                is_dinner = (arrival_time >= self.problem.DINNER_START and arrival_time <= self.problem.DINNER_END)
                
                # Add more flexible dinner window check
                # Allow dinner visits that are slightly outside the official window
                is_dinner_flexible = (arrival_time >= self.problem.DINNER_START - 30 and 
                                    arrival_time <= self.problem.DINNER_END + 30)
                
                # Check if we already have a meal of this type today
                has_lunch = False
                has_dinner = False
                
                for loc, arr_time, _, _ in route:
                    if self.problem.locations[loc]["type"] == "hawker":
                        if arr_time >= self.problem.LUNCH_START and arr_time <= self.problem.LUNCH_END:
                            has_lunch = True
                        elif arr_time >= self.problem.DINNER_START and arr_time <= self.problem.DINNER_END:
                            has_dinner = True
                
                # Prevent multiple lunch visits
                if is_lunch and has_lunch:
                    return False
                
                # Prevent multiple dinner visits, but with flexibility
                if (is_dinner or is_dinner_flexible) and has_dinner:
                    return False
                
                # If it's neither lunch nor dinner, check if it's reasonably close to a meal time
                if not (is_lunch or is_dinner or is_dinner_flexible):
                    # Only allow hawker visits within extended windows for meal times
                    close_to_lunch = (arrival_time >= self.problem.LUNCH_START - 30 and arrival_time <= self.problem.LUNCH_END + 30)
                    close_to_dinner = (arrival_time >= self.problem.DINNER_START - 45 and arrival_time <= self.problem.DINNER_END + 45)
                    
                    if not (close_to_lunch or close_to_dinner):
                        return False
                        
                # Priority boost for dinner! If we don't have dinner yet, be more lenient
                if not has_dinner and (is_dinner or is_dinner_flexible):
                    # We really want to add dinner, so give it special treatment
                    # This is a dinner insertion, we should prioritize it!
                    pass  # Allow it to continue regardless of other conditions
                    
            except KeyError:
                # Missing transport data
                return False
        
        # Check if we're about to insert a location right after the same location
        # (prevents immediate return to the same place)
        if position > 1:
            prev_loc = route[position-1][0]
            if prev_loc == location_idx:
                return False
        
        # Check if we're about to create a pattern like A -> B -> A
        # (prevents unnecessary back-and-forth)
        if position > 2:
            loc_two_back = route[position-2][0]
            if loc_two_back == location_idx:
                return False
        
        # Check if inserting this location would exceed the maximum visits per day
        # (prevents overcrowded days)
        attraction_count = 0
        hawker_count = 0
        
        for loc, _, _, _ in route:
            if self.problem.locations[loc]["type"] == "attraction":
                attraction_count += 1
            elif self.problem.locations[loc]["type"] == "hawker":
                hawker_count += 1
        
        # Add the location we're trying to insert
        if location_type == "attraction":
            attraction_count += 1
        elif location_type == "hawker":
            hawker_count += 1
        
        # Check against maximum allowed visits - relaxed for hawkers to ensure dinner can be added
        MAX_ATTRACTIONS_PER_DAY = 4  # Reasonable limit
        MAX_HAWKERS_PER_DAY = 3      # Reasonable limit (lunch, snack, dinner)
        
        if attraction_count > MAX_ATTRACTIONS_PER_DAY:
            return False
        
        # Be more permissive with hawker count if we don't have dinner yet
        has_lunch, has_dinner = self.has_lunch_and_dinner(day)
        if hawker_count > MAX_HAWKERS_PER_DAY and (has_lunch and has_dinner):
            return False
        
        # Now perform the original feasibility checks
        # Get preceding location
        prev_loc, _, prev_departure, _ = route[position-1]
        
        # Calculate travel time
        transport_hour = self.problem.get_transport_hour(prev_departure)
        try:
            transport_key = (self.problem.locations[prev_loc]["name"], 
                        self.problem.locations[location_idx]["name"], 
                        transport_hour)
            transport_data = self.problem.transport_matrix[transport_key][transport_mode]
            
            # Calculate arrival time
            arrival_time = prev_departure + transport_data["duration"]
            
            # Special handling for hawkers (enforce meal windows)
            if self.problem.locations[location_idx]["type"] == "hawker":
                # Check if it's lunch or dinner time
                is_lunch = (arrival_time >= self.problem.LUNCH_START and arrival_time <= self.problem.LUNCH_END)
                is_dinner = (arrival_time >= self.problem.DINNER_START and arrival_time <= self.problem.DINNER_END)
                
                # Add more flexible dinner window check
                is_dinner_flexible = (arrival_time >= self.problem.DINNER_START - 30 and 
                                arrival_time <= self.problem.DINNER_END + 30)
                
                if not (is_lunch or is_dinner or is_dinner_flexible):
                    # Only allow hawker visits within 30 minutes of meal windows
                    close_to_lunch = (arrival_time >= self.problem.LUNCH_START - 30 and arrival_time <= self.problem.LUNCH_END + 30)
                    close_to_dinner = (arrival_time >= self.problem.DINNER_START - 45 and arrival_time <= self.problem.DINNER_END + 45)
                    
                    if not (close_to_lunch or close_to_dinner):
                        return False
            
            # Calculate departure time
            location_duration = self.problem.locations[location_idx]["duration"]
            departure_time = arrival_time + location_duration
            
            # Check if we return to hotel too late
            hotel_idx = 0
            if position == len(route) and location_idx != hotel_idx:
                # Need to calculate return to hotel
                transport_hour = self.problem.get_transport_hour(departure_time)
                try:
                    transport_key = (self.problem.locations[location_idx]["name"], 
                                self.problem.locations[hotel_idx]["name"], 
                                transport_hour)
                    transport_data = self.problem.transport_matrix[transport_key][transport_mode]
                    return_time = departure_time + transport_data["duration"]
                    
                    # Be more flexible with return time if adding a dinner hawker
                    if self.problem.locations[location_idx]["type"] == "hawker" and not has_dinner:
                        # Allow slightly later returns for dinner
                        if return_time > self.problem.HARD_LIMIT_END_TIME + 30:
                            return False
                    else:
                        if return_time > self.problem.HARD_LIMIT_END_TIME:
                            return False
                except KeyError:
                    # Missing transport data
                    return False
            
            # Check if next location can still be reached on time
            if position < len(route):
                next_loc, next_arrival, _, next_transport = route[position]
                
                # Calculate time to next location
                transport_hour = self.problem.get_transport_hour(departure_time)
                try:
                    transport_key = (self.problem.locations[location_idx]["name"], 
                                self.problem.locations[next_loc]["name"], 
                                transport_hour)
                    transport_data = self.problem.transport_matrix[transport_key][next_transport]
                    
                    new_next_arrival = departure_time + transport_data["duration"]
                    
                    # If the next location is a hawker, check time windows
                    if self.problem.locations[next_loc]["type"] == "hawker":
                        # Current arrival time
                        if next_arrival >= self.problem.LUNCH_START and next_arrival <= self.problem.LUNCH_END:
                            # It's a lunch visit, ensure we're still in lunch window
                            if new_next_arrival > self.problem.LUNCH_END:
                                return False
                        elif next_arrival >= self.problem.DINNER_START and next_arrival <= self.problem.DINNER_END:
                            # It's a dinner visit, ensure we're still in dinner window
                            if new_next_arrival > self.problem.DINNER_END:
                                return False
                
                except KeyError:
                    # Missing transport data
                    return False
            
            return True
            
        except KeyError:
            # Missing transport data
            return False
    
    def has_lunch_and_dinner(self, day):
        """
        Check if a day's route includes both lunch and dinner visits.
        
        Args:
            day: Day index
            
        Returns:
            tuple: (has_lunch, has_dinner)
        """
        route = self.routes[day]
        has_lunch = False
        has_dinner = False
        
        for loc, arrival_time, _, _ in route:
            if self.problem.locations[loc]["type"] == "hawker":
                if arrival_time >= self.problem.LUNCH_START and arrival_time <= self.problem.LUNCH_END:
                    has_lunch = True
                elif arrival_time >= self.problem.DINNER_START and arrival_time <= self.problem.DINNER_END:
                    has_dinner = True
        
        return has_lunch, has_dinner
    
    def get_visited_attractions(self):
        """
        Get all attraction indices that are visited in this solution.
        
        Returns:
            set: Set of visited attraction indices
        """
        visited = set()
        
        for day in range(self.num_days):
            for loc, _, _, _ in self.routes[day]:
                if self.problem.locations[loc]["type"] == "attraction":
                    visited.add(loc)
        
        return visited

    def is_feasible(self):
        """
        Enhanced check to determine if the solution is feasible.
        Added checks for practical considerations like duplicate locations
        and proper meal scheduling.
        
        Returns:
            bool: True if solution is feasible
        """
        # Check that each day has a hotel start and end
        for day in range(self.num_days):
            route = self.routes[day]
            
            # Check for empty routes
            if len(route) < 1:
                return False
            
            # Check start at hotel
            first_loc, first_time, _, _ = route[0]
            if first_loc != 0 or first_time < self.problem.START_TIME:
                return False
            
            # Check if the day ends at hotel or can return to hotel in time
            last_loc, _, last_departure, _ = route[-1]
            if last_loc != 0:
                # Calculate return to hotel
                hotel_idx = 0
                transport_hour = self.problem.get_transport_hour(last_departure)
                try:
                    transport_key = (self.problem.locations[last_loc]["name"], 
                                self.problem.locations[hotel_idx]["name"], 
                                transport_hour)
                    # Try both transit and drive
                    if "transit" in self.problem.transport_matrix.get(transport_key, {}):
                        transport_data = self.problem.transport_matrix[transport_key]["transit"]
                    else:
                        transport_data = self.problem.transport_matrix[transport_key]["drive"]
                    
                    return_time = last_departure + transport_data["duration"]
                    if return_time > self.problem.HARD_LIMIT_END_TIME:
                        return False
                except KeyError:
                    # Missing transport data
                    return False
            
            # Check for duplicate locations within each day
            locations_visited = {}
            for i, (loc_idx, arrival_time, _, _) in enumerate(route):
                loc_type = self.problem.locations[loc_idx]["type"]
                
                # Hotel should only appear at start and end
                if loc_type == "hotel" and i > 0 and i < len(route) - 1:
                    # Hotel appears in the middle of the day (not allowed)
                    return False
                
                # Each attraction should appear at most once across all days
                if loc_type == "attraction":
                    if loc_idx in locations_visited:
                        # Same attraction appears more than once in a day
                        return False
                    locations_visited[loc_idx] = True
            
            # Check for proper meal scheduling
            lunch_visits = 0
            dinner_visits = 0
            
            for loc_idx, arrival_time, _, _ in route:
                if self.problem.locations[loc_idx]["type"] == "hawker":
                    # Check if it's a lunch or dinner visit
                    if arrival_time >= self.problem.LUNCH_START and arrival_time <= self.problem.LUNCH_END:
                        lunch_visits += 1
                    elif arrival_time >= self.problem.DINNER_START and arrival_time <= self.problem.DINNER_END:
                        dinner_visits += 1
            
            # Must have exactly one lunch and one dinner
            if lunch_visits != 1 or dinner_visits != 1:
                return False
            
            # Check for reasonable number of attractions per day
            attraction_count = sum(1 for loc, _, _, _ in route if self.problem.locations[loc]["type"] == "attraction")
            if attraction_count > 4:  # More than 4 attractions in a day is unrealistic
                return False
        
        # Check that attractions are visited at most once across all days
        attraction_visits = {}
        for day in range(self.num_days):
            for loc, _, _, _ in self.routes[day]:
                if self.problem.locations[loc]["type"] == "attraction":
                    if loc in attraction_visits:
                        return False
                    attraction_visits[loc] = True
        
        # Check for logical sequencing - no A->B->A patterns (unnecessary backtracking)
        for day in range(self.num_days):
            route = self.routes[day]
            for i in range(2, len(route)):
                if route[i][0] == route[i-2][0] and route[i][0] != 0:  # Allow hotel to repeat
                    # Found an A->B->A pattern
                    return False
        
        # Check for budget constraint
        total_cost = self.get_total_cost()
        if total_cost > self.problem.budget:
            return False
        
        return True
    
    def get_total_cost(self):
        """
        Calculate the total cost of the solution.
        
        Returns:
            float: Total cost in SGD
        """
        total_cost = self.problem.NUM_DAYS * self.problem.HOTEL_COST  # Hotel cost
        
        # Add costs for each day's route
        for day in range(self.num_days):
            route = self.routes[day]
            
            # Process each location in sequence
            for i in range(1, len(route)):
                prev_loc, _, prev_departure, _ = route[i-1]
                curr_loc, _, _, transport_mode = route[i]
                
                # Calculate transport cost
                transport_hour = self.problem.get_transport_hour(prev_departure)
                try:
                    transport_key = (self.problem.locations[prev_loc]["name"], 
                                   self.problem.locations[curr_loc]["name"], 
                                   transport_hour)
                    transport_data = self.problem.transport_matrix[transport_key][transport_mode]
                    total_cost += transport_data["price"]
                except KeyError:
                    # Missing transport data, use default
                    total_cost += 5  # Default cost
                
                # Add location costs
                loc_type = self.problem.locations[curr_loc]["type"]
                if loc_type == "attraction":
                    total_cost += self.problem.locations[curr_loc]["entrance_fee"]
                elif loc_type == "hawker":
                    total_cost += 10  # Standard meal cost
        
        return total_cost
    
    def get_total_travel_time(self):
        """
        Calculate the total travel time of the solution.
        
        Returns:
            float: Total travel time in minutes
        """
        total_travel_time = 0
        
        # Add travel times for each day's route
        for day in range(self.num_days):
            route = self.routes[day]
            
            # Process each location in sequence
            for i in range(1, len(route)):
                prev_loc, _, prev_departure, _ = route[i-1]
                curr_loc, curr_arrival, _, transport_mode = route[i]
                
                # Calculate transport time
                transport_time = curr_arrival - prev_departure
                total_travel_time += transport_time
        
        return total_travel_time
    
    def get_total_satisfaction(self):
        """
        Calculate the total satisfaction of the solution.
        
        Returns:
            float: Total satisfaction score
        """
        total_satisfaction = 0
        
        # Add satisfaction for each day's route
        for day in range(self.num_days):
            route = self.routes[day]
            
            # Process each location
            for loc, _, _, _ in route:
                loc_type = self.problem.locations[loc]["type"]
                if loc_type == "attraction":
                    total_satisfaction += self.problem.locations[loc]["satisfaction"]
                elif loc_type == "hawker":
                    total_satisfaction += self.problem.locations[loc]["rating"]
        
        return total_satisfaction

    def collect_constraint_violations(self):
        """
        Collect detailed information about constraint violations in the solution.
        
        Returns:
            list: List of constraint violation dictionaries with explanations
        """
        violations = []
        
        # 1. Check hotel start/end for each day
        for day in range(self.num_days):
            route = self.routes[day]
            
            # Empty route check
            if len(route) == 0:
                violations.append({
                    "type": "empty_route",
                    "day": day,
                    "details": f"Day {day+1} has no locations"
                })
                continue
            
            # Check start at hotel
            first_loc, first_time, _, _ = route[0]
            if first_loc != 0:
                violations.append({
                    "type": "hotel_start_missing",
                    "day": day,
                    "details": f"Day {day+1} does not start at hotel"
                })
            
            if first_time < self.problem.START_TIME:
                violations.append({
                    "type": "early_start",
                    "day": day,
                    "details": f"Day {day+1} starts before {self.problem.START_TIME//60}:00 AM"
                })
            
            # Check end at hotel
            last_loc, _, last_departure, _ = route[-1]
            if last_loc != 0:
                violations.append({
                    "type": "hotel_end_missing",
                    "day": day,
                    "details": f"Day {day+1} does not end at hotel"
                })
                
                # Check if we would return to hotel too late
                hotel_idx = 0
                transport_hour = self.problem.get_transport_hour(last_departure)
                try:
                    transport_key = (self.problem.locations[last_loc]["name"], 
                                self.problem.locations[hotel_idx]["name"], 
                                transport_hour)
                    if "transit" in self.problem.transport_matrix.get(transport_key, {}):
                        transport_data = self.problem.transport_matrix[transport_key]["transit"]
                    else:
                        transport_data = self.problem.transport_matrix[transport_key]["drive"]
                    
                    return_time = last_departure + transport_data["duration"]
                    if return_time > self.problem.HARD_LIMIT_END_TIME:
                        violations.append({
                            "type": "late_return",
                            "day": day,
                            "details": f"Day {day+1} would return to hotel after {self.problem.HARD_LIMIT_END_TIME//60}:00 PM"
                        })
                except (KeyError, TypeError):
                    # If no transport data, can't determine return time
                    pass
        
        # 2. Check for duplicate locations within each day
        for day in range(self.num_days):
            route = self.routes[day]
            location_counts = {}
            
            for i, (loc_idx, arrival_time, _, _) in enumerate(route):
                if loc_idx not in location_counts:
                    location_counts[loc_idx] = 0
                location_counts[loc_idx] += 1
            
            # Check for duplicates
            for loc_idx, count in location_counts.items():
                if count > 1 and loc_idx != 0:  # Allow hotel to appear twice (start and end)
                    loc_type = self.problem.locations[loc_idx]["type"]
                    loc_name = self.problem.locations[loc_idx]["name"]
                    
                    violations.append({
                        "type": f"duplicate_{loc_type}",
                        "day": day,
                        "location": loc_idx,
                        "name": loc_name,
                        "count": count,
                        "details": f"Day {day+1} visits {loc_name} ({loc_type}) {count} times"
                    })
        
        # 3. Check for attraction uniqueness across all days
        attraction_visits = {}
        for day in range(self.num_days):
            for loc_idx, _, _, _ in self.routes[day]:
                if self.problem.locations[loc_idx]["type"] == "attraction":
                    if loc_idx not in attraction_visits:
                        attraction_visits[loc_idx] = []
                    attraction_visits[loc_idx].append(day)
        
        # Check for duplicate attraction visits
        for attr_idx, days in attraction_visits.items():
            if len(days) > 1:
                attr_name = self.problem.locations[attr_idx]["name"]
                day_list = ", ".join([f"Day {d+1}" for d in days])
                
                violations.append({
                    "type": "duplicate_attraction_across_days",
                    "location": attr_idx,
                    "name": attr_name,
                    "days": days,
                    "details": f"Attraction {attr_name} visited on multiple days: {day_list}"
                })
        
        # 4. Check for proper meal scheduling
        for day in range(self.num_days):
            route = self.routes[day]
            lunch_visits = []
            dinner_visits = []
            
            for i, (loc_idx, arrival_time, _, _) in enumerate(route):
                if self.problem.locations[loc_idx]["type"] == "hawker":
                    # Check if it's a lunch or dinner visit
                    if arrival_time >= self.problem.LUNCH_START and arrival_time <= self.problem.LUNCH_END:
                        lunch_visits.append((i, loc_idx, self.problem.locations[loc_idx]["name"]))
                    elif arrival_time >= self.problem.DINNER_START and arrival_time <= self.problem.DINNER_END:
                        dinner_visits.append((i, loc_idx, self.problem.locations[loc_idx]["name"]))
                    else:
                        # Hawker visit outside meal times
                        violations.append({
                            "type": "hawker_outside_meal_time",
                            "day": day,
                            "location": loc_idx,
                            "name": self.problem.locations[loc_idx]["name"],
                            "time": arrival_time,
                            "details": f"Day {day+1} visits hawker {self.problem.locations[loc_idx]['name']} outside of meal times"
                        })
            
            # Check lunch
            if len(lunch_visits) == 0:
                violations.append({
                    "type": "missing_lunch",
                    "day": day,
                    "details": f"Day {day+1} has no lunch visit"
                })
            elif len(lunch_visits) > 1:
                hawker_names = [name for _, _, name in lunch_visits]
                violations.append({
                    "type": "multiple_lunches",
                    "day": day,
                    "count": len(lunch_visits),
                    "hawkers": hawker_names,
                    "details": f"Day {day+1} has {len(lunch_visits)} lunch visits: {', '.join(hawker_names)}"
                })
            
            # Check dinner
            if len(dinner_visits) == 0:
                violations.append({
                    "type": "missing_dinner",
                    "day": day,
                    "details": f"Day {day+1} has no dinner visit"
                })
            elif len(dinner_visits) > 1:
                hawker_names = [name for _, _, name in dinner_visits]
                violations.append({
                    "type": "multiple_dinners",
                    "day": day,
                    "count": len(dinner_visits),
                    "hawkers": hawker_names,
                    "details": f"Day {day+1} has {len(dinner_visits)} dinner visits: {', '.join(hawker_names)}"
                })
        
        # 5. Check for A->B->A patterns (unnecessary backtracking)
        for day in range(self.num_days):
            route = self.routes[day]
            for i in range(2, len(route)):
                if route[i][0] == route[i-2][0] and route[i][0] != 0:  # Allow hotel to repeat
                    # Found an A->B->A pattern
                    pattern = f"{self.problem.locations[route[i-2][0]]['name']} -> "
                    pattern += f"{self.problem.locations[route[i-1][0]]['name']} -> "
                    pattern += f"{self.problem.locations[route[i][0]]['name']}"
                    
                    violations.append({
                        "type": "unnecessary_backtracking",
                        "day": day,
                        "pattern": pattern,
                        "details": f"Day {day+1} has unnecessary backtracking: {pattern}"
                    })
        
        # 6. Check budget constraint
        total_cost = self.get_total_cost()
        if total_cost > self.problem.budget:
            violations.append({
                "type": "budget_exceeded",
                "cost": total_cost,
                "budget": self.problem.budget,
                "details": f"Total cost ${total_cost:.2f} exceeds budget ${self.problem.budget:.2f}"
            })
        
        # 7. Check attraction and hawker counts
        for day in range(self.num_days):
            route = self.routes[day]
            
            attraction_count = 0
            hawker_count = 0
            
            for loc_idx, _, _, _ in route:
                if self.problem.locations[loc_idx]["type"] == "attraction":
                    attraction_count += 1
                elif self.problem.locations[loc_idx]["type"] == "hawker":
                    hawker_count += 1
            
            if attraction_count > 4:  # More than 4 attractions in a day is unrealistic
                violations.append({
                    "type": "too_many_attractions",
                    "day": day,
                    "count": attraction_count,
                    "details": f"Day {day+1} has {attraction_count} attractions (maximum reasonable is 4)"
                })
            
            if hawker_count > 3:  # More than 3 hawkers in a day is unrealistic
                violations.append({
                    "type": "too_many_hawkers",
                    "day": day,
                    "count": hawker_count,
                    "details": f"Day {day+1} has {hawker_count} hawker visits (maximum reasonable is 3)"
                })
        
        return violations

    def evaluate(self):
        """
        Evaluate the solution to get objectives and feasibility.
        Added post-processing to fix common issues and improved feasibility checking.
        
        Returns:
            dict: Dictionary with evaluation results
        """
        # First, post-process the solution to fix common issues
        processed_solution = self.post_process_solution()
        
        # Check feasibility
        is_feasible = processed_solution.is_feasible()
        
        # Calculate costs
        total_cost = processed_solution.get_total_cost()
        total_travel_time = processed_solution.get_total_travel_time()
        total_satisfaction = processed_solution.get_total_satisfaction()
        
        # Extract daily routes in the format expected by visualization
        daily_routes = []
        for day in range(processed_solution.num_days):
            route = []
            for loc, arrival_time, departure_time, transport_mode in processed_solution.routes[day]:
                # Skip hotel at start if there are more locations
                if loc == 0 and len(route) == 0 and len(processed_solution.routes[day]) > 1:
                    continue
                    
                # Add location to route
                route.append({
                    "location": loc,
                    "name": self.problem.locations[loc]["name"],
                    "type": self.problem.locations[loc]["type"],
                    "time": float(arrival_time),
                    "transport_from_prev": transport_mode
                })
            daily_routes.append(route)
        
        # Get visited attractions
        visited_attractions = []
        for attr_idx in processed_solution.get_visited_attractions():
            visited_attractions.append(self.problem.locations[attr_idx]["name"])
        
        # Collect constraint violations using the new method
        violations = processed_solution.collect_constraint_violations() if not is_feasible else []
        
        # Group violations by type for easier analysis
        inequality_violations = []
        equality_violations = []
        
        # Categorize violations (this helps with backward compatibility)
        for violation in violations:
            violation_type = violation.get("type", "")
            
            # Put equality constraints in equality_violations, the rest in inequality_violations
            if violation_type in ["hotel_start_missing", "hotel_end_missing", "flow_conservation_violated", 
                            "attraction_source_dest_mismatch"]:
                equality_violations.append(violation)
            else:
                inequality_violations.append(violation)
        
        return {
            "is_feasible": is_feasible,
            "total_cost": float(total_cost),
            "total_travel_time": float(total_travel_time),
            "total_satisfaction": float(total_satisfaction),
            "daily_routes": daily_routes,
            "visited_attractions": visited_attractions,
            "constraint_violations": violations,  # New field with all violations
            "inequality_violations": inequality_violations,  # For compatibility
            "equality_violations": equality_violations       # For compatibility
        }

    def __str__(self):
        """String representation of the solution"""
        result = f"VRP Solution ({self.num_days} days):\n"
        result += f"Feasible: {self.is_feasible()}\n"
        result += f"Total Cost: ${self.get_total_cost():.2f}\n"
        result += f"Total Travel Time: {self.get_total_travel_time():.1f} minutes\n"
        result += f"Total Satisfaction: {self.get_total_satisfaction():.1f}\n\n"
        
        for day in range(self.num_days):
            result += f"Day {day+1}:\n"
            for i, (loc, arrival, departure, transport) in enumerate(self.routes[day]):
                loc_name = self.problem.locations[loc]["name"]
                loc_type = self.problem.locations[loc]["type"]
                
                # Format times
                arrival_hr = int(arrival // 60)
                arrival_min = int(arrival % 60)
                departure_hr = int(departure // 60)
                departure_min = int(departure % 60)
                
                # Add transport info if not first location
                transport_info = f" via {transport}" if transport else ""
                
                result += f"  {i+1}. [{arrival_hr:02d}:{arrival_min:02d} - {departure_hr:02d}:{departure_min:02d}] {loc_name} ({loc_type}){transport_info}\n"
            result += "\n"
        
        return result

    def post_process_solution(self):
        """
        Post-process the solution to fix common issues:
        - Remove redundant hotel visits in the middle of the day
        - Remove duplicate hawker visits
        - Ensure each day has exactly one lunch and one dinner
        - Prioritize dinner scheduling
        
        Returns:
            VRPSolution: A cleaned-up solution
        """
        # Make a copy of the solution
        new_solution = self.clone()
        
        # Process each day
        for day in range(new_solution.num_days):
            route = new_solution.routes[day]
            
            # Skip days with too few locations
            if len(route) < 3:
                continue
            
            # 1. Remove redundant hotel visits in the middle of the day
            i = 1  # Start from second position (after hotel start)
            while i < len(route) - 1:  # Don't remove last position
                loc_idx, _, _, _ = route[i]
                if self.problem.locations[loc_idx]["type"] == "hotel":
                    # Remove this hotel visit if it's not at start or end
                    route.pop(i)
                    # Recalculate times after removal
                    new_solution.recalculate_route_times(day, i)
                else:
                    i += 1
            
            # 2. Fix duplicate hawker visits - keep only the one closest to ideal meal time
            # First find all hawker visits
            hawker_visits = []
            for i, (loc_idx, arrival, _, _) in enumerate(route):
                if self.problem.locations[loc_idx]["type"] == "hawker":
                    # Determine if it's lunch or dinner
                    if arrival >= self.problem.LUNCH_START and arrival <= self.problem.LUNCH_END:
                        ideal_time = self.problem.LUNCH_START + 90  # 12:30 PM
                        hawker_visits.append((i, loc_idx, arrival, "lunch", abs(arrival - ideal_time)))
                    elif arrival >= self.problem.DINNER_START and arrival <= self.problem.DINNER_END:
                        ideal_time = self.problem.DINNER_START + 90  # 6:30 PM
                        hawker_visits.append((i, loc_idx, arrival, "dinner", abs(arrival - ideal_time)))
                    else:
                        # Unknown meal type, use closest meal time
                        if abs(arrival - (self.problem.LUNCH_START + 90)) < abs(arrival - (self.problem.DINNER_START + 90)):
                            hawker_visits.append((i, loc_idx, arrival, "lunch", abs(arrival - (self.problem.LUNCH_START + 90))))
                        else:
                            hawker_visits.append((i, loc_idx, arrival, "dinner", abs(arrival - (self.problem.DINNER_START + 90))))
            
            # Group hawker visits by meal type
            lunch_visits = [v for v in hawker_visits if v[3] == "lunch"]
            dinner_visits = [v for v in hawker_visits if v[3] == "dinner"]
            
            # For each meal type, keep only the best visit (closest to ideal time)
            positions_to_remove = []
            
            # Process lunch visits - keep only the best one
            if len(lunch_visits) > 1:
                # Sort by time difference (ascending)
                lunch_visits.sort(key=lambda x: x[4])
                # Mark all except the best for removal
                positions_to_remove.extend([pos for pos, _, _, _, _ in lunch_visits[1:]])
            
            # Process dinner visits - keep only the best one
            if len(dinner_visits) > 1:
                # Sort by time difference (ascending)
                dinner_visits.sort(key=lambda x: x[4])
                # Mark all except the best for removal
                positions_to_remove.extend([pos for pos, _, _, _, _ in dinner_visits[1:]])
            
            # Sort positions in reverse order to avoid index shifting issues
            positions_to_remove.sort(reverse=True)
            
            # Remove the marked positions
            for pos in positions_to_remove:
                route.pop(pos)
                # Recalculate times after each removal
                new_solution.recalculate_route_times(day, pos)
            
            # 3. HIGH PRIORITY: Add dinner if missing
            # First recalculate which meal types we have after removals
            has_lunch, has_dinner = new_solution.has_lunch_and_dinner(day)
            
            # PRIORITIZE ADDING DINNER FIRST
            if not has_dinner:
                # Find an optimal dinner time - try multiple potential positions
                dinner_positions = []
                
                # Start with ideal dinner time (6:30 PM)
                ideal_dinner = self.problem.DINNER_START + 90  # 6:30 PM
                
                # Find the position where dinner should be inserted (right before the location after dinner time)
                ideal_pos = len(route)  # Default to end of route
                for i, (_, arrival, _, _) in enumerate(route):
                    if arrival > ideal_dinner:
                        ideal_pos = i
                        break
                
                dinner_positions.append(ideal_pos)
                
                # Also try early dinner (5:30 PM)
                early_dinner = self.problem.DINNER_START + 30  # 5:30 PM
                early_pos = len(route)  # Default to end of route
                for i, (_, arrival, _, _) in enumerate(route):
                    if arrival > early_dinner:
                        early_pos = i
                        break
                
                if early_pos != ideal_pos:
                    dinner_positions.append(early_pos)
                
                # Also try late dinner (7:30 PM)
                late_dinner = self.problem.DINNER_START + 150  # 7:30 PM
                late_pos = len(route)  # Default to end of route
                for i, (_, arrival, _, _) in enumerate(route):
                    if arrival > late_dinner:
                        late_pos = i
                        break
                
                if late_pos != ideal_pos and late_pos != early_pos:
                    dinner_positions.append(late_pos)
                
                # Ensure end position is included
                if len(route) not in dinner_positions:
                    dinner_positions.append(len(route))
                
                # Find an available hawker for dinner
                avail_hawkers = []
                for h in range(self.problem.num_locations):
                    if self.problem.locations[h]["type"] == "hawker":
                        # Check if this hawker is already used in this day
                        if not any(loc == h for loc, _, _, _ in route):
                            avail_hawkers.append((h, self.problem.locations[h].get("rating", 0)))
                
                # Sort hawkers by rating (highest first)
                avail_hawkers.sort(key=lambda x: x[1], reverse=True)
                
                # Try each position and hawker until we find a feasible insertion
                dinner_added = False
                for pos in dinner_positions:
                    for hawker_idx, _ in avail_hawkers:
                        # Try both transit and drive
                        for transport_mode in ["transit", "drive"]:
                            if new_solution.is_feasible_insertion(day, pos, hawker_idx, transport_mode):
                                new_solution.insert_location(day, pos, hawker_idx, transport_mode)
                                dinner_added = True
                                break
                        if dinner_added:
                            break
                    if dinner_added:
                        break
                
                # Last resort: Try to force a dinner hawker by relaxing constraints
                if not dinner_added and avail_hawkers:
                    # Use the highest rated available hawker
                    hawker_idx = avail_hawkers[0][0]
                    
                    # Try to insert toward end of day
                    pos = len(new_solution.routes[day])
                    
                    # Try to insert anyway and adjust times if needed
                    for transport_mode in ["transit", "drive"]:
                        if new_solution.insert_location(day, pos, hawker_idx, transport_mode):
                            dinner_added = True
                            break
            
            # 4. Now add lunch if missing (after handling dinner)
            has_lunch, has_dinner = new_solution.has_lunch_and_dinner(day)
            
            if not has_lunch:
                # Find the best position to insert a lunch hawker
                lunch_time = self.problem.LUNCH_START + 60  # 12:00 PM
                best_pos = 1  # Default to after hotel
                
                # Find the position where we should insert lunch (right before the location after lunch time)
                for i, (_, arrival, _, _) in enumerate(route):
                    if arrival > lunch_time:
                        best_pos = i
                        break
                
                # Find an available hawker for lunch
                avail_hawkers = []
                for h in range(self.problem.num_locations):
                    if self.problem.locations[h]["type"] == "hawker":
                        # Check if this hawker is already used in this day
                        if not any(loc == h for loc, _, _, _ in route):
                            avail_hawkers.append((h, self.problem.locations[h].get("rating", 0)))
                
                # Sort hawkers by rating (highest first)
                avail_hawkers.sort(key=lambda x: x[1], reverse=True)
                
                # Try each available hawker
                lunch_added = False
                for hawker_idx, _ in avail_hawkers:
                    # Try both transit and drive
                    for transport_mode in ["transit", "drive"]:
                        if new_solution.is_feasible_insertion(day, best_pos, hawker_idx, transport_mode):
                            new_solution.insert_location(day, best_pos, hawker_idx, transport_mode)
                            lunch_added = True
                            break
                    if lunch_added:
                        break
            
            # 5. Make sure each day ends at the hotel
            if route[-1][0] != 0:  # If last location is not the hotel
                # Try to add hotel as the last location
                hotel_idx = 0
                position = len(route)
                
                # Try both transport modes
                if new_solution.is_feasible_insertion(day, position, hotel_idx, "transit"):
                    new_solution.insert_location(day, position, hotel_idx, "transit")
                elif new_solution.is_feasible_insertion(day, position, hotel_idx, "drive"):
                    new_solution.insert_location(day, position, hotel_idx, "drive")
        
        return new_solution