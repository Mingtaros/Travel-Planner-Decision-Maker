"""
VRP Solution Representation
==========================

This module provides the core data structure for representing travel itinerary solutions
using a position-based Vehicle Routing Problem (VRP) approach.

The VRPSolution class handles:
- Route representation for multi-day itineraries
- Feasibility checking against time, budget, and logical constraints
- Location insertion and removal operations
- Comprehensive solution evaluation
- Detailed constraint violation reporting

Each solution tracks the complete schedule of locations visited on each day,
including arrival/departure times and transportation modes.
"""

import copy
import numpy as np
import logging

logger = logging.getLogger(__name__)

class VRPSolution:
    """
    Position-based representation for travel itineraries.
    
    This class manages the complete state of a travel itinerary solution,
    providing methods to manipulate routes, check feasibility, and evaluate
    solution quality.
    
    A solution consists of multiple daily routes, where each route is a sequence
    of locations with arrival/departure times and transport modes. The solution
    enforces complex constraints like meal timing, attraction uniqueness, and
    budget limitations.
    
    Attributes:
        problem: Reference to the TravelItineraryProblem instance
        num_days: Number of days in the itinerary
        routes: List of daily routes, where each route is a list of location tuples
               (location_idx, arrival_time, departure_time, transport_mode)
        hotel_return_transport: Transport mode for returning to hotel (default: 'drive')
        hotel_transit_duration: Duration for hotel return in minutes
        MAX_HAWKERS_PER_DAY: Maximum number of hawker centers to visit per day (default: 2)
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
        self.hotel_return_transport = ['drive'] * self.num_days
        self.hotel_transit_duration = 0
        self.MAX_HAWKERS_PER_DAY = 2
        
        # Initialize all days to start at hotel
        hotel_idx = 0  # Assuming hotel is at index 0
        for day in range(self.num_days):
            # Add hotel as first location (no transport for first location)
            self.routes[day].append((hotel_idx, problem.START_TIME, problem.START_TIME, None))
    
    def _get_transport_data(self, origin_idx, dest_idx, departure_time, mode="transit"):
        """Get transport data between two locations at a specific time."""
        transport_hour = self.problem.get_transport_hour(departure_time)
        transport_key = (
            self.problem.locations[origin_idx]["name"],
            self.problem.locations[dest_idx]["name"],
            transport_hour
        )
        
        try:
            return self.problem.transport_matrix[transport_key][mode]
        except KeyError:
            # Log warning about missing data
            logger.warning(f"Missing transport data for {transport_key} with mode {mode}")
            # Return default data
            return {
                "duration": 30,  # Default 30 minutes
                "price": 5       # Default $5
            }
    
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
        try:
            transport_data = self._get_transport_data(prev_loc, location_idx, prev_departure, transport_mode)
        
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
        
        This method adds a new location to a day's route, calculating the appropriate
        arrival and departure times based on travel duration from the previous location.
        For hawker centers, it can enforce meal timing windows (lunch or dinner).
        
        Args:
            day (int): Day index to modify (0-indexed)
            position (int): Position in the route to insert the location (1 = after hotel)
            location_idx (int): Index of the location to insert
            transport_mode (str): Transportation mode to use ("transit" or "drive")
            meal (str, optional): For hawkers, specifies "Lunch" or "Dinner" to enforce timing
        
        Returns:
            tuple: (success, departure_time)
                - success (bool): True if insertion was successful
                - departure_time (int): Departure time from the inserted location
        
        Note:
            This method does not check feasibility - use is_feasible_insertion() first
            to determine if the insertion would create a valid solution.
        """
        route = self.routes[day]
        
        # Ensure position is valid
        if position < 1 or position > len(route):
            logger.warning(f"Invalid position {position} for insertion")
            return False
        
        # Get preceding location
        prev_loc, _, prev_departure, _ = route[position-1]
        
        # Calculate travel time to new location
        try:
            transport_data = self._get_transport_data(prev_loc, location_idx, prev_departure, transport_mode)
            
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
            curr_loc, arrival_time, _, transport_mode = route[i]
            
            # Calculate travel time
            transport_data = self._get_transport_data(prev_loc, curr_loc, prev_departure, transport_mode)
            # Calculate arrival time
            actual_arrival_time = prev_departure + transport_data["duration"]
            
            # Special handling for hawkers (enforce meal windows)
            if self.problem.locations[curr_loc]["type"] == "hawker" and arrival_time > actual_arrival_time:
                # Check if it's lunch or dinner time
                if arrival_time >= self.problem.LUNCH_START and arrival_time <= self.problem.LUNCH_END:
                    # Lunch visit - ensure it's within lunch window
                    arrival_time = max(self.problem.LUNCH_START, actual_arrival_time)
                elif arrival_time >= self.problem.DINNER_START and arrival_time <= self.problem.DINNER_END:
                    # Dinner visit - ensure it's within dinner window
                    arrival_time = max(self.problem.DINNER_START, actual_arrival_time)
            else:
                arrival_time = actual_arrival_time
            
            # Calculate departure time
            location_duration = self.problem.locations[curr_loc]["duration"]
            departure_time = arrival_time + location_duration
            
            # Update the route with new times
            route[i] = (curr_loc, arrival_time, departure_time, transport_mode)
        
        return True

    def is_feasible_insertion(self, day, position, location_idx, transport_mode="transit"):
        """
        Check if inserting a location at a position would create a feasible solution.
        
        Performs comprehensive checks including:
        - Time window constraints (location open hours, meal times)
        - Budget limitations
        - Maximum attractions/hawkers per day
        - Uniqueness constraints (no duplicate attractions)
        - Meal scheduling (proper lunch/dinner timing)
        
        Args:
            day (int): Day index to check
            position (int): Position to insert
            location_idx (int): Location index to insert
            transport_mode (str): Transportation mode to use
        
        Returns:
            bool: True if the insertion would create a feasible solution
            
        Note:
            This check is crucial before any insertion to maintain solution validity.
            It prevents creating solutions that violate problem constraints.
        """
        route = self.routes[day]
        
        # Ensure position is valid
        if position < 1 or position > len(route):
            # logger.warning(f"Invalid position {position} for insertion")
            return False
        
        # Location-type specific checks
        location_type = self.problem.locations[location_idx]["type"]
        
        # Get preceding location
        prev_loc, _, prev_departure, _ = route[position-1]
        
        # Check if we're trying to insert the same location
        if prev_loc == location_idx:
            # logger.warning(f"Cannot insert same location {location_idx} consecutively")
            return False
        
        # Calculate arrival time
        transport_data = self._get_transport_data(prev_loc, location_idx, prev_departure, transport_mode)
        arrival_time = prev_departure + transport_data["duration"]
        arr_transit_cost = transport_data["price"]
        loc_cost = 0
        # Check uniqueness constraints based on location type
        # No inserting hotel
        if location_type == "hotel":
            return False
        
        elif location_type == "attraction":
            loc_cost = self.problem.locations[location_idx].get('entrance_fee', 0)
            # Check if this attraction is already in any route (attractions can only be visited once)
            for d in range(self.num_days):
                for loc, _, _, _ in self.routes[d]:
                    if loc == location_idx:
                        return False
        
        elif location_type == "hawker":
            
            has_lunch, has_dinner, _, _ = self.has_lunch_and_dinner(day)

            if has_lunch and has_dinner:
                return False
            
            loc_cost = self.problem.locations[location_idx].get('avg_food_price', 0)
            
            if (arrival_time < self.problem.LUNCH_START or arrival_time < self.problem.LUNCH_END) and not has_lunch:
                arrival_time = max(arrival_time, self.problem.LUNCH_START)
            elif (arrival_time < self.problem.DINNER_START or arrival_time < self.problem.DINNER_END) and not has_dinner:
                arrival_time = max(arrival_time, self.problem.DINNER_START)
            
            # Check if this would be a lunch or dinner visit
            is_lunch = (arrival_time >= self.problem.LUNCH_START and arrival_time <= self.problem.LUNCH_END)
            is_dinner = (arrival_time >= self.problem.DINNER_START and arrival_time <= self.problem.DINNER_END)
            
            # Prevent multiple dinner visits, but with flexibility
            if is_dinner and has_dinner:
                # logger.warning("Cannot insert multiple dinner visits in a day")
                return False
            
            # If it's neither lunch nor dinner, not allowed
            if not (is_lunch or is_dinner):
                # logger.warning("Hawker visit must be within lunch or dinner hours")
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
        
        # Check against maximum allowed visits - relaxed for hawkers to ensure dinner can be added   # Reasonable limit (lunch, dinner)
        if attraction_count > self.problem.MAX_ATTRACTION_PER_DAY:
            return False
        
        if hawker_count > self.MAX_HAWKERS_PER_DAY:
            return False
        
        # Now perform the original feasibility checks
        # Calculate departure time
        location_duration = self.problem.locations[location_idx]["duration"]
        departure_time = arrival_time + location_duration
        
        # Check if we return to hotel too late
        hotel_idx = 0
        if position == len(route):
            # Need to calculate return to hotel
            next_loc = hotel_idx
            next_transport = self.hotel_return_transport[day]
        else:
            # Check if next location can still be reached on time
            next_loc, next_arrival, _, next_transport = route[position]
        
        # Check if we are repeating the same location
        if next_loc == location_idx and next_loc != hotel_idx:
            return False
        
        transport_data = self._get_transport_data(location_idx, next_loc, departure_time, next_transport)
        new_next_arrival = departure_time + transport_data["duration"]
        dep_transit_cost = transport_data["price"]
        
        if position == len(route):
            if new_next_arrival > self.problem.HARD_LIMIT_END_TIME:
                # logger.warning("Cannot return to hotel after Day End")
                return False
        else:
            if new_next_arrival > next_arrival:
                # logger.warning("Cannot reach next location in time")
                return False
            else:
                new_next_arrival = next_arrival
                
        # Check for budget constraint
        total_cost = self.get_total_cost() + arr_transit_cost + loc_cost + dep_transit_cost
        if total_cost > self.problem.budget:
            # logger.warning(f"Exceeding budget constraint (${self.problem.budget})")
            return False
        
        return True
    
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
        lunch_hawker_idx = None
        dinner_hawker_idx = None
        
        for loc, arrival_time, _, _ in route:
            if self.problem.locations[loc]["type"] == "hawker":
                if arrival_time >= self.problem.LUNCH_START and arrival_time <= self.problem.LUNCH_END:
                    has_lunch = True
                    lunch_hawker_idx = loc
                elif arrival_time >= self.problem.DINNER_START and arrival_time <= self.problem.DINNER_END:
                    has_dinner = True
                    dinner_hawker_idx = loc
        
        return has_lunch, has_dinner, lunch_hawker_idx, dinner_hawker_idx
    
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
    
    def get_hotel_return(self, day):
        """
        Calculate the return to hotel time and cost for a day's route.
        """
        route = self.routes[day]

        last_loc, _, last_departure, _ = route[-1]
        if last_loc == 0:
            return last_departure, 0, 0
        
        transport_data = self._get_transport_data(last_loc, 0, last_departure, self.hotel_return_transport[day])
        return_transit_duration = transport_data["duration"]
        return_transit_cost = transport_data["price"]
        hotel_arrival = last_departure + return_transit_duration
        
        return hotel_arrival, return_transit_cost, return_transit_duration
        
    def is_feasible(self):
        """
        Determine if the complete solution satisfies all constraints.
        
        Performs comprehensive validation including:
        - Hotel start and end for each day
        - Daily time window adherence (start after 9 AM, end before hard limit)
        - Budget constraints
        - Location uniqueness (each attraction visited at most once)
        - Meal scheduling (exactly one lunch and one dinner per day)
        - Reasonable daily attraction count
        
        Returns:
            bool: True if the solution satisfies all constraints
            
        Note:
            This is a comprehensive check used to validate complete solutions.
            For incremental validation during construction, use is_feasible_insertion().
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
            hotel_arrival, return_cost, _ = self.get_hotel_return(day)
            if hotel_arrival > self.problem.HARD_LIMIT_END_TIME:
                return False
            
            # Check for duplicate locations within each day and hotel in between the day
            locations_visited = {}
            for i, (loc_idx, arrival_time, _, _) in enumerate(route):
                loc_type = self.problem.locations[loc_idx]["type"]
                
                # Hotel should only appear at start and end
                if loc_type == "hotel" and i > 0:
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
            if attraction_count > self.problem.MAX_ATTRACTION_PER_DAY:  # More than 4 attractions in a day is unrealistic
                return False
        
        # Check that attractions are visited at most once across all days
        attraction_visits = {}
        for day in range(self.num_days):
            for loc, _, _, _ in self.routes[day]:
                if self.problem.locations[loc]["type"] == "attraction":
                    if loc in attraction_visits:
                        return False
                    attraction_visits[loc] = True
        
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
        total_cost = 0
        
        # Add costs for each day's route
        for day in range(self.num_days):
            route = self.routes[day]
            
            # Process each location in sequence
            for i in range(1, len(route)):
                prev_loc, _, prev_departure, _ = route[i-1]
                curr_loc, _, _, transport_mode = route[i]
                
                # Calculate transport cost
                transport_data = self._get_transport_data(prev_loc, curr_loc, prev_departure, transport_mode)
                total_cost += transport_data["price"]
                
                # Add location costs
                loc_type = self.problem.locations[curr_loc]["type"]
                if loc_type == "attraction":
                    total_cost += self.problem.locations[curr_loc]["entrance_fee"]
                elif loc_type == "hawker":
                    total_cost += self.problem.locations[curr_loc]["avg_food_price"]
                
                # logger.info(f"Adding cost for {self.problem.locations[curr_loc]['name']}: {transport_data['price']} + {self.problem.locations[curr_loc]['entrance_fee' if loc_type == 'attraction' else 'avg_food_price']}")
            
            hotel_arrival, return_cost, _ = self.get_hotel_return(day)
            total_cost += return_cost
            
            # logger.info(f"Adding return cost for day {day+1}: {return_cost}")
            
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
                
                raw_transit_time = curr_arrival - prev_departure
                
                location_type = self.problem.locations[curr_loc]["type"]
                rest_period = 0
                
                if location_type == "hawker":
                    
                    # Calculate transport cost
                    transport_data = self._get_transport_data(prev_loc, curr_loc, prev_departure, transport_mode)
                    actual_transit_time = transport_data["duration"]
                    rest_period = raw_transit_time - actual_transit_time
                    
                # Subtract rest period from transit time
                transport_time = raw_transit_time - int(rest_period)
                total_travel_time += transport_time
            
            hotel_arrival, return_cost, return_duration = self.get_hotel_return(day)
            total_travel_time += return_duration
            
        return int(total_travel_time)
    
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
        
        return round(total_satisfaction, 1)

    def collect_constraint_violations(self):
        """
        Collect detailed information about constraint violations in the solution.
        
        Performs a comprehensive analysis of the solution, identifying all constraint
        violations with detailed explanations. This is valuable for debugging and
        for providing feedback to users about why a solution is infeasible.
        
        Checks include:
        - Hotel start/end for each day
        - Duplicate location visits
        - Attraction uniqueness across days
        - Meal scheduling (lunch/dinner timing)
        - Budget constraints
        - Maximum attractions/hawkers per day
        
        Returns:
            list: List of constraint violation dictionaries, each containing:
                - type (str): Type of violation
                - details (str): Human-readable explanation
                - Additional fields specific to each violation type
        """
        violations = []
        
        # 1. Check hotel start/end for each day
        for day in range(self.num_days):
            route = self.routes[day]
            
            # Empty route check
            if len(route) == 0:
                violations.append({
                    "type": "empty_route",
                    "day": day+1,
                    "details": f"Day {day+1} has no locations"
                })
                continue
            
            # Check start at hotel
            first_loc, first_time, _, _ = route[0]
            if first_loc != 0:
                violations.append({
                    "type": "hotel_start_missing",
                    "day": day+1,
                    "details": f"Day {day+1} does not start at hotel"
                })
            
            if first_time < self.problem.START_TIME:
                violations.append({
                    "type": "early_start",
                    "day": day+1,
                    "details": f"Day {day+1} starts before {self.problem.START_TIME//60}:00 AM"
                })
            
            # Check end at hotel
            hotel_arrival, return_cost, return_duration = self.get_hotel_return(day)
            if hotel_arrival > self.problem.HARD_LIMIT_END_TIME:
                violations.append({
                    "type": "late_return",
                    "day": day+1,
                    "details": f"Day {day+1} would return to hotel after {self.problem.HARD_LIMIT_END_TIME//60}:00 PM"
                })
        
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
                        "day": day+1,
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
                            "day": day+1,
                            "location": loc_idx,
                            "name": self.problem.locations[loc_idx]["name"],
                            "time": arrival_time,
                            "details": f"Day {day+1} visits hawker {self.problem.locations[loc_idx]['name']} outside of meal times"
                        })
            
            # Check lunch
            if len(lunch_visits) == 0:
                violations.append({
                    "type": "missing_lunch",
                    "day": day+1,
                    "details": f"Day {day+1} has no lunch visit"
                })
            elif len(lunch_visits) > 1:
                hawker_names = [name for _, _, name in lunch_visits]
                violations.append({
                    "type": "multiple_lunches",
                    "day": day+1,
                    "count": len(lunch_visits),
                    "hawkers": hawker_names,
                    "details": f"Day {day+1} has {len(lunch_visits)} lunch visits: {', '.join(hawker_names)}"
                })
            
            # Check dinner
            if len(dinner_visits) == 0:
                violations.append({
                    "type": "missing_dinner",
                    "day": day+1,
                    "details": f"Day {day+1} has no dinner visit"
                })
            elif len(dinner_visits) > 1:
                hawker_names = [name for _, _, name in dinner_visits]
                violations.append({
                    "type": "multiple_dinners",
                    "day": day+1,
                    "count": len(dinner_visits),
                    "hawkers": hawker_names,
                    "details": f"Day {day+1} has {len(dinner_visits)} dinner visits: {', '.join(hawker_names)}"
                })
        
        # 5. Check budget constraint
        total_cost = self.get_total_cost()
        if total_cost > self.problem.budget:
            violations.append({
                "type": "budget_exceeded",
                "cost": total_cost,
                "budget": self.problem.budget,
                "details": f"Total cost ${total_cost:.2f} exceeds budget ${self.problem.budget:.2f}"
            })
        
        # 6. Check attraction and hawker counts
        for day in range(self.num_days):
            route = self.routes[day]
            
            attraction_count = 0
            hawker_count = 0
            
            for loc_idx, _, _, _ in route:
                if self.problem.locations[loc_idx]["type"] == "attraction":
                    attraction_count += 1
                elif self.problem.locations[loc_idx]["type"] == "hawker":
                    hawker_count += 1
            
            if attraction_count > self.problem.MAX_ATTRACTION_PER_DAY:  # More than 4 attractions in a day is unrealistic
                violations.append({
                    "type": "too_many_attractions",
                    "day": day+1,
                    "count": attraction_count,
                    "details": f"Day {day+1} has {attraction_count} attractions (maximum reasonable is 4)"
                })
            
            if hawker_count > 2:  # More than 3 hawkers in a day is unrealistic
                violations.append({
                    "type": "too_many_hawkers",
                    "day": day+1,
                    "count": hawker_count,
                    "details": f"Day {day+1} has {hawker_count} hawker visits (maximum reasonable is 2)"
                })
        
        return violations

    def evaluate(self):
        """
        Evaluate the solution to calculate all performance metrics.
        
        Computes comprehensive metrics including:
        - Feasibility status
        - Total cost (transportation + attractions + meals)
        - Total travel time in minutes
        - Total satisfaction score
        - Daily routes in a format suitable for visualization
        - Visited attractions list
        - Detailed constraint violations (if any)
        
        Returns:
            dict: Evaluation results with the following keys:
                - is_feasible (bool): Whether the solution is valid
                - total_cost (float): Total cost in SGD
                - total_travel_time (float): Total travel time in minutes
                - total_satisfaction (float): Total satisfaction score
                - daily_routes (list): Formatted daily routes for visualization
                - visited_attractions (list): Names of visited attractions
                - constraint_violations (list): Detailed violation reports
                - inequality_violations (list): Inequality constraint violations
                - equality_violations (list): Equality constraint violations
        """
        solution = self.clone()
        
        # Check feasibility
        is_feasible = solution.is_feasible()
        
        # Calculate costs
        total_cost = solution.get_total_cost()
        total_travel_time = solution.get_total_travel_time()
        total_satisfaction = solution.get_total_satisfaction()
        
        # Extract daily routes in the format expected by visualization
        daily_routes = []
        for day in range(solution.num_days):
            route = []
            for loc, arrival_time, departure_time, transport_mode in solution.routes[day]:
                    
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
        for attr_idx in solution.get_visited_attractions():
            visited_attractions.append(self.problem.locations[attr_idx]["name"])
        
        # Collect constraint violations using the new method
        violations = solution.collect_constraint_violations() if not is_feasible else []
        
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