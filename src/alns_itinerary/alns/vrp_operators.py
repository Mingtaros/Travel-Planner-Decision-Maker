"""
Destroy and Repair Operators for Travel Itinerary Optimization
=============================================================

This module implements specialized operators for the Adaptive Large Neighborhood Search
(ALNS) algorithm used in travel itinerary optimization. These operators intelligently
modify solutions during the search process:

Destroy Operators:
- Remove parts of solutions in strategic ways to explore the search space
- Focus on different aspects: transit efficiency, satisfaction, cost, timing

Repair Operators:
- Rebuild partial solutions by inserting locations in efficient ways
- Ensure meal scheduling while maximizing overall satisfaction
- Maintain feasibility with respect to time, budget, and logical constraints

The operators work with the VRPSolution objects, which represent travel itineraries
as position-based Vehicle Routing Problem solutions.
"""

import random
import logging
import numpy as np
from collections import defaultdict
from utils.export_json_itinerary import export_json_itinerary
from datetime import datetime
from heapq import heappush, heappop
import math

logger = logging.getLogger(__name__)

class VRPOperators:
    """
    Collection of destroy and repair operators for travel itinerary optimization.
    
    This class provides a set of specialized operators that intelligently modify
    solutions during the ALNS search process. The operators are designed to handle
    the unique constraints of travel itineraries, such as meal scheduling, attraction
    uniqueness, and budget limitations.
    
    The operators are grouped into two categories:
    1. Destroy operators: Remove parts of solutions to explore the search space
    2. Repair operators: Rebuild partial solutions by inserting locations
    
    Attributes:
        min_subsequence (int): Minimum subsequence length for destruction operations
        destroy_remove_percentage (float): Percentage of locations to remove
        destroy_distant_loc_weights (list): Weights for evaluating location distance
        destroy_expensive_threshold (float): Threshold for expensive attraction removal
        destroy_day_hawker_preserve (float): Probability to preserve hawkers during day destruction
        ideal_meal_offset (int): Ideal offset from meal window start (in minutes)
        repair_budget_limit (float): Budget limit factor for repair operations
        repair_free_bonus (float): Bonus factor for free attractions
        repair_insertion_regret (float): Regret factor for single-option insertions
        repair_transit_weights (list): Weights for transit efficiency evaluation
        repair_satisfaction_weights (list): Weights for satisfaction-driven insertion
    """
    
    def __init__(
        self, 
        destroy_remove_percentage, 
        destroy_distant_loc_weights, 
        destroy_expensive_threshold,
        destroy_day_hawker_preserve,
        repair_transit_weights,
        repair_satisfaction_weights,
        ideal_meal_offset = 90,
        repair_budget_limit = 0.95,
        repair_free_bonus = 2,
        repair_insertion_regret = 1000,
        seed=None):
        """
        Initialize the VRP operators with configuration parameters.
        
        Args:
            destroy_remove_percentage (float): Percentage of locations to remove during destroy
                                            operations (typically 0.1-0.5)
            destroy_distant_loc_weights (list): Weights [time_weight, cost_weight] for evaluating
                                            the transit efficiency of locations
            destroy_expensive_threshold (float): Budget threshold ratio for triggering
                                                expensive attraction removal (0.0-1.0)
            destroy_day_hawker_preserve (float): Probability to preserve hawker centers
                                                during day destruction (0.0-1.0)
            repair_transit_weights (list): Weights [time_weight, cost_weight] for
                                        transit-efficient insertion
            repair_satisfaction_weights (list): Weights [time_weight, cost_weight] for
                                            satisfaction-driven insertion
            ideal_meal_offset (int): Minutes from start of meal window for ideal
                                    meal timing (default: 90)
            repair_budget_limit (float): Fraction of total budget to use as limit
                                        during repair operations (default: 0.95)
            repair_free_bonus (float): Bonus multiplier for attractions with no cost
                                    (default: 2)
            repair_insertion_regret (float): Regret value for locations with only
                                            one insertion option (default: 1000)
            seed (int, optional): Random seed for reproducibility
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        self.min_subsequence = 3
        self.destroy_remove_percentage = destroy_remove_percentage
        self.destroy_distant_loc_weights = destroy_distant_loc_weights
        self.destroy_expensive_threshold = destroy_expensive_threshold
        self.destroy_day_hawker_preserve = destroy_day_hawker_preserve
        self.ideal_meal_offset = ideal_meal_offset
        self.repair_budget_limit = repair_budget_limit
        self.repair_free_bonus = repair_free_bonus
        self.repair_insertion_regret = repair_insertion_regret
        self.repair_transit_weights = repair_transit_weights
        self.repair_satisfaction_weights = repair_satisfaction_weights

    #----------------
    # Destroy Operators
    #----------------

    def destroy_targeted_subsequence(self, problem, new_solution):
        """
        Remove a targeted subsequence of attractions from a daily route.
        
        This operator focuses on removing a subsequence of attractions from a random day,
        while preserving the meal structure (lunch and dinner). It identifies periods
        between meals and selectively removes attractions from these periods, making it
        easier to maintain feasibility when repairing the solution.
        
        Args:
            problem: TravelItineraryProblem instance containing constraints
            new_solution: VRPSolution instance to modify
            
        Returns:
            VRPSolution: Modified solution with a subsequence removed
            
        Note:
            This operator is particularly effective for exploring alternative
            arrangements of attractions within a day without disrupting meal timing.
        """
        
        # Select a random day
        day = random.randint(0, new_solution.num_days - 1)
        route = new_solution.routes[day]
        
        # Need at least 4 locations to remove a subsequence (hotel -> loc1 -> loc2 -> hotel)
        if len(route) < self.min_subsequence:
            return new_solution
        
        # Identify time periods between meals (avoid removing meals)
        morning_period = []  # Before lunch
        afternoon_period = []  # Between lunch and dinner
        evening_period = []  # After dinner
        
        lunch_pos = None
        dinner_pos = None
        
        # Find lunch and dinner positions
        for pos, (loc_idx, arrival, _, _) in enumerate(route):
            if problem.locations[loc_idx]["type"] == "hawker":
                if arrival >= problem.LUNCH_START and arrival <= problem.LUNCH_END:
                    lunch_pos = pos
                elif arrival >= problem.DINNER_START and arrival <= problem.DINNER_END:
                    dinner_pos = pos
        
        # Build time periods
        for pos in range(1, len(route)):  # Skip hotel start/end
            loc_type = problem.locations[route[pos][0]]["type"]
            if loc_type == "attraction":  # Only consider removing attractions
                if lunch_pos is not None and pos < lunch_pos:
                    morning_period.append(pos)
                elif lunch_pos is not None and dinner_pos is not None and lunch_pos < pos < dinner_pos:
                    afternoon_period.append(pos)
                elif dinner_pos is not None and pos > dinner_pos:
                    evening_period.append(pos)
        
        # Choose a period to target based on which has the most attractions
        periods = [period for period in [morning_period, afternoon_period, evening_period] if period]
        
        if not periods:
            return new_solution  # No valid periods found
        
        target_period = random.choice(periods)
        
        # Choose a subsequence to remove (up to 2 consecutive attractions)
        if len(target_period) >= 2:
            start_idx = random.randint(0, len(target_period) - 1)
            end_idx = min(start_idx + random.randint(0, 1), len(target_period) - 1)
            positions_to_remove = sorted(target_period[start_idx:end_idx + 1], reverse=True)
        elif len(target_period) == 1:
            positions_to_remove = [target_period[0]]
        else:
            return new_solution  # No positions to remove
        
        # Remove the selected positions
        for pos in positions_to_remove:
            new_solution.remove_location(day, pos)
        
        return new_solution

    def destroy_worst_attractions(self, problem, new_solution):
        """
        Remove attractions with the worst satisfaction-to-cost ratio.
        
        This operator identifies and removes attractions that provide the least value
        for their cost, creating opportunities to insert better attractions during the
        repair phase. It carefully avoids disrupting meal scheduling, ensuring the
        solution remains feasible after destruction.
        
        Args:
            problem: TravelItineraryProblem instance containing constraints
            new_solution: VRPSolution instance to modify
            
        Returns:
            VRPSolution: Modified solution with low-value attractions removed
            
        Note:
            This operator helps optimize the budget allocation by removing attractions
            that don't provide sufficient satisfaction relative to their cost.
        """
        
        # Identify attraction visits with their value ratio
        attraction_visits = []
        
        # Calculate value ratio for each visited attraction
        for day in range(new_solution.num_days):
            route = new_solution.routes[day]
            
            # Find lunch and dinner positions to avoid removing attractions that would disrupt meal timing
            lunch_pos = dinner_pos = None
            for pos, (loc_idx, arrival, _, _) in enumerate(route):
                if problem.locations[loc_idx]["type"] == "hawker":
                    if arrival >= problem.LUNCH_START and arrival <= problem.LUNCH_END:
                        lunch_pos = pos
                    elif arrival >= problem.DINNER_START and arrival <= problem.DINNER_END:
                        dinner_pos = pos
            
            for pos, (loc_idx, _, _, _) in enumerate(route):
                if problem.locations[loc_idx]["type"] == "attraction":
                    satisfaction = problem.locations[loc_idx].get("satisfaction", 0)
                    cost = problem.locations[loc_idx].get("entrance_fee", 1)
                    duration = problem.locations[loc_idx].get("duration", 60)
                    
                    # Calculate value ratio (lower is worse)
                    value_ratio = satisfaction / (cost + duration/60)

                    attraction_visits.append((day, pos, loc_idx, value_ratio))
        
        # Sort by value ratio (ascending - worst first)
        attraction_visits.sort(key=lambda x: x[3])
        
        # Remove up to 25% of worst attractions, but at least 1 if any are visited
        num_to_remove = max(1, int(len(attraction_visits) * self.destroy_remove_percentage))
        num_to_remove = min(num_to_remove, len(attraction_visits))
        
        # Track positions that have been removed to avoid issues with shifting indices
        removed = defaultdict(set)
        
        # Remove worst attractions
        for i in range(num_to_remove):
            day, pos, _, _ = attraction_visits[i]
            
            # Adjust position for previous removals on the same day
            removed_before_pos = len([p for p in removed[day] if p < pos])
            adjusted_pos = pos - removed_before_pos
            
            # Remove the attraction
            if new_solution.remove_location(day, adjusted_pos):
                removed[day].add(pos)
        
        return new_solution

    def destroy_distant_locations(self, problem, new_solution):
        """
        Remove attractions and hawkers that require excessive travel time or cost,
        including meal-time hawkers.
        
        Args:
            problem: TravelItineraryProblem instance
            new_solution: VRPSolution instance
            
        Returns:
            VRPSolution: Modified solution
        """
        # Collect all transit information
        transit_data = []
        
        # Analyze transit for each location
        for day in range(new_solution.num_days):
            route = new_solution.routes[day]
            
            # Skip days with insufficient locations
            if len(route) < self.min_subsequence:
                continue
            
            # Process each location (excluding start hotel)
            for pos in range(1, len(route)):
                loc_idx = route[pos][0]
                
                # Skip only hotel
                if loc_idx == 0:
                    continue
                
                # Get transit to this location
                prev_loc_idx = route[pos-1][0]
                prev_departure = route[pos-1][2]
                transport_mode = route[pos][3]
                
                # Get transit data from transport matrix
                transport_hour = problem.get_transport_hour(prev_departure)
                transport_key = (problem.locations[prev_loc_idx]["name"],
                            problem.locations[loc_idx]["name"],
                            transport_hour)
                
                transport_data = problem.transport_matrix[transport_key][transport_mode]
                real_transit_time = transport_data["duration"]
                transit_cost = transport_data["price"]
                
                # Get transit from this location to next (if not last location)
                outbound_transit_time = 0
                outbound_transit_cost = 0
                
                if pos < len(route) - 1:
                    next_loc_idx = route[pos+1][0]
                    departure = route[pos][2]
                    next_transport = route[pos+1][3]
                else:
                    # If last location, use return to hotel
                    next_loc_idx = 0
                    departure = route[pos][2]
                    next_transport = new_solution.hotel_return_transport
                    
                transport_hour = problem.get_transport_hour(departure)
                transport_key = (problem.locations[loc_idx]["name"],
                            problem.locations[next_loc_idx]["name"],
                            transport_hour)
                
                transport_data = problem.transport_matrix[transport_key][next_transport]
                outbound_transit_time = transport_data["duration"]
                outbound_transit_cost = transport_data["price"]
                
                # Calculate total transit impact (inbound + outbound)
                total_transit_time = real_transit_time + outbound_transit_time
                total_transit_cost = transit_cost + outbound_transit_cost
                
                # Calculate transit efficiency metric
                location_value = 0
                if problem.locations[loc_idx]["type"] == "attraction":
                    # For attractions, use satisfaction
                    location_value = problem.locations[loc_idx].get("satisfaction", 1)
                elif problem.locations[loc_idx]["type"] == "hawker":
                    # For hawkers, use rating
                    location_value = problem.locations[loc_idx].get("rating", 1) * 2  # Scale rating
                
                log_time = math.log(total_transit_time + 1)
                log_cost = math.log(total_transit_cost + 1)
                # Higher score = worse transit efficiency (more time/cost per value)
                transit_efficiency = (log_time * self.destroy_distant_loc_weights[0] + log_cost * self.destroy_distant_loc_weights[1]) / (location_value + 1)
                
                # Add to transit data collection
                transit_data.append((day, pos, loc_idx, transit_efficiency, total_transit_time))
        
        # Sort by transit efficiency (descending - worst first)
        transit_data.sort(key=lambda x: x[3], reverse=True)
        
        # Determine how many to remove (up to 30% of candidates or at most 3)
        num_to_remove = min(int(self.destroy_remove_percentage*10), max(1, int(len(transit_data) * self.destroy_remove_percentage)))
        
        # Track removed positions
        removed = defaultdict(set)
        
        # Remove the locations with worst transit efficiency
        for i in range(min(num_to_remove, len(transit_data))):
            day, pos, loc_idx, _, _ = transit_data[i]
            
            # Adjust position for previous removals on the same day
            removed_before_pos = len([p for p in removed[day] if p < pos])
            adjusted_pos = pos - removed_before_pos
            
            # Remove the location
            if new_solution.remove_location(day, adjusted_pos):
                removed[day].add(pos)
        
        return new_solution

    def destroy_expensive_attractions(self, problem, new_solution):
        """
        Focus on removing expensive attractions to improve budget utilization.
        Balances cost reduction with maintaining satisfaction.
        
        Args:
            problem: TravelItineraryProblem instance
            solution: VRPSolution instance
            
        Returns:
            VRPSolution: Modified solution
        """
        
        # Get current solution cost
        total_cost = new_solution.get_total_cost()
        
        if total_cost <= problem.budget * self.destroy_expensive_threshold:
            return new_solution
            
        # Identify expensive attractions
        expensive_attractions = []
        
        for day in range(new_solution.num_days):
            route = new_solution.routes[day]
            
            for pos, (loc_idx, _, _, _) in enumerate(route):
                if problem.locations[loc_idx]["type"] == "attraction":
                    cost = problem.locations[loc_idx].get("entrance_fee", 0)
                    satisfaction = problem.locations[loc_idx].get("satisfaction", 0)
                    
                    # Calculate cost-to-satisfaction ratio (higher is worse)
                    if satisfaction > 0:
                        cost_ratio = cost * 2 / satisfaction
                    else:
                        cost_ratio = float('inf')
                    
                    expensive_attractions.append((day, pos, loc_idx, cost, cost_ratio))
        
        # Sort by cost ratio (descending - worst first)
        expensive_attractions.sort(key=lambda x: x[4], reverse=True)
        
        # Remove up to 30% of expensive attractions
        num_to_remove = max(1, int(len(expensive_attractions) * self.destroy_remove_percentage))
        num_to_remove = min(num_to_remove, len(expensive_attractions))
        
        # Track positions that have been removed to avoid issues with shifting indices
        removed = defaultdict(set)
        
        # Remove expensive attractions
        for i in range(num_to_remove):
            day, pos, _, _, _ = expensive_attractions[i]
            
            # Adjust position for previous removals on the same day
            removed_before_pos = len([p for p in removed[day] if p < pos])
            adjusted_pos = pos - removed_before_pos
            
            # Remove the attraction
            if new_solution.remove_location(day, adjusted_pos):
                removed[day].add(pos)
        
        return new_solution

    def destroy_selected_day(self, problem, new_solution):
        """
        More focused day destruction that maintains the start and end hotel visits
        and preserves lunch and dinner, but removes other activities from a selected day.
        Helps with reorganizing an entire day at once.
        
        Args:
            problem: TravelItineraryProblem instance
            solution: VRPSolution instance
            
        Returns:
            VRPSolution: Modified solution
        """
        
        # Select a random day
        day = random.randint(0, new_solution.num_days - 1)
        route = new_solution.routes[day]
        
        # Need at least 3 locations to destroy (hotel -> loc1 -> loc2)
        if len(route) < 3:
            return new_solution
        
        # Find all attraction positions and hawker positions
        attraction_positions = []
        hawker_positions = []
        
        for pos, (loc_idx, arrival, _, _) in enumerate(route):
            if problem.locations[loc_idx]["type"] == "attraction":
                attraction_positions.append(pos)
            elif problem.locations[loc_idx]["type"] == "hawker":
                hawker_positions.append(pos)
        
        # If no attractions to remove, return the original solution
        if not attraction_positions:
            return new_solution
        
        # Small chance to keep hawkers in place and just remove attractions
        preserve_hawkers = random.random() < self.destroy_day_hawker_preserve
        
        # Positions to remove (attractions and possibly hawkers)
        positions_to_remove = attraction_positions.copy()
        if not preserve_hawkers:
            positions_to_remove.extend(hawker_positions)
        
        # Sort positions in reverse order to avoid index issues when removing
        positions_to_remove.sort(reverse=True)
        
        # Remove all those positions
        for pos in positions_to_remove:
            new_solution.remove_location(day, pos)
        
        return new_solution

    #----------------
    # Repair Operators
    #----------------

    def repair_regret_insertion(self, problem, new_solution):
        """
        Repair a solution using regret-based insertion with improved meal scheduling.
        
        This sophisticated repair operator first ensures each day has proper meal scheduling
        (lunch and dinner at appropriate times), then uses a regret-based approach to
        insert attractions. The regret value represents how much worse the solution would
        be if an attraction is not inserted at its best position, prioritizing attractions
        with limited insertion options.
        
        The operator:
        1. Ensures each day has lunch and dinner at appropriate times
        2. Calculates regret values for each unvisited attraction
        3. Inserts attractions in order of highest regret
        4. Respects budget and time constraints
        
        Args:
            problem: TravelItineraryProblem instance containing constraints
            new_solution: VRPSolution instance to repair
            
        Returns:
            VRPSolution: Repaired solution with meals and attractions
            
        Note:
            This is one of the most effective repair operators as it balances immediate
            greedy choices with foresight about future insertion opportunities.
        """
        
        # logger.info("Repairing solution with regret-based insertion")
        # logger.info(f"Initial solution routes: {new_solution.routes}")
            
        # Make a copy of the solution to avoid modifying the original
        # new_solution = solution.clone()
        
        # First, ensure each day has lunch and dinner
        for day in range(new_solution.num_days):
            # logger.info(f"Checking day {day+1} for meal scheduling")
            # Check for lunch and dinner
            has_lunch, has_dinner, lunch_hawker_idx, dinner_hawker_idx = new_solution.has_lunch_and_dinner(day)
            # logger.info(f"Has lunch: {has_lunch}, has dinner: {has_dinner}")
            
            # Get hawker centers ordered by rating
            hawkers = [(i, problem.locations[i].get("rating", 0)) 
                    for i in range(problem.num_locations) 
                    if problem.locations[i]["type"] == "hawker"]
            hawkers.sort(key=lambda x: x[1], reverse=True)  # Sort by rating (highest first)
            
            if not has_lunch:
                # Try each hawker until one can be inserted during lunch window
                for hawker_idx, _ in hawkers:
                    if dinner_hawker_idx is not None and dinner_hawker_idx == hawker_idx:
                        # Skip hawker if used for dinner
                        continue
                    # Try to identify the best position for lunch
                    best_pos = None
                    best_time_diff = float('inf')
                    
                    # Aim for 12:30 PM (ideal lunch time)
                    ideal_lunch = problem.LUNCH_START + self.ideal_meal_offset
                    
                    # Try each position
                    for pos in range(1, len(new_solution.routes[day]) + 1):
                        # Try both transit and drive
                        for transport_mode in ["transit", "drive"]:
                            if new_solution.is_feasible_insertion(day, pos, hawker_idx, transport_mode):
                                # Clone solution to test insertion
                                test_sol = new_solution.clone()
                                test_sol.insert_location(day, pos, hawker_idx, transport_mode)
                                # logger.debug(f"Testing lunch insertion at day {day+1}, pos {pos}, hawker {hawker_idx}, mode {transport_mode}")
                                
                                # Get the actual arrival time
                                arrival_time = test_sol.routes[day][pos][1]
                                
                                # Check if within lunch window or close to it
                                if (arrival_time >= problem.LUNCH_START and 
                                    arrival_time <= problem.LUNCH_END):
                                    # Calculate how close to ideal
                                    time_diff = abs(arrival_time - ideal_lunch)
                                    
                                    if time_diff < best_time_diff:
                                        best_time_diff = time_diff
                                        best_pos = (pos, transport_mode)
                    
                    # Insert lunch hawker at the best position if found
                    if best_pos:
                        pos, transport_mode = best_pos
                        new_solution.insert_location(day, pos, hawker_idx, transport_mode, 'Lunch')
                        lunch_hawker_idx = hawker_idx
                        # logger.debug(f"Inserted lunch hawker at day {day+1}, pos {pos}, hawker {hawker_idx}, mode {transport_mode}")
                        break
            
            # Add dinner first (prioritize dinner scheduling)
            if not has_dinner:
                # Try each hawker until one can be inserted during dinner window
                for hawker_idx, _ in hawkers:
                    # Skip hawkers already used for dinner today
                    if lunch_hawker_idx is not None and hawker_idx == lunch_hawker_idx:
                        continue
                    # Try to identify the best position for dinner
                    best_pos = None
                    best_time_diff = float('inf')
                    
                    # Aim for 6:30 PM (ideal dinner time)
                    ideal_dinner = problem.DINNER_START + self.ideal_meal_offset
                    
                    # Try each position
                    for pos in range(1, len(new_solution.routes[day]) + 1):
                        # Try both transit and drive
                        for transport_mode in ["transit", "drive"]:
                            # logger.info(f"Checking dinner insertion at day {day+1}, pos {pos}, hawker {hawker_idx}, mode {transport_mode}")
                            if new_solution.is_feasible_insertion(day, pos, hawker_idx, transport_mode):
                                # Clone solution to test insertion
                                test_sol = new_solution.clone()
                                test_sol.insert_location(day, pos, hawker_idx, transport_mode)
                                # logger.info(f"Testing dinner insertion at day {day+1}, pos {pos}, hawker {hawker_idx}, mode {transport_mode}")
                                # Get the actual arrival time
                                arrival_time = test_sol.routes[day][pos][1]
                                
                                # Check if within dinner window
                                if (arrival_time >= problem.DINNER_START and 
                                    arrival_time <= problem.DINNER_END):
                                    # Calculate how close to ideal
                                    time_diff = abs(arrival_time - ideal_dinner)
                                    
                                    if time_diff < best_time_diff:
                                        best_time_diff = time_diff
                                        best_pos = (pos, transport_mode)
                    
                    # Insert dinner hawker at the best position if found
                    if best_pos:
                        pos, transport_mode = best_pos
                        new_solution.insert_location(day, pos, hawker_idx, transport_mode, 'Dinner')
                        # logger.info(f"Inserted dinner hawker at day {day+1}, pos {pos}, hawker {hawker_idx}, mode {transport_mode}")
                        break
        
        # Now apply regret-based insertion for attractions
        # Get unvisited attractions
        visited_attractions = new_solution.get_visited_attractions()
        
        unvisited_attractions = [i for i in range(problem.num_locations) 
                            if problem.locations[i]["type"] == "attraction" 
                            and i not in visited_attractions]
        
        # Apply regret insertion until no more attractions can be inserted or budget is reached
        budget_limit = problem.budget * self.repair_budget_limit  # 95% of budget to leave some slack
        
        while unvisited_attractions and new_solution.get_total_cost() < budget_limit:
            # Calculate regret values for each unvisited attraction
            regret_values = []
            
            for attr_idx in unvisited_attractions:
                # Find the best and second-best insertion positions
                insertion_costs = []
                
                for day in range(new_solution.num_days):
                    for pos in range(1, len(new_solution.routes[day]) + 1):
                        # Try both transit and drive
                        for transport_mode in ["transit", "drive"]:
                            if new_solution.is_feasible_insertion(day, pos, attr_idx, transport_mode):
                                # Calculate cost of insertion
                                test_sol = new_solution.clone()
                                test_sol.insert_location(day, pos, attr_idx, transport_mode)
                                
                                # Use negative satisfaction as cost (we want to maximize satisfaction)
                                satisfaction_increase = test_sol.get_total_satisfaction() - new_solution.get_total_satisfaction()
                                cost_increase = test_sol.get_total_cost() - new_solution.get_total_cost()
                                
                                # Calculate normalized cost (lower is better)
                                if cost_increase > 0:
                                    normalized_cost = -satisfaction_increase / cost_increase
                                else:
                                    normalized_cost = -satisfaction_increase * self.repair_free_bonus
                                
                                insertion_costs.append((normalized_cost, day, pos, transport_mode))
                
                # Sort insertion costs (lowest normalized cost first)
                insertion_costs.sort()
                
                # Calculate regret value
                if len(insertion_costs) >= 2:
                    # Regret is the difference between best and second-best
                    regret = insertion_costs[1][0] - insertion_costs[0][0]
                    best_insertion = insertion_costs[0]
                elif len(insertion_costs) == 1:
                    # Only one feasible insertion, set high regret
                    regret = self.repair_insertion_regret
                    best_insertion = insertion_costs[0]
                else:
                    # No feasible insertion, skip this attraction
                    continue
                
                regret_values.append((attr_idx, regret, best_insertion))
            
            # If no regret values, we're done
            if not regret_values:
                # logger.info("No more feasible attractions to insert")
                break
            
            # Sort by regret (highest first)
            regret_values.sort(key=lambda x: x[1], reverse=True)
            
            # Insert the attraction with the highest regret
            attr_idx, _, best_insertion = regret_values[0]
            _, day, pos, transport_mode = best_insertion
            
            # Insert the attraction
            new_solution.insert_location(day, pos, attr_idx, transport_mode)
            
            # Remove from unvisited list
            unvisited_attractions.remove(attr_idx)
            
            # Check if we've reached the budget limit
            if new_solution.get_total_cost() >= budget_limit:
                break
        
        return new_solution

    def repair_transit_efficient_insertion(self, problem, new_solution):
        """
        Robust repair operator ensuring meal feasibility with systematic insertion strategy
        
        Args:
            problem: TravelItineraryProblem instance
            new_solution: VRPSolution instance
            
        Returns:
            VRPSolution: Feasible solution with proper meal insertions
        """
        def find_meal_insertion_positions(day_route, meal_type):
            """
            Find potential meal insertion positions respecting time windows
            
            Args:
                day_route: Route for a specific day
                meal_type: 'lunch' or 'dinner'
            
            Returns:
                list: Possible insertion positions
            """
            # Define time windows
            if meal_type == 'lunch':
                start_window = problem.LUNCH_START
                end_window = problem.LUNCH_END
            else:  # dinner
                start_window = problem.DINNER_START
                end_window = problem.DINNER_END
            
            possible_positions = []
            
            # Check each possible insertion position
            for pos in range(1, len(day_route) + 1):
                # Special handling for first/last route positions
                if pos == 1:
                    if problem.START_TIME < end_window:
                        possible_positions.append(pos)
                elif pos == len(day_route):
                    # Check if we can insert at the end of the route
                    if day_route[pos-1][2] <= end_window:
                        possible_positions.append(pos)
                else:
                    # Check between existing route points
                    prev_time = day_route[pos-1][2]
                    if prev_time < end_window:
                        possible_positions.append(pos)
            
            return possible_positions
        
        def find_hawker_for_meal(hawker_indices, day, meal_type, other_hawker=-1):
            """
            Find a suitable hawker for a specific meal
            
            Args:
                hawker_indices: List of hawker location indices
                day: Day index
                meal_type: 'lunch' or 'dinner'
            
            Returns:
                tuple: (hawker_index, best_position, transport_mode) or None
            """
            route = new_solution.routes[day]
            
            # Find possible insertion positions
            possible_positions = find_meal_insertion_positions(route, meal_type)
            
            # Prioritize hawkers
            best_hawker = None
            best_score = float('-inf')
            best_position = None
            best_transport_mode = None
            
            # Iterate through possible hawkers and positions
            for hawker_idx in hawker_indices:
                if other_hawker is not None and hawker_idx == other_hawker:
                    # Skip if hawker is already used for the other meal
                    continue
                
                for pos in possible_positions:
                    for transport_mode in ["drive", "transit"]:
                        # Check feasibility of insertion
                        if not new_solution.is_feasible_insertion(day, pos, hawker_idx, transport_mode):
                            continue
                        
                        # Calculate a score based on rating and location
                        rating = problem.locations[hawker_idx].get("rating", 0)
                        
                        # Additional scoring logic
                        try:
                            # Get transport details from previous location
                            prev_loc = route[pos-1][0]
                            departure_time = route[pos-1][2]
                            
                            transport_hour = problem.get_transport_hour(departure_time)
                            transport_key = (problem.locations[prev_loc]["name"], 
                                            problem.locations[hawker_idx]["name"], 
                                            transport_hour)
                            
                            transport_data = problem.transport_matrix[transport_key][transport_mode]
                            
                            log_time = math.log(transport_data["duration"] + 1)
                            log_cost = math.log(transport_data["price"] + 1)
                            
                            # Score considers rating and minimizes transit overhead
                            score = rating / (log_time * self.repair_transit_weights[0] + log_cost * self.repair_transit_weights[1] + 1)
                            
                            # Update best hawker if score is better
                            if score > best_score:
                                best_score = score
                                best_hawker = hawker_idx
                                best_position = pos
                                best_transport_mode = transport_mode
                        
                        except (KeyError, TypeError):
                            # Skip if transport data is unavailable
                            continue
            
            return best_hawker, best_position, best_transport_mode
        
        # Iterate through each day
        for day in range(new_solution.num_days):
            # Get hawker indices
            hawker_indices = [i for i in range(problem.num_locations) 
                            if problem.locations[i]["type"] == "hawker"]
            # Check existing meal status
            route = new_solution.routes[day]
            has_lunch, has_dinner, lunch_hawker_idx, dinner_hawker_idx = new_solution.has_lunch_and_dinner(day)
            
            # Strategy for meal insertion
            if not has_lunch:
                # Look for lunch hawker
                lunch_hawker, lunch_pos, lunch_transport = find_hawker_for_meal(
                    hawker_indices, day, 'lunch', dinner_hawker_idx
                )
                
                # Insert lunch if found
                if lunch_hawker is not None:
                    lunch_hawker_idx = lunch_hawker
                    new_solution.insert_location(
                        day, lunch_pos, lunch_hawker, lunch_transport, 'Lunch'
                    )
            
            # Refresh route after potential lunch insertion
            route = new_solution.routes[day]
            
            if not has_dinner:
                # Look for dinner hawker
                dinner_hawker, dinner_pos, dinner_transport = find_hawker_for_meal(
                    hawker_indices, day, 'dinner', lunch_hawker_idx
                )
                
                # Insert dinner if found
                if dinner_hawker is not None:
                    new_solution.insert_location(
                        day, dinner_pos, dinner_hawker, dinner_transport, 'Dinner'
                    )
        
        # Continue with attraction insertion (you can use existing logic)
        def schedule_attractions(new_solution):
            """
            Efficient attraction scheduling
            """
            # Track already visited attractions
            visited_attractions = new_solution.get_visited_attractions()
            unvisited_attractions = [
                i for i in range(problem.num_locations) 
                if problem.locations[i]["type"] == "attraction" 
                and i not in visited_attractions
            ]
            
            # Budget tracking
            budget_limit = problem.budget * self.repair_budget_limit
            
            # Simple greedy attraction insertion
            for attr_idx in unvisited_attractions:
                if new_solution.get_total_cost() >= budget_limit:
                    break
                
                # Try to insert in each day
                inserted = False
                for day in range(new_solution.num_days):
                    route = new_solution.routes[day]
                    
                    for pos in range(1, len(route) + 1):
                        for transport_mode in ["transit", "drive"]:
                            if new_solution.is_feasible_insertion(day, pos, attr_idx, transport_mode):
                                new_solution.insert_location(
                                    day, pos, attr_idx, transport_mode
                                )
                                inserted = True
                                break
                        
                        if inserted:
                            break
                    
                    if inserted:
                        break
            
            return new_solution
        
        # Run attraction scheduling
        schedule_attractions(new_solution)
        
        return new_solution

    def repair_satisfaction_driven_insertion(self, problem, new_solution):
        """
        Repair a solution prioritizing overall satisfaction.
        
        This operator focuses on maximizing the total satisfaction of the itinerary.
        It first ensures proper meal scheduling for each day, then inserts attractions
        in order of their satisfaction rating, while considering transit efficiency.
        
        The operator:
        1. Ensures each day has lunch and dinner at appropriate times
        2. Sorts unvisited attractions by satisfaction (highest first)
        3. Inserts attractions in priority order where feasible
        4. Maintains budget and time constraints
        
        Args:
            problem: TravelItineraryProblem instance containing constraints
            new_solution: VRPSolution instance to repair
            
        Returns:
            VRPSolution: Repaired solution optimized for satisfaction
            
        Note:
            This operator works well for maximizing customer enjoyment when budget
            is not the primary constraint.
        """
        def find_meal_insertion_positions(day_route, meal_type):
            """
            Find potential meal insertion positions respecting time windows
            
            Args:
                day_route: Route for a specific day
                meal_type: 'lunch' or 'dinner'
            
            Returns:
                list: Possible insertion positions
            """
            # Define time windows
            if meal_type == 'lunch':
                start_window = problem.LUNCH_START
                end_window = problem.LUNCH_END
            else:  # dinner
                start_window = problem.DINNER_START
                end_window = problem.DINNER_END
            
            possible_positions = []
            
            # Check each possible insertion position
            for pos in range(1, len(day_route) + 1):
                # Special handling for first/last route positions
                if pos == 1:
                    if problem.START_TIME < end_window:
                        possible_positions.append(pos)
                elif pos == len(day_route):
                    # Check if we can insert at the end of the route
                    if day_route[pos-1][2] <= end_window:
                        possible_positions.append(pos)
                else:
                    # Check between existing route points
                    prev_time = day_route[pos-1][2]
                    if prev_time < end_window:
                        possible_positions.append(pos)
            
            return possible_positions
        
        def find_hawker_for_meal(hawker_indices, day, meal_type, other_hawker=-1):
            """
            Find a suitable hawker for a specific meal
            
            Args:
                hawker_indices: List of hawker location indices
                day: Day index
                meal_type: 'lunch' or 'dinner'
            
            Returns:
                tuple: (hawker_index, best_position, transport_mode) or None
            """
            route = new_solution.routes[day]
            
            # Find possible insertion positions
            possible_positions = find_meal_insertion_positions(route, meal_type)
            
            # Prioritize hawkers
            best_hawker = None
            best_score = float('-inf')
            best_position = None
            best_transport_mode = None
            
            # Iterate through possible hawkers and positions
            for hawker_idx in hawker_indices:
                if other_hawker is not None and hawker_idx == other_hawker:
                    continue
                for pos in possible_positions:
                    for transport_mode in ["drive", "transit"]:
                        # Check feasibility of insertion
                        if not new_solution.is_feasible_insertion(day, pos, hawker_idx, transport_mode):
                            continue
                        
                        # Calculate a score based on rating and satisfaction
                        rating = problem.locations[hawker_idx].get("rating", 0)
                        
                        # Additional scoring logic
                        try:
                            # Get transport details from previous location
                            prev_loc = route[pos-1][0]
                            departure_time = route[pos-1][2]
                            
                            transport_hour = problem.get_transport_hour(departure_time)
                            transport_key = (problem.locations[prev_loc]["name"], 
                                            problem.locations[hawker_idx]["name"], 
                                            transport_hour)
                            
                            transport_data = problem.transport_matrix[transport_key][transport_mode]
                            
                            log_time = math.log(transport_data["duration"] + 1)
                            log_cost = math.log(transport_data["price"] + 1)
                            
                            # Satisfaction-driven scoring:
                            # Prioritize highly-rated hawkers with minimal transit overhead
                            score = rating / (log_time * self.repair_satisfaction_weights[0] + log_cost * self.repair_satisfaction_weights[1] + 1)
                            
                            # Update best hawker if score is better
                            if score > best_score:
                                best_score = score
                                best_hawker = hawker_idx
                                best_position = pos
                                best_transport_mode = transport_mode
                        
                        except (KeyError, TypeError):
                            # Skip if transport data is unavailable
                            continue
            
            return best_hawker, best_position, best_transport_mode
        
        # Iterate through each day
        for day in range(new_solution.num_days):
            # Get hawker indices
            hawker_indices = [i for i in range(problem.num_locations) 
                            if problem.locations[i]["type"] == "hawker"]
            
            # Check existing meal status
            route = new_solution.routes[day]
            has_lunch, has_dinner, lunch_hawker_idx, dinner_hawker_idx = new_solution.has_lunch_and_dinner(day)
            # Strategy for meal insertion
            if not has_lunch:
                # Look for lunch hawker
                lunch_hawker, lunch_pos, lunch_transport = find_hawker_for_meal(
                    hawker_indices, day, 'lunch', dinner_hawker_idx
                )
                
                # Insert lunch if found
                if lunch_hawker is not None:
                    lunch_hawker_idx = lunch_hawker
                    new_solution.insert_location(
                        day, lunch_pos, lunch_hawker, lunch_transport, 'Lunch'
                    )
            
            # Refresh route after potential lunch insertion
            route = new_solution.routes[day]
            
            if not has_dinner:
                # Look for dinner hawker
                dinner_hawker, dinner_pos, dinner_transport = find_hawker_for_meal(
                    hawker_indices, day, 'dinner', lunch_hawker_idx
                )
                
                # Insert dinner if found
                if dinner_hawker is not None:
                    new_solution.insert_location(
                        day, dinner_pos, dinner_hawker, dinner_transport, 'Dinner'
                    )
        
        def schedule_attractions(new_solution):
            """
            Satisfaction-driven attraction scheduling
            """
            # Track already visited attractions
            visited_attractions = new_solution.get_visited_attractions()
            unvisited_attractions = [
                i for i in range(problem.num_locations) 
                if problem.locations[i]["type"] == "attraction" 
                and i not in visited_attractions
            ]
            
            # Sort attractions by satisfaction in descending order
            unvisited_attractions.sort(
                key=lambda attr: problem.locations[attr].get("satisfaction", 0), 
                reverse=True
            )
            
            # Budget tracking
            budget_limit = problem.budget * self.repair_budget_limit
            
            # Satisfaction-driven attraction insertion
            for attr_idx in unvisited_attractions:
                if new_solution.get_total_cost() >= budget_limit:
                    break
                
                # Try to insert in each day
                inserted = False
                for day in range(new_solution.num_days):
                    route = new_solution.routes[day]
                    
                    # Prioritize positions based on current route satisfaction
                    # This helps maintain a balanced satisfaction distribution
                    for pos in range(1, len(route) + 1):
                        for transport_mode in ["transit", "drive"]:
                            if new_solution.is_feasible_insertion(day, pos, attr_idx, transport_mode):
                                # Additional check: assess the impact on route satisfaction
                                # This could be a method in the VRPSolution class
                                new_solution.insert_location(
                                    day, pos, attr_idx, transport_mode
                                )
                                inserted = True
                                break
                        
                        if inserted:
                            break
                    
                    if inserted:
                        break
                
                # Optional: add a random chance to explore alternative insertions
                # This prevents getting stuck in local optima
                if len(unvisited_attractions) > 10 and random.random() < 0.1:
                    break
            
            return new_solution
        
        # Run attraction scheduling
        schedule_attractions(new_solution)
        
        return new_solution