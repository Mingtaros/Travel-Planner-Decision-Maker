"""
Destroy and repair operators for the VRP-based travel itinerary optimizer.
This module contains specialized operators for the position-based VRP representation.
"""

import random
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

#----------------
# Destroy Operators
#----------------

def destroy_random_day_subsequence(problem, solution):
    """
    Remove a random subsequence from a route on a random day
    
    Args:
        problem: TravelItineraryProblem instance
        solution: VRPSolution instance
        
    Returns:
        VRPSolution: Modified solution
    """
    # Make a copy of the solution to avoid modifying the original
    new_solution = solution.clone()
    
    # Select a random day
    day = random.randint(0, new_solution.num_days - 1)
    route = new_solution.routes[day]
    
    # Need at least 4 locations to remove a subsequence (hotel -> loc1 -> loc2 -> hotel)
    if len(route) < 4:
        return new_solution
    
    # Choose start and end of subsequence to remove (avoid first and last positions)
    start_pos = random.randint(1, len(route) - 3)
    max_length = min(3, len(route) - start_pos - 1)  # Max 3 locations or what's available
    length = random.randint(1, max_length)
    
    # Remove the subsequence
    for _ in range(length):
        new_solution.remove_location(day, start_pos)
    
    return new_solution

def destroy_worst_attractions(problem, solution):
    """
    Remove attractions with the worst satisfaction-to-cost ratio
    
    Args:
        problem: TravelItineraryProblem instance
        solution: VRPSolution instance
        
    Returns:
        VRPSolution: Modified solution
    """
    # Make a copy of the solution to avoid modifying the original
    new_solution = solution.clone()
    
    # Identify attraction visits with their value ratio
    attraction_visits = []
    
    # Calculate value ratio for each visited attraction
    for day in range(new_solution.num_days):
        route = new_solution.routes[day]
        
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
    
    # Remove up to 30% of worst attractions, but at least 1 if any are visited
    num_to_remove = max(1, int(len(attraction_visits) * 0.3))
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

def destroy_random_attractions(problem, solution):
    """
    Remove a random selection of attractions
    
    Args:
        problem: TravelItineraryProblem instance
        solution: VRPSolution instance
        
    Returns:
        VRPSolution: Modified solution
    """
    # Make a copy of the solution to avoid modifying the original
    new_solution = solution.clone()
    
    # Find all attraction positions
    attraction_positions = []
    
    for day in range(new_solution.num_days):
        route = new_solution.routes[day]
        
        for pos, (loc_idx, _, _, _) in enumerate(route):
            if problem.locations[loc_idx]["type"] == "attraction":
                attraction_positions.append((day, pos))
    
    # Randomly select attractions to remove (40-60%)
    removal_percentage = random.uniform(0.4, 0.6)
    num_to_remove = max(1, int(len(attraction_positions) * removal_percentage))
    num_to_remove = min(num_to_remove, len(attraction_positions))
    
    if attraction_positions:
        to_remove = random.sample(attraction_positions, num_to_remove)
        
        # Sort in reverse order of (day, position) to avoid issues with shifting indices
        to_remove.sort(reverse=True)
        
        # Remove selected attractions
        for day, pos in to_remove:
            new_solution.remove_location(day, pos)
    
    return new_solution

def destroy_random_meals(problem, solution, preserve_one_per_day=True):
    """
    Remove some random meal (hawker) visits
    
    Args:
        problem: TravelItineraryProblem instance
        solution: VRPSolution instance
        preserve_one_per_day: Keep at least one meal per day
        
    Returns:
        VRPSolution: Modified solution
    """
    # Make a copy of the solution to avoid modifying the original
    new_solution = solution.clone()
    
    # Find all hawker visits by day
    hawker_positions = defaultdict(list)
    
    for day in range(new_solution.num_days):
        route = new_solution.routes[day]
        
        for pos, (loc_idx, arrival, _, _) in enumerate(route):
            if problem.locations[loc_idx]["type"] == "hawker":
                is_lunch = (arrival >= problem.LUNCH_START and arrival <= problem.LUNCH_END)
                is_dinner = (arrival >= problem.DINNER_START and arrival <= problem.DINNER_END)
                meal_type = "lunch" if is_lunch else "dinner" if is_dinner else "other"
                hawker_positions[day].append((pos, meal_type))
    
    # Remove random hawkers, preserving at least one per day if specified
    removable_hawkers = []
    
    for day, positions in hawker_positions.items():
        if preserve_one_per_day and len(positions) <= 1:
            # Skip days with only one hawker
            continue
        
        if preserve_one_per_day:
            # Randomly select all but one to be candidates for removal
            to_preserve = random.randint(0, len(positions) - 1)
            for i, (pos, _) in enumerate(positions):
                if i != to_preserve:
                    removable_hawkers.append((day, pos))
        else:
            # All hawkers are candidates for removal
            for pos, _ in positions:
                removable_hawkers.append((day, pos))
    
    # Randomly select hawkers to remove (30-40%)
    removal_percentage = random.uniform(0.3, 0.4)
    num_to_remove = max(1, int(len(removable_hawkers) * removal_percentage))
    num_to_remove = min(num_to_remove, len(removable_hawkers))
    
    if removable_hawkers:
        # Sort in reverse order of (day, position) to avoid issues with shifting indices
        removable_hawkers.sort(reverse=True)
        
        # Select random subset to remove if there are more than we want to remove
        if len(removable_hawkers) > num_to_remove:
            to_remove = random.sample(removable_hawkers, num_to_remove)
            to_remove.sort(reverse=True)  # Sort again to ensure correct order
        else:
            to_remove = removable_hawkers
        
        # Remove selected hawkers
        for day, pos in to_remove:
            new_solution.remove_location(day, pos)
    
    return new_solution

def destroy_time_window_violations(problem, solution):
    """
    Remove locations that cause time window violations
    
    Args:
        problem: TravelItineraryProblem instance
        solution: VRPSolution instance
        
    Returns:
        VRPSolution: Modified solution
    """
    # Make a copy of the solution to avoid modifying the original
    new_solution = solution.clone()
    
    # Find time window violations
    violations = []
    
    for day in range(new_solution.num_days):
        route = new_solution.routes[day]
        
        # Check for meal time window violations
        for pos, (loc_idx, arrival, _, _) in enumerate(route):
            if problem.locations[loc_idx]["type"] == "hawker":
                is_lunch = (arrival >= problem.LUNCH_START and arrival <= problem.LUNCH_END)
                is_dinner = (arrival >= problem.DINNER_START and arrival <= problem.DINNER_END)
                
                if not (is_lunch or is_dinner):
                    # This hawker is visited outside meal windows
                    violations.append((day, pos))
            
            # Check for late returns to hotel
            elif loc_idx == 0 and pos > 0:
                if arrival > problem.HARD_LIMIT_END_TIME:
                    # Find non-hotel location before this that could be causing the issue
                    if pos > 1:
                        violations.append((day, pos-1))
    
    # If no explicit violations found, check for locations that might be causing tight schedules
    if not violations:
        for day in range(new_solution.num_days):
            route = new_solution.routes[day]
            
            # Look for locations close to time windows
            for pos in range(1, len(route) - 1):
                _, _, departure, _ = route[pos]
                next_loc, next_arrival, _, _ = route[pos+1]
                
                # If next location is a hawker and arrival time is close to window boundary
                if problem.locations[next_loc]["type"] == "hawker":
                    if (next_arrival >= problem.LUNCH_START - 15 and next_arrival <= problem.LUNCH_START + 15) or \
                       (next_arrival >= problem.LUNCH_END - 15 and next_arrival <= problem.LUNCH_END + 15) or \
                       (next_arrival >= problem.DINNER_START - 15 and next_arrival <= problem.DINNER_START + 15) or \
                       (next_arrival >= problem.DINNER_END - 15 and next_arrival <= problem.DINNER_END + 15):
                        violations.append((day, pos))
    
    # Remove some of the violations (up to 2 per day)
    day_count = defaultdict(int)
    
    # Sort violations to remove in reverse order to avoid issues with shifting indices
    violations.sort(reverse=True)
    
    for day, pos in violations:
        if day_count[day] < 2:  # Limit to 2 removals per day
            if new_solution.remove_location(day, pos):
                day_count[day] += 1
    
    return new_solution

def destroy_day_shuffle(problem, solution):
    """
    Randomize the order of locations on a day while preserving hotel start/end
    
    Args:
        problem: TravelItineraryProblem instance
        solution: VRPSolution instance
        
    Returns:
        VRPSolution: Modified solution
    """
    # Make a copy of the solution to avoid modifying the original
    new_solution = solution.clone()
    
    # Select a random day
    day = random.randint(0, new_solution.num_days - 1)
    route = new_solution.routes[day]
    
    # Need at least 4 locations to shuffle (hotel -> loc1 -> loc2 -> hotel)
    if len(route) < 4:
        return new_solution
    
    # Extract all non-hotel locations
    locations = [(loc_idx, transport) for loc_idx, _, _, transport in route[1:-1]]
    
    # Shuffle their order
    random.shuffle(locations)
    
    # Create a new route
    new_route = [route[0]]  # Start with hotel
    
    # Add shuffled locations
    for loc_idx, transport in locations:
        # Default transport mode if None
        transport_mode = transport if transport else "transit"
        
        # Determine insert position
        position = len(new_route)
        
        # Try to insert the location
        if new_solution.is_feasible_insertion(day, position, loc_idx, transport_mode):
            new_solution.insert_location(day, position, loc_idx, transport_mode)
        else:
            # If not feasible, try alternate transport mode
            alt_transport = "drive" if transport_mode == "transit" else "transit"
            if new_solution.is_feasible_insertion(day, position, loc_idx, alt_transport):
                new_solution.insert_location(day, position, loc_idx, alt_transport)
    
    # Add hotel return if not already present
    if new_solution.routes[day][-1][0] != 0:
        # Try to add return to hotel
        hotel_idx = 0
        position = len(new_solution.routes[day])
        
        # Try both transport modes
        if new_solution.is_feasible_insertion(day, position, hotel_idx, "transit"):
            new_solution.insert_location(day, position, hotel_idx, "transit")
        elif new_solution.is_feasible_insertion(day, position, hotel_idx, "drive"):
            new_solution.insert_location(day, position, hotel_idx, "drive")
    
    return new_solution

#----------------
# Repair Operators
#----------------

def repair_greedy_insertion(problem, solution):
    """
    Repair solution by greedily inserting missing hawkers and high-value attractions
    
    Args:
        problem: TravelItineraryProblem instance
        solution: VRPSolution instance
        
    Returns:
        VRPSolution: Modified solution
    """
    # Make a copy of the solution to avoid modifying the original
    new_solution = solution.clone()
    
    # First, ensure each day has lunch and dinner
    for day in range(new_solution.num_days):
        # Check for lunch and dinner
        has_lunch, has_dinner = new_solution.has_lunch_and_dinner(day)
        
        # Get hawker centers ordered by rating
        hawkers = [(i, problem.locations[i].get("rating", 0)) 
                 for i in range(problem.num_locations) 
                 if problem.locations[i]["type"] == "hawker"]
        hawkers.sort(key=lambda x: x[1], reverse=True)  # Sort by rating (highest first)
        
        # Add lunch if missing
        if not has_lunch:
            # Try each hawker until one can be inserted during lunch window
            for hawker_idx, _ in hawkers:
                # Try to identify the best position for lunch
                best_pos = None
                best_time_diff = float('inf')
                
                for pos in range(1, len(new_solution.routes[day]) + 1):
                    # Try both transit and drive
                    for transport_mode in ["transit", "drive"]:
                        if new_solution.is_feasible_insertion(day, pos, hawker_idx, transport_mode):
                            # Clone solution to test insertion
                            test_sol = new_solution.clone()
                            test_sol.insert_location(day, pos, hawker_idx, transport_mode)
                            
                            # Get the actual arrival time
                            arrival_time = test_sol.routes[day][pos][1]
                            
                            # Calculate how close it is to ideal lunch time (12:30 PM)
                            ideal_lunch = problem.LUNCH_START + 90  # 12:30 PM
                            time_diff = abs(arrival_time - ideal_lunch)
                            
                            if time_diff < best_time_diff:
                                best_time_diff = time_diff
                                best_pos = (pos, transport_mode)
                
                # Insert lunch hawker at the best position if found
                if best_pos:
                    pos, transport_mode = best_pos
                    new_solution.insert_location(day, pos, hawker_idx, transport_mode)
                    break
        
        # Add dinner if missing
        if not has_dinner:
            # Try each hawker until one can be inserted during dinner window
            for hawker_idx, _ in hawkers:
                # Try to identify the best position for dinner
                best_pos = None
                best_time_diff = float('inf')
                
                for pos in range(1, len(new_solution.routes[day]) + 1):
                    # Try both transit and drive
                    for transport_mode in ["transit", "drive"]:
                        if new_solution.is_feasible_insertion(day, pos, hawker_idx, transport_mode):
                            # Clone solution to test insertion
                            test_sol = new_solution.clone()
                            test_sol.insert_location(day, pos, hawker_idx, transport_mode)
                            
                            # Get the actual arrival time
                            arrival_time = test_sol.routes[day][pos][1]
                            
                            # Calculate how close it is to ideal dinner time (6:30 PM)
                            ideal_dinner = problem.DINNER_START + 90  # 6:30 PM
                            time_diff = abs(arrival_time - ideal_dinner)
                            
                            if time_diff < best_time_diff:
                                best_time_diff = time_diff
                                best_pos = (pos, transport_mode)
                
                # Insert dinner hawker at the best position if found
                if best_pos:
                    pos, transport_mode = best_pos
                    new_solution.insert_location(day, pos, hawker_idx, transport_mode)
                    break
    
    # Next, add high-value attractions
    # Get unvisited attractions ordered by value
    visited_attractions = new_solution.get_visited_attractions()
    
    attractions = []
    for i in range(problem.num_locations):
        if problem.locations[i]["type"] == "attraction" and i not in visited_attractions:
            satisfaction = problem.locations[i].get("satisfaction", 0)
            cost = problem.locations[i].get("entrance_fee", 1)
            duration = problem.locations[i].get("duration", 60)
            
            # Calculate value ratio (higher is better)
            value_ratio = satisfaction / (cost + duration/60)
            attractions.append((i, value_ratio))
    
    # Sort by value ratio (highest first)
    attractions.sort(key=lambda x: x[1], reverse=True)
    
    # Try to add up to 5 attractions
    added_count = 0
    max_to_add = min(5, len(attractions))
    
    for attr_idx, _ in attractions:
        if added_count >= max_to_add:
            break
        
        # For each attraction, try to find the best insertion point across all days
        best_insertion = None
        best_cost_increase = float('inf')
        
        for day in range(new_solution.num_days):
            # Try inserting at each position between lunch and dinner
            for pos in range(1, len(new_solution.routes[day])):
                # Check the time of this position and next position
                _, arrival_time, _, _ = new_solution.routes[day][pos]
                
                # Skip if we're already past lunch (try to insert after lunch)
                if arrival_time < problem.LUNCH_END:
                    continue
                
                # Skip if we're already at dinner time
                if arrival_time >= problem.DINNER_START:
                    break
                
                # Try both transit and drive
                for transport_mode in ["transit", "drive"]:
                    if new_solution.is_feasible_insertion(day, pos+1, attr_idx, transport_mode):
                        # Clone solution to test insertion
                        test_sol = new_solution.clone()
                        test_sol.insert_location(day, pos+1, attr_idx, transport_mode)
                        
                        # Calculate cost increase
                        original_cost = new_solution.get_total_cost()
                        new_cost = test_sol.get_total_cost()
                        cost_increase = new_cost - original_cost
                        
                        if cost_increase < best_cost_increase:
                            best_cost_increase = cost_increase
                            best_insertion = (day, pos+1, transport_mode)
        
        # Insert the attraction at the best position if found
        if best_insertion:
            day, pos, transport_mode = best_insertion
            new_solution.insert_location(day, pos, attr_idx, transport_mode)
            added_count += 1
    
    # Ensure each day ends at hotel
    for day in range(new_solution.num_days):
        route = new_solution.routes[day]
        
        # Add hotel at the end if not already there
        if route[-1][0] != 0:
            hotel_idx = 0
            position = len(route)
            
            # Try both transport modes
            if new_solution.is_feasible_insertion(day, position, hotel_idx, "transit"):
                new_solution.insert_location(day, position, hotel_idx, "transit")
            elif new_solution.is_feasible_insertion(day, position, hotel_idx, "drive"):
                new_solution.insert_location(day, position, hotel_idx, "drive")
    
    return new_solution

def repair_regret_insertion(problem, solution):
    """
    Repair solution using a regret-based insertion heuristic
    
    Args:
        problem: TravelItineraryProblem instance
        solution: VRPSolution instance
        
    Returns:
        VRPSolution: Modified solution
    """
    # Make a copy of the solution to avoid modifying the original
    new_solution = solution.clone()
    
    # First, ensure each day has lunch and dinner (same as greedy repair for meals)
    for day in range(new_solution.num_days):
        # Check for lunch and dinner
        has_lunch, has_dinner = new_solution.has_lunch_and_dinner(day)
        
        # Get hawker centers ordered by rating
        hawkers = [(i, problem.locations[i].get("rating", 0)) 
                 for i in range(problem.num_locations) 
                 if problem.locations[i]["type"] == "hawker"]
        hawkers.sort(key=lambda x: x[1], reverse=True)  # Sort by rating (highest first)
        
        # Add lunch if missing
        if not has_lunch:
            # Try to insert during lunch window (similar to greedy repair)
            for hawker_idx, _ in hawkers:
                # Try to identify the best position for lunch
                best_pos = None
                best_time_diff = float('inf')
                
                for pos in range(1, len(new_solution.routes[day]) + 1):
                    # Try both transit and drive
                    for transport_mode in ["transit", "drive"]:
                        if new_solution.is_feasible_insertion(day, pos, hawker_idx, transport_mode):
                            # Clone solution to test insertion
                            test_sol = new_solution.clone()
                            test_sol.insert_location(day, pos, hawker_idx, transport_mode)
                            
                            # Get the actual arrival time
                            arrival_time = test_sol.routes[day][pos][1]
                            
                            # Check if it's within lunch window
                            if arrival_time >= problem.LUNCH_START and arrival_time <= problem.LUNCH_END:
                                # Calculate how close it is to ideal lunch time (12:30 PM)
                                ideal_lunch = problem.LUNCH_START + 90  # 12:30 PM
                                time_diff = abs(arrival_time - ideal_lunch)
                                
                                if time_diff < best_time_diff:
                                    best_time_diff = time_diff
                                    best_pos = (pos, transport_mode)
                
                # Insert lunch hawker at the best position if found
                if best_pos:
                    pos, transport_mode = best_pos
                    new_solution.insert_location(day, pos, hawker_idx, transport_mode)
                    break
        
        # Add dinner if missing
        if not has_dinner:
            # Try to insert during dinner window (similar to greedy repair)
            for hawker_idx, _ in hawkers:
                # Try to identify the best position for dinner
                best_pos = None
                best_time_diff = float('inf')
                
                for pos in range(1, len(new_solution.routes[day]) + 1):
                    # Try both transit and drive
                    for transport_mode in ["transit", "drive"]:
                        if new_solution.is_feasible_insertion(day, pos, hawker_idx, transport_mode):
                            # Clone solution to test insertion
                            test_sol = new_solution.clone()
                            test_sol.insert_location(day, pos, hawker_idx, transport_mode)
                            
                            # Get the actual arrival time
                            arrival_time = test_sol.routes[day][pos][1]
                            
                            # Check if it's within dinner window
                            if arrival_time >= problem.DINNER_START and arrival_time <= problem.DINNER_END:
                                # Calculate how close it is to ideal dinner time (6:30 PM)
                                ideal_dinner = problem.DINNER_START + 90  # 6:30 PM
                                time_diff = abs(arrival_time - ideal_dinner)
                                
                                if time_diff < best_time_diff:
                                    best_time_diff = time_diff
                                    best_pos = (pos, transport_mode)
                
                # Insert dinner hawker at the best position if found
                if best_pos:
                    pos, transport_mode = best_pos
                    new_solution.insert_location(day, pos, hawker_idx, transport_mode)
                    break
    
    # Now apply regret-based insertion for attractions
    # Get unvisited attractions
    visited_attractions = new_solution.get_visited_attractions()
    
    unvisited_attractions = [i for i in range(problem.num_locations) 
                           if problem.locations[i]["type"] == "attraction" 
                           and i not in visited_attractions]
    
    # Apply regret insertion until no more attractions can be inserted or budget is reached
    budget_limit = problem.budget * 0.95  # 95% of budget to leave some slack
    
    while unvisited_attractions and new_solution.get_total_cost() < budget_limit:
        # Calculate regret values for each unvisited attraction
        regret_values = []
        
        for attr_idx in unvisited_attractions:
            # Find the best and second-best insertion positions
            insertion_costs = []
            
            for day in range(new_solution.num_days):
                for pos in range(1, len(new_solution.routes[day])):
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
                                normalized_cost = -satisfaction_increase  # Free satisfaction!
                            
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
                regret = 1000
                best_insertion = insertion_costs[0]
            else:
                # No feasible insertion, skip this attraction
                continue
            
            regret_values.append((attr_idx, regret, best_insertion))
        
        # If no regret values, we're done
        if not regret_values:
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
    
    # Ensure each day ends at hotel
    for day in range(new_solution.num_days):
        route = new_solution.routes[day]
        
        # Add hotel at the end if not already there
        if route[-1][0] != 0:
            hotel_idx = 0
            position = len(route)
            
            # Try both transport modes
            if new_solution.is_feasible_insertion(day, position, hotel_idx, "transit"):
                new_solution.insert_location(day, position, hotel_idx, "transit")
            elif new_solution.is_feasible_insertion(day, position, hotel_idx, "drive"):
                new_solution.insert_location(day, position, hotel_idx, "drive")
    
    return new_solution

def repair_time_based_insertion(problem, solution):
    """
    Repair solution focusing on time window efficiency,
    with enhanced dinner scheduling prioritization
    
    Args:
        problem: TravelItineraryProblem instance
        solution: VRPSolution instance
        
    Returns:
        VRPSolution: Modified solution
    """
    # Make a copy of the solution to avoid modifying the original
    new_solution = solution.clone()
    
    # First, handle lunch and dinner with higher priority for dinner
    for day in range(new_solution.num_days):
        # Check for lunch and dinner
        has_lunch, has_dinner = new_solution.has_lunch_and_dinner(day)
        
        # Get hawker centers ordered by rating
        hawkers = [(i, problem.locations[i].get("rating", 0)) 
                 for i in range(problem.num_locations) 
                 if problem.locations[i]["type"] == "hawker"]
        hawkers.sort(key=lambda x: x[1], reverse=True)  # Sort by rating (highest first)
        
        # ---------------
        # PRIORITIZE DINNER FIRST (this is the key change)
        # ---------------
        # Add dinner if missing - HIGHER PRIORITY than lunch
        if not has_dinner:
            # Allow a broader dinner window for initial insertion to increase chance of success
            early_dinner_time = problem.DINNER_START - 30  # 30 minutes earlier than standard
            extended_dinner_end = problem.DINNER_END + 30  # 30 minutes later than standard
            
            # Try multiple positions within the dinner window
            attempted_positions = []
            
            # First try at ideal dinner time (6:30 PM)
            ideal_dinner = problem.DINNER_START + 90  # 6:30 PM
            
            # Get the position where dinner should be inserted
            target_pos = 1  # Default to after hotel
            for pos, (_, arrival, _, _) in enumerate(new_solution.routes[day]):
                if arrival > ideal_dinner:
                    break
                target_pos = pos + 1
                
            attempted_positions.append(target_pos)
            
            # Try early dinner
            early_pos = 1
            for pos, (_, arrival, _, _) in enumerate(new_solution.routes[day]):
                if arrival > early_dinner_time:
                    break
                early_pos = pos + 1
                
            if early_pos not in attempted_positions:
                attempted_positions.append(early_pos)
            
            # Try late dinner
            late_pos = 1
            for pos, (_, arrival, _, _) in enumerate(new_solution.routes[day]):
                if arrival > extended_dinner_end - 60:  # Allow at least 60 min for dinner
                    break
                late_pos = pos + 1
                
            if late_pos not in attempted_positions and late_pos != target_pos:
                attempted_positions.append(late_pos)
            
            # Add end position if not already included
            if len(new_solution.routes[day]) not in attempted_positions:
                attempted_positions.append(len(new_solution.routes[day]))
            
            # Try to insert a hawker at each position, prioritizing the target position
            dinner_inserted = False
            
            # First try the target (ideal) position
            for hawker_idx, _ in hawkers:
                # Skip hawkers already used for lunch today
                is_used_for_lunch = False
                for loc, arrival, _, _ in new_solution.routes[day]:
                    if loc == hawker_idx and arrival >= problem.LUNCH_START and arrival <= problem.LUNCH_END:
                        is_used_for_lunch = True
                        break
                
                if is_used_for_lunch:
                    continue
                
                # Try both transit and drive at the ideal position
                for transport_mode in ["transit", "drive"]:
                    if new_solution.is_feasible_insertion(day, target_pos, hawker_idx, transport_mode):
                        new_solution.insert_location(day, target_pos, hawker_idx, transport_mode)
                        dinner_inserted = True
                        break
                
                if dinner_inserted:
                    break
            
            # If not successful, try all other positions
            if not dinner_inserted:
                for pos in attempted_positions:
                    if pos == target_pos:  # Already tried this one
                        continue
                        
                    for hawker_idx, _ in hawkers:
                        # Skip hawkers already used for lunch today
                        is_used_for_lunch = False
                        for loc, arrival, _, _ in new_solution.routes[day]:
                            if loc == hawker_idx and arrival >= problem.LUNCH_START and arrival <= problem.LUNCH_END:
                                is_used_for_lunch = True
                                break
                        
                        if is_used_for_lunch:
                            continue
                        
                        # Try both transit and drive
                        for transport_mode in ["transit", "drive"]:
                            if new_solution.is_feasible_insertion(day, pos, hawker_idx, transport_mode):
                                # Test if this would be in dinner window
                                test_sol = new_solution.clone()
                                test_sol.insert_location(day, pos, hawker_idx, transport_mode)
                                arrival = test_sol.routes[day][pos][1]
                                
                                # Use extended dinner window to improve chances
                                if arrival >= early_dinner_time and arrival <= extended_dinner_end:
                                    new_solution.insert_location(day, pos, hawker_idx, transport_mode)
                                    dinner_inserted = True
                                    break
                        
                        if dinner_inserted:
                            break
                    
                    if dinner_inserted:
                        break
            
            # Last resort: Try to force a dinner hawker toward end of day
            if not dinner_inserted:
                # Get the latest feasible position
                latest_pos = len(new_solution.routes[day])
                
                # Try with less strict constraints
                for hawker_idx, _ in hawkers:
                    if not any(loc == hawker_idx for loc, _, _, _ in new_solution.routes[day]):
                        for transport_mode in ["transit", "drive"]:
                            # We'll try to insert anyway and adjust times if needed
                            if new_solution.insert_location(day, latest_pos, hawker_idx, transport_mode):
                                dinner_inserted = True
                                break
                    
                    if dinner_inserted:
                        break
        
        # Add lunch if missing
        if not has_lunch:
            # Find optimal lunch window
            lunch_time = problem.LUNCH_START + 60  # Target 12:00 PM
            
            # Get the position where lunch should be inserted
            target_pos = 1  # Default to after hotel
            for pos, (_, arrival, _, _) in enumerate(new_solution.routes[day]):
                if arrival > lunch_time:
                    break
                target_pos = pos + 1
            
            # Try to insert a hawker at this position
            for hawker_idx, _ in hawkers:
                # Skip hawkers already used for dinner today
                is_used_for_dinner = False
                for loc, arrival, _, _ in new_solution.routes[day]:
                    if loc == hawker_idx and arrival >= problem.DINNER_START and arrival <= problem.DINNER_END:
                        is_used_for_dinner = True
                        break
                
                if is_used_for_dinner:
                    continue
                
                # Try both transit and drive
                for transport_mode in ["transit", "drive"]:
                    if new_solution.is_feasible_insertion(day, target_pos, hawker_idx, transport_mode):
                        new_solution.insert_location(day, target_pos, hawker_idx, transport_mode)
                        has_lunch = True
                        break
                if has_lunch:
                    break
            
            # If couldn't insert at target position, try all positions
            if not has_lunch:
                for pos in range(1, len(new_solution.routes[day]) + 1):
                    for hawker_idx, _ in hawkers:
                        for transport_mode in ["transit", "drive"]:
                            if new_solution.is_feasible_insertion(day, pos, hawker_idx, transport_mode):
                                new_solution.insert_location(day, pos, hawker_idx, transport_mode)
                                has_lunch = True
                                break
                        if has_lunch:
                            break
                    if has_lunch:
                        break
    
    # Next add attractions based on time efficiency
    # Get unvisited attractions ordered by time efficiency
    visited_attractions = new_solution.get_visited_attractions()
    
    attractions = []
    for i in range(problem.num_locations):
        if problem.locations[i]["type"] == "attraction" and i not in visited_attractions:
            satisfaction = problem.locations[i].get("satisfaction", 0)
            duration = problem.locations[i].get("duration", 60)
            
            # Calculate time efficiency (higher is better)
            time_efficiency = satisfaction / duration
            attractions.append((i, time_efficiency))
    
    # Sort by time efficiency (highest first)
    attractions.sort(key=lambda x: x[1], reverse=True)
    
    # Try to insert attractions in time gaps between meals
    for day in range(new_solution.num_days):
        route = new_solution.routes[day]
        
        # Find lunch and dinner positions
        lunch_pos = None
        dinner_pos = None
        
        for pos, (loc_idx, arrival, _, _) in enumerate(route):
            if problem.locations[loc_idx]["type"] == "hawker":
                if arrival >= problem.LUNCH_START and arrival <= problem.LUNCH_END:
                    lunch_pos = pos
                elif arrival >= problem.DINNER_START and arrival <= problem.DINNER_END:
                    dinner_pos = pos
        
        # If we have both lunch and dinner, try inserting attractions between them
        if lunch_pos is not None and dinner_pos is not None and lunch_pos < dinner_pos:
            # Try to insert after lunch and before dinner
            target_pos = lunch_pos + 1
            
            # Try each attraction
            for attr_idx, _ in attractions:
                if attr_idx in visited_attractions:
                    continue
                
                # Try both transit and drive
                for transport_mode in ["transit", "drive"]:
                    if new_solution.is_feasible_insertion(day, target_pos, attr_idx, transport_mode):
                        new_solution.insert_location(day, target_pos, attr_idx, transport_mode)
                        visited_attractions.add(attr_idx)
                        
                        # Update target position
                        target_pos += 1
                        break
        
        # Try to insert attractions in morning (before lunch) if there's time
        if lunch_pos is not None and lunch_pos > 1:
            target_pos = 1  # After hotel, before lunch
            
            # Try each attraction
            for attr_idx, _ in attractions:
                if attr_idx in visited_attractions:
                    continue
                
                # Try both transit and drive
                for transport_mode in ["transit", "drive"]:
                    if new_solution.is_feasible_insertion(day, target_pos, attr_idx, transport_mode):
                        new_solution.insert_location(day, target_pos, attr_idx, transport_mode)
                        visited_attractions.add(attr_idx)
                        
                        # Update target position
                        target_pos += 1
                        break
    
    # Ensure each day ends at hotel
    for day in range(new_solution.num_days):
        route = new_solution.routes[day]
        
        # Add hotel at the end if not already there
        if route[-1][0] != 0:
            hotel_idx = 0
            position = len(route)
            
            # Try both transport modes
            if new_solution.is_feasible_insertion(day, position, hotel_idx, "transit"):
                new_solution.insert_location(day, position, hotel_idx, "transit")
            elif new_solution.is_feasible_insertion(day, position, hotel_idx, "drive"):
                new_solution.insert_location(day, position, hotel_idx, "drive")
    
    return new_solution