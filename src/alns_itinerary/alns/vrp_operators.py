"""
Optimized destroy and repair operators for the VRP-based travel itinerary optimizer.
These operators are refined to reduce constraint violations and improve solution quality.
"""

import random
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

#----------------
# Destroy Operators
#----------------

def destroy_targeted_subsequence(problem, solution):
    """
    Improved version of subsequence removal that preserves meal timing and avoids
    creating infeasible solutions. Focuses on time periods with low impact on constraints.
    
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
    for pos in range(1, len(route) - 1):  # Skip hotel start/end
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

def destroy_worst_attractions(problem, solution):
    """
    Remove attractions with the worst satisfaction-to-cost ratio.
    This version is more selective and avoids disrupting meal scheduling.
    
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
                
                # Check if removal would disrupt meal timing
                # Avoid removing attractions that are the only ones between meals
                is_critical = False
                if lunch_pos is not None and dinner_pos is not None:
                    if lunch_pos < pos < dinner_pos:
                        attractions_in_period = sum(1 for p, (l, _, _, _) in enumerate(route) 
                                                if lunch_pos < p < dinner_pos and problem.locations[l]["type"] == "attraction")
                        if attractions_in_period <= 1:
                            is_critical = True
                
                if not is_critical:
                    attraction_visits.append((day, pos, loc_idx, value_ratio))
    
    # Sort by value ratio (ascending - worst first)
    attraction_visits.sort(key=lambda x: x[3])
    
    # Remove up to 25% of worst attractions, but at least 1 if any are visited
    num_to_remove = max(1, int(len(attraction_visits) * 0.25))
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

def destroy_time_window_violations(problem, solution):
    """
    Remove locations that cause time window violations.
    This is a key operator for maintaining feasibility.
    
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

def destroy_expensive_attractions(problem, solution):
    """
    Focus on removing expensive attractions to improve budget utilization.
    Balances cost reduction with maintaining satisfaction.
    
    Args:
        problem: TravelItineraryProblem instance
        solution: VRPSolution instance
        
    Returns:
        VRPSolution: Modified solution
    """
    # Make a copy of the solution to avoid modifying the original
    new_solution = solution.clone()
    
    # Get current solution cost
    total_cost = new_solution.get_total_cost()
    
    # If already under budget or very close, don't destroy expensive attractions
    if total_cost <= problem.budget * 1.05:
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
                    cost_ratio = cost / satisfaction
                else:
                    cost_ratio = float('inf')
                
                expensive_attractions.append((day, pos, loc_idx, cost, cost_ratio))
    
    # Sort by cost ratio (descending - worst first)
    expensive_attractions.sort(key=lambda x: x[4], reverse=True)
    
    # Remove up to 30% of expensive attractions
    num_to_remove = max(1, int(len(expensive_attractions) * 0.3))
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

def destroy_selected_day(problem, solution):
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
    # Make a copy of the solution to avoid modifying the original
    new_solution = solution.clone()
    
    # Select a random day
    day = random.randint(0, new_solution.num_days - 1)
    route = new_solution.routes[day]
    
    # Need at least 4 locations to destroy (hotel -> loc1 -> loc2 -> hotel)
    if len(route) < 4:
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
    preserve_hawkers = random.random() < 0.7
    
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

def repair_regret_insertion(problem, solution):
    """
    Enhanced regret-based insertion with improved meal scheduling.
    
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
        
        # Add dinner first (prioritize dinner scheduling)
        if not has_dinner:
            # Try each hawker until one can be inserted during dinner window
            for hawker_idx, _ in hawkers:
                # Try to identify the best position for dinner
                best_pos = None
                best_time_diff = float('inf')
                
                # Aim for 6:30 PM (ideal dinner time)
                ideal_dinner = problem.DINNER_START + 90
                
                # Try each position
                for pos in range(1, len(new_solution.routes[day]) + 1):
                    # Try both transit and drive
                    for transport_mode in ["transit", "drive"]:
                        if new_solution.is_feasible_insertion(day, pos, hawker_idx, transport_mode):
                            # Clone solution to test insertion
                            test_sol = new_solution.clone()
                            test_sol.insert_location(day, pos, hawker_idx, transport_mode)
                            
                            # Get the actual arrival time
                            arrival_time = test_sol.routes[day][pos][1]
                            
                            # Check if within dinner window or close to it
                            if (arrival_time >= problem.DINNER_START - 30 and 
                                arrival_time <= problem.DINNER_END + 30):
                                # Calculate how close to ideal
                                time_diff = abs(arrival_time - ideal_dinner)
                                
                                if time_diff < best_time_diff:
                                    best_time_diff = time_diff
                                    best_pos = (pos, transport_mode)
                
                # Insert dinner hawker at the best position if found
                if best_pos:
                    pos, transport_mode = best_pos
                    new_solution.insert_location(day, pos, hawker_idx, transport_mode, 'Dinner')
                    break
        
        # Add lunch second
        if not has_lunch:
            # Try each hawker until one can be inserted during lunch window
            for hawker_idx, _ in hawkers:
                # Skip hawkers already used for dinner today
                is_used_for_dinner = False
                for loc, arrival, _, _ in new_solution.routes[day]:
                    if loc == hawker_idx and arrival >= problem.DINNER_START and arrival <= problem.DINNER_END:
                        is_used_for_dinner = True
                        break
                
                if is_used_for_dinner:
                    continue
                
                # Try to identify the best position for lunch
                best_pos = None
                best_time_diff = float('inf')
                
                # Aim for 12:30 PM (ideal lunch time)
                ideal_lunch = problem.LUNCH_START + 90
                
                # Try each position
                for pos in range(1, len(new_solution.routes[day]) + 1):
                    # Try both transit and drive
                    for transport_mode in ["transit", "drive"]:
                        if new_solution.is_feasible_insertion(day, pos, hawker_idx, transport_mode):
                            # Clone solution to test insertion
                            test_sol = new_solution.clone()
                            test_sol.insert_location(day, pos, hawker_idx, transport_mode)
                            
                            # Get the actual arrival time
                            arrival_time = test_sol.routes[day][pos][1]
                            
                            # Check if within lunch window or close to it
                            if (arrival_time >= problem.LUNCH_START - 30 and 
                                arrival_time <= problem.LUNCH_END + 30):
                                # Calculate how close to ideal
                                time_diff = abs(arrival_time - ideal_lunch)
                                
                                if time_diff < best_time_diff:
                                    best_time_diff = time_diff
                                    best_pos = (pos, transport_mode)
                
                # Insert lunch hawker at the best position if found
                if best_pos:
                    pos, transport_mode = best_pos
                    new_solution.insert_location(day, pos, hawker_idx, transport_mode, 'Lunch')
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
                                normalized_cost = -satisfaction_increase * 2  # Double bonus for free satisfaction
                            
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
        if len(route) == 0 or route[-1][0] != 0:
            hotel_idx = 0
            position = len(route)
            
            # Try both transport modes
            inserted = False
            for transport_mode in ["transit", "drive"]:
                if new_solution.is_feasible_insertion(day, position, hotel_idx, transport_mode):
                    new_solution.insert_location(day, position, hotel_idx, transport_mode)
                    inserted = True
                    break
            
            # If we couldn't insert with feasible insertion, try direct insertion as last resort
            if not inserted and len(route) > 0:
                # Get the last location
                last_loc, _, last_departure, _ = route[-1]
                
                # Try to calculate direct return
                transport_hour = problem.get_transport_hour(last_departure)
                try:
                    transport_key = (problem.locations[last_loc]["name"], 
                                    problem.locations[hotel_idx]["name"], 
                                    transport_hour)
                    
                    # Choose fastest transport mode
                    fastest_mode = "transit"
                    fastest_time = float('inf')
                    
                    for mode in ["transit", "drive"]:
                        if mode in problem.transport_matrix.get(transport_key, {}):
                            duration = problem.transport_matrix[transport_key][mode]["duration"]
                            if duration < fastest_time:
                                fastest_time = duration
                                fastest_mode = mode
                    
                    # Force insert hotel return
                    new_solution.insert_location(day, position, hotel_idx, fastest_mode)
                except (KeyError, TypeError):
                    # If we can't find transport data, use drive as default
                    new_solution.insert_location(day, position, hotel_idx, "drive")
    
    return new_solution

def repair_time_based_insertion(problem, solution):
    """
    Enhanced repair focusing on optimal time slots and meal scheduling.
    Prioritizes dinner scheduling first, then lunch, then attractions in
    appropriate time gaps.
    
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
        
        # PRIORITIZE DINNER FIRST
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
                        new_solution.insert_location(day, target_pos, hawker_idx, transport_mode, 'Dinner')
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
                                    new_solution.insert_location(day, pos, hawker_idx, transport_mode, 'Dinner')
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
                            if new_solution.insert_location(day, latest_pos, hawker_idx, transport_mode, 'Dinner'):
                                dinner_inserted = True
                                break
                    
                    if dinner_inserted:
                        break
        
        # Add lunch if missing
        if not has_lunch:
            # Find optimal lunch window
            lunch_time = problem.LUNCH_START + 60  # Target 12:00 PM
            extended_lunch_start = problem.LUNCH_START - 30  # Extended window
            extended_lunch_end = problem.LUNCH_END + 30
            
            # Try multiple positions for lunch
            lunch_positions = []
            
            # Add the ideal lunch position
            target_pos = 1  # Default to after hotel
            for pos, (_, arrival, _, _) in enumerate(new_solution.routes[day]):
                if arrival > lunch_time:
                    break
                target_pos = pos + 1
            
            lunch_positions.append(target_pos)
            
            # Try early lunch position
            early_pos = 1
            for pos, (_, arrival, _, _) in enumerate(new_solution.routes[day]):
                if arrival > extended_lunch_start:
                    break
                early_pos = pos + 1
            
            if early_pos != target_pos:
                lunch_positions.append(early_pos)
            
            # Try all positions systematically
            lunch_inserted = False
            
            # First try the ideal positions in order
            for pos in lunch_positions:
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
                        if new_solution.is_feasible_insertion(day, pos, hawker_idx, transport_mode):
                            # Test if this would be in lunch window
                            test_sol = new_solution.clone()
                            test_sol.insert_location(day, pos, hawker_idx, transport_mode)
                            arrival = test_sol.routes[day][pos][1]
                            
                            # Use extended lunch window to improve chances
                            if arrival >= extended_lunch_start and arrival <= extended_lunch_end:
                                new_solution.insert_location(day, pos, hawker_idx, transport_mode, 'Lunch')
                                lunch_inserted = True
                                break
                    
                    if lunch_inserted:
                        break
                
                if lunch_inserted:
                    break
            
            # If still not inserted, try any position
            if not lunch_inserted:
                for pos in range(1, len(new_solution.routes[day]) + 1):
                    for hawker_idx, _ in hawkers:
                        # Skip hawkers already used for dinner
                        is_used_for_dinner = False
                        for loc, arrival, _, _ in new_solution.routes[day]:
                            if loc == hawker_idx and arrival >= problem.DINNER_START and arrival <= problem.DINNER_END:
                                is_used_for_dinner = True
                                break
                        
                        if is_used_for_dinner:
                            continue
                            
                        for transport_mode in ["transit", "drive"]:
                            if new_solution.is_feasible_insertion(day, pos, hawker_idx, transport_mode):
                                new_solution.insert_location(day, pos, hawker_idx, transport_mode, 'Lunch')
                                lunch_inserted = True
                                break
                        if lunch_inserted:
                            break
                    if lunch_inserted:
                        break
    
    # Next add attractions based on a combined efficiency metric
    # Get unvisited attractions ordered by value efficiency
    visited_attractions = new_solution.get_visited_attractions()
    
    # Calculate budget remaining
    budget_left = problem.budget - new_solution.get_total_cost()
    
    attractions = []
    for i in range(problem.num_locations):
        if problem.locations[i]["type"] == "attraction" and i not in visited_attractions:
            satisfaction = problem.locations[i].get("satisfaction", 0)
            duration = problem.locations[i].get("duration", 60)
            cost = problem.locations[i].get("entrance_fee", 1)
            
            # Calculate combined efficiency (higher is better)
            # Consider both time efficiency and cost efficiency
            if cost > 0:
                value_efficiency = satisfaction / (cost * 0.5 + duration / 60 * 0.5)
            else:
                value_efficiency = satisfaction / (duration / 60)
                
            # Bonus for affordable attractions when budget is tight
            if budget_left < 100 and cost < 30:
                value_efficiency *= 1.2
                
            attractions.append((i, value_efficiency))
    
    # Sort by efficiency (highest first)
    attractions.sort(key=lambda x: x[1], reverse=True)
    
    # Try to insert attractions in optimal time gaps
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
        
        # First priority: Insert between lunch and dinner (afternoon)
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
        
        # Second priority: Insert in morning (before lunch)
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
        
        # Third priority: Insert in evening (after dinner)
        if dinner_pos is not None and dinner_pos < len(route) - 1:
            target_pos = dinner_pos + 1
            
            # Try each attraction
            for attr_idx, _ in attractions:
                if attr_idx in visited_attractions:
                    continue
                
                # Try both transit and drive
                for transport_mode in ["transit", "drive"]:
                    if new_solution.is_feasible_insertion(day, target_pos, attr_idx, transport_mode):
                        # Make sure this doesn't push us past our end time
                        test_sol = new_solution.clone()
                        test_sol.insert_location(day, target_pos, attr_idx, transport_mode)
                        
                        # Verify we can still return to hotel in time
                        if test_sol.is_feasible():
                            new_solution.insert_location(day, target_pos, attr_idx, transport_mode)
                            visited_attractions.add(attr_idx)
                            
                            # Update target position
                            target_pos += 1
                            break
    
    # Ensure each day ends at hotel
    for day in range(new_solution.num_days):
        route = new_solution.routes[day]
        
        # Add hotel at the end if not already there
        if len(route) == 0 or route[-1][0] != 0:
            hotel_idx = 0
            position = len(route)
            
            # Try both transport modes
            inserted = False
            for transport_mode in ["transit", "drive"]:
                if new_solution.is_feasible_insertion(day, position, hotel_idx, transport_mode):
                    new_solution.insert_location(day, position, hotel_idx, transport_mode)
                    inserted = True
                    break
            
            # If we couldn't insert with feasible insertion, try direct insertion as last resort
            if not inserted and len(route) > 0:
                # Get the last location
                last_loc, _, last_departure, _ = route[-1]
                
                # Try to calculate direct return
                transport_hour = problem.get_transport_hour(last_departure)
                try:
                    transport_key = (problem.locations[last_loc]["name"], 
                                    problem.locations[hotel_idx]["name"], 
                                    transport_hour)
                    
                    # Choose fastest transport mode
                    fastest_mode = "transit"
                    fastest_time = float('inf')
                    
                    for mode in ["transit", "drive"]:
                        if mode in problem.transport_matrix.get(transport_key, {}):
                            duration = problem.transport_matrix[transport_key][mode]["duration"]
                            if duration < fastest_time:
                                fastest_time = duration
                                fastest_mode = mode
                    
                    # Force insert hotel return
                    new_solution.insert_location(day, position, hotel_idx, fastest_mode)
                except (KeyError, TypeError):
                    # If we can't find transport data, use drive as default
                    new_solution.insert_location(day, position, hotel_idx, "drive")
    
    return new_solution

def repair_balanced_solution(problem, solution):
    """
    Repair operator that focuses on creating a balanced solution across days,
    distributing attractions evenly and ensuring proper meal scheduling.
    
    Args:
        problem: TravelItineraryProblem instance
        solution: VRPSolution instance
        
    Returns:
        VRPSolution: Modified solution
    """
    # Make a copy of the solution to avoid modifying the original
    new_solution = solution.clone()
    
    # First ensure each day has lunch and dinner (similar to other repair operators)
    for day in range(new_solution.num_days):
        has_lunch, has_dinner = new_solution.has_lunch_and_dinner(day)
        
        # Get hawker centers ordered by rating
        hawkers = [(i, problem.locations[i].get("rating", 0)) 
                 for i in range(problem.num_locations) 
                 if problem.locations[i]["type"] == "hawker"]
        hawkers.sort(key=lambda x: x[1], reverse=True)
        
        # Add dinner first (similar to time_based_insertion)
        if not has_dinner:
            # Try each hawker until one can be inserted during dinner window
            for hawker_idx, _ in hawkers:
                dinner_inserted = False
                for pos in range(1, len(new_solution.routes[day]) + 1):
                    for transport_mode in ["transit", "drive"]:
                        if new_solution.is_feasible_insertion(day, pos, hawker_idx, transport_mode):
                            new_solution.insert_location(day, pos, hawker_idx, transport_mode, 'Dinner')
                            dinner_inserted = True
                            break
                    if dinner_inserted:
                        break
                if dinner_inserted:
                    break
        
        # Add lunch second
        if not has_lunch:
            # Try each hawker until one can be inserted during lunch window
            for hawker_idx, _ in hawkers:
                # Skip hawkers already used for dinner
                is_used_for_dinner = False
                for loc, arrival, _, _ in new_solution.routes[day]:
                    if loc == hawker_idx and arrival >= problem.DINNER_START and arrival <= problem.DINNER_END:
                        is_used_for_dinner = True
                        break
                
                if is_used_for_dinner:
                    continue
                    
                lunch_inserted = False
                for pos in range(1, len(new_solution.routes[day]) + 1):
                    for transport_mode in ["transit", "drive"]:
                        if new_solution.is_feasible_insertion(day, pos, hawker_idx, transport_mode):
                            new_solution.insert_location(day, pos, hawker_idx, transport_mode, 'Lunch')
                            lunch_inserted = True
                            break
                    if lunch_inserted:
                        break
                if lunch_inserted:
                    break
    
    # Get unvisited attractions
    visited_attractions = new_solution.get_visited_attractions()
    unvisited_attractions = [i for i in range(problem.num_locations) 
                           if problem.locations[i]["type"] == "attraction" 
                           and i not in visited_attractions]
    
    # Count attractions per day
    day_attraction_counts = {}
    for day in range(new_solution.num_days):
        count = 0
        for loc, _, _, _ in new_solution.routes[day]:
            if problem.locations[loc]["type"] == "attraction":
                count += 1
        day_attraction_counts[day] = count
    
    # Sort attractions by value
    attractions_by_value = []
    for attr_idx in unvisited_attractions:
        satisfaction = problem.locations[attr_idx].get("satisfaction", 0)
        cost = problem.locations[attr_idx].get("entrance_fee", 1)
        duration = problem.locations[attr_idx].get("duration", 60)
        
        # Calculate value ratio (higher is better)
        value_ratio = satisfaction / (cost + duration/60)
        attractions_by_value.append((attr_idx, value_ratio))
    
    # Sort by value ratio (highest first)
    attractions_by_value.sort(key=lambda x: x[1], reverse=True)
    
    # Distribute attractions evenly across days, prioritizing days with fewer attractions
    while attractions_by_value and new_solution.get_total_cost() < problem.budget * 0.95:
        # Sort days by attraction count (ascending)
        sorted_days = sorted(day_attraction_counts.items(), key=lambda x: x[1])
        
        # Take the highest-value unvisited attraction
        attr_idx, _ = attractions_by_value.pop(0)
        
        # Try to insert in the day with the fewest attractions
        inserted = False
        for day, _ in sorted_days:
            # Find lunch and dinner positions
            lunch_pos = dinner_pos = None
            for pos, (loc_idx, arrival, _, _) in enumerate(new_solution.routes[day]):
                if problem.locations[loc_idx]["type"] == "hawker":
                    if arrival >= problem.LUNCH_START and arrival <= problem.LUNCH_END:
                        lunch_pos = pos
                    elif arrival >= problem.DINNER_START and arrival <= problem.DINNER_END:
                        dinner_pos = pos
            
            # Try inserting in afternoon first (between lunch and dinner)
            if lunch_pos is not None and dinner_pos is not None and lunch_pos < dinner_pos:
                target_pos = lunch_pos + 1
                for transport_mode in ["transit", "drive"]:
                    if new_solution.is_feasible_insertion(day, target_pos, attr_idx, transport_mode):
                        new_solution.insert_location(day, target_pos, attr_idx, transport_mode)
                        day_attraction_counts[day] += 1
                        inserted = True
                        break
            
            # If not inserted, try morning
            if not inserted and lunch_pos is not None:
                target_pos = 1  # After hotel
                for transport_mode in ["transit", "drive"]:
                    if new_solution.is_feasible_insertion(day, target_pos, attr_idx, transport_mode):
                        new_solution.insert_location(day, target_pos, attr_idx, transport_mode)
                        day_attraction_counts[day] += 1
                        inserted = True
                        break
            
            if inserted:
                break
        
        # If we couldn't insert in any day, stop trying with this attraction
        if not inserted:
            continue
    
    # Ensure each day ends at hotel
    for day in range(new_solution.num_days):
        route = new_solution.routes[day]
        
        # Add hotel at the end if not already there
        if len(route) == 0 or route[-1][0] != 0:
            hotel_idx = 0
            position = len(route)
            
            # Try both transport modes
            for transport_mode in ["transit", "drive"]:
                if new_solution.is_feasible_insertion(day, position, hotel_idx, transport_mode):
                    new_solution.insert_location(day, position, hotel_idx, transport_mode)
                    break
    
    return new_solution