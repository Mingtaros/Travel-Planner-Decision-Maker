import numpy as np
import random
import logging
from collections import defaultdict

logger = logging.getLogger("repair_operators")

def repair_greedy(problem, solution):
    """
    Repair the solution using a greedy approach based on satisfaction-to-cost ratio
    
    Args:
        problem: TravelItineraryProblem instance
        solution: Partially destroyed solution vector
        
    Returns:
        np.ndarray: Repaired solution
    """
    # Reshape solution into x_var and u_var
    x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                            problem.num_locations, problem.num_locations)
    u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
    
    # Calculate current cost and visited attractions
    current_cost = problem.NUM_DAYS * problem.HOTEL_COST
    visited_attractions = set()
    visited_hawkers_by_day = defaultdict(list)
    
    # Calculate current cost and track visited locations
    for day in range(problem.NUM_DAYS):
        for j in range(problem.num_transport_types):
            for k in range(problem.num_locations):
                for l in range(problem.num_locations):
                    if k == l or x_var[day, j, k, l] == 0:
                        continue
                    
                    # This route is used
                    try:
                        # Get transport cost
                        transport_hour = problem.get_transport_hour(u_var[day, k])
                        transport_key = (problem.locations[k]["name"], problem.locations[l]["name"], transport_hour)
                        transport_cost = problem.transport_matrix[transport_key][problem.transport_types[j]]["price"]
                        current_cost += transport_cost
                        
                        # Add location costs
                        if problem.locations[l]["type"] == "attraction":
                            current_cost += problem.locations[l].get("entrance_fee",0)
                            visited_attractions.add(l)
                        elif problem.locations[l]["type"] == "hawker":
                            current_cost += 10  # Assumed meal cost
                            visited_hawkers_by_day[day].append(l)
                    except KeyError:
                        # Missing transport data, skip
                        pass
    
    # Sort attractions by value
    attraction_values = []
    for i in range(problem.num_locations):
        if problem.locations[i]["type"] == "attraction" and i not in visited_attractions:
            attr = problem.locations[i]
            satisfaction = attr.get("satisfaction", 0)
            cost = attr.get("entrance_fee", 1)
            duration = attr.get("duration", 60)
            
            # Calculate value ratio (higher is better)
            value_ratio = satisfaction / (cost + duration/10)
            
            attraction_values.append((i, value_ratio))
    
    # Sort by value ratio (descending)
    attraction_values.sort(key=lambda x: x[1], reverse=True)
    
    # Get all hawkers
    hawkers = [i for i in range(problem.num_locations) 
               if problem.locations[i]["type"] == "hawker"]
    
    # For each day, check if we need to add lunch or dinner
    for day in range(problem.NUM_DAYS):
        # Check if we have at least one hawker visit for lunch and dinner
        lunch_visit = False
        dinner_visit = False
        
        for hawker_idx in visited_hawkers_by_day[day]:
            arrival_time = u_var[day, hawker_idx]
            if arrival_time >= problem.LUNCH_START and arrival_time <= problem.LUNCH_END:
                lunch_visit = True
            if arrival_time >= problem.DINNER_START and arrival_time <= problem.DINNER_END:
                dinner_visit = True
        
        # Current last location (highest time)
        if np.sum(u_var[day, :]) > 0:
            time_order = np.argsort(u_var[day, :])
            last_loc_idx = time_order[-1]
            last_time = u_var[day, last_loc_idx]
        else:
            # If no locations for this day, start from hotel
            last_loc_idx = 0
            last_time = problem.START_TIME
            u_var[day, 0] = problem.START_TIME
        
        # If no lunch visit, add one
        if not lunch_visit:
            # Find a suitable hawker
            available_hawkers = [h for h in hawkers if h not in visited_hawkers_by_day[day]]
            if available_hawkers:
                hawker_idx = random.choice(available_hawkers)
                
                # Choose transport type
                transport_choice = 0  # Default to transit
                try:
                    transport_hour = problem.get_transport_hour(last_time)
                    transport_key = (problem.locations[last_loc_idx]["name"], 
                                    problem.locations[hawker_idx]["name"], 
                                    transport_hour)
                    
                    transit_data = problem.transport_matrix[transport_key]["transit"]
                    drive_data = problem.transport_matrix[transport_key]["drive"]
                    
                    # Choose driving if significantly faster and budget allows
                    if (drive_data["duration"] < transit_data["duration"] * 0.7 and 
                        current_cost + drive_data["price"] <= problem.budget):
                        transport_choice = 1
                    
                    # Add route
                    x_var[day, transport_choice, last_loc_idx, hawker_idx] = 1
                    
                    # Calculate arrival time
                    if transport_choice == 0:
                        travel_time = transit_data["duration"]
                        current_cost += transit_data["price"]
                    else:
                        travel_time = drive_data["duration"]
                        current_cost += drive_data["price"]
                    
                    # Set lunch time
                    lunch_time = max(last_time + travel_time, problem.LUNCH_START)
                    lunch_duration = problem.locations[hawker_idx]["duration"]
                    u_var[day, hawker_idx] = lunch_time
                    
                    # Add meal cost
                    current_cost += 10
                    
                    # Update last location
                    last_loc_idx = hawker_idx
                    last_time = lunch_time + lunch_duration
                    
                    # Update visited hawkers
                    visited_hawkers_by_day[day].append(hawker_idx)
                except KeyError:
                    # Missing transport data
                    pass
        
        # Add attractions if budget allows
        remaining_budget = problem.budget - current_cost
        
        # Find the time after lunch but before dinner
        max_time_before_dinner = problem.DINNER_START - last_time
        
        # Try to add attractions
        for attr_idx, _ in attraction_values:
            if attr_idx in visited_attractions:
                continue
            
            # Check if adding this attraction would exceed budget
            attr_cost = problem.locations[attr_idx].get("entrance_fee",0)
            
            try:
                # Calculate transport cost and time
                transport_hour = problem.get_transport_hour(last_time)
                transport_key = (problem.locations[last_loc_idx]["name"], 
                                problem.locations[attr_idx]["name"], 
                                transport_hour)
                
                transit_data = problem.transport_matrix[transport_key]["transit"]
                drive_data = problem.transport_matrix[transport_key]["drive"]
                
                # Choose transport method
                transport_choice = 0
                transport_time = transit_data["duration"]
                transport_cost = transit_data["price"]
                
                # If driving saves time and budget allows, choose driving
                if (drive_data["duration"] < transit_data["duration"] * 0.7 and 
                    current_cost + drive_data["price"] + attr_cost <= problem.budget):
                    transport_choice = 1
                    transport_time = drive_data["duration"]
                    transport_cost = drive_data["price"]
                
                # Check if there's enough time before dinner
                attr_duration = problem.locations[attr_idx]["duration"]
                total_time_needed = transport_time + attr_duration
                
                if total_time_needed > max_time_before_dinner:
                    continue
                
                # Check if adding this attraction would exceed budget
                if current_cost + transport_cost + attr_cost > problem.budget:
                    continue
                
                # Add this attraction
                x_var[day, transport_choice, last_loc_idx, attr_idx] = 1
                
                # Update time and cost
                current_time = last_time + transport_time
                current_cost += transport_cost + attr_cost
                
                # Set attraction time
                u_var[day, attr_idx] = current_time
                
                # Update last location
                last_loc_idx = attr_idx
                last_time = current_time + attr_duration
                
                # Update max time before dinner
                max_time_before_dinner = problem.DINNER_START - last_time
                
                # Mark as visited
                visited_attractions.add(attr_idx)
                
                # Limit to 2-3 attractions per day
                if len([a for a in visited_attractions if a in u_var[day, :]]) >= 2:
                    break
            except KeyError:
                # Missing transport data, skip
                continue
        
        # If no dinner visit, add one
        if not dinner_visit:
            # Find a suitable hawker (different from lunch if possible)
            available_hawkers = [h for h in hawkers if h not in visited_hawkers_by_day[day]]
            if not available_hawkers and hawkers:
                # If no other options, can use a hawker that's already been visited
                available_hawkers = hawkers
            
            if available_hawkers:
                hawker_idx = random.choice(available_hawkers)
                
                # Choose transport type
                transport_choice = 0  # Default to transit
                try:
                    transport_hour = problem.get_transport_hour(last_time)
                    transport_key = (problem.locations[last_loc_idx]["name"], 
                                    problem.locations[hawker_idx]["name"], 
                                    transport_hour)
                    
                    transit_data = problem.transport_matrix[transport_key]["transit"]
                    drive_data = problem.transport_matrix[transport_key]["drive"]
                    
                    # Choose driving if significantly faster and budget allows
                    if (drive_data["duration"] < transit_data["duration"] * 0.7 and 
                        current_cost + drive_data["price"] <= problem.budget):
                        transport_choice = 1
                    
                    # Add route
                    x_var[day, transport_choice, last_loc_idx, hawker_idx] = 1
                    
                    # Calculate arrival time
                    if transport_choice == 0:
                        travel_time = transit_data["duration"]
                        current_cost += transit_data["price"]
                    else:
                        travel_time = drive_data["duration"]
                        current_cost += drive_data["price"]
                    
                    # Set dinner time
                    dinner_time = max(last_time + travel_time, problem.DINNER_START)
                    dinner_duration = problem.locations[hawker_idx]["duration"]
                    u_var[day, hawker_idx] = dinner_time
                    
                    # Add meal cost
                    current_cost += 10
                    
                    # Update last location
                    last_loc_idx = hawker_idx
                    last_time = dinner_time + dinner_duration
                    
                    # Update visited hawkers
                    visited_hawkers_by_day[day].append(hawker_idx)
                except KeyError:
                    # Missing transport data
                    pass
        
        # Ensure we return to the hotel
        try:
            # Find fastest transport method to get back
            transport_hour = problem.get_transport_hour(last_time)
            transport_key = (problem.locations[last_loc_idx]["name"], 
                           problem.locations[0]["name"], 
                           transport_hour)
            
            transit_data = problem.transport_matrix[transport_key]["transit"]
            drive_data = problem.transport_matrix[transport_key]["drive"]
            
            # Choose fastest transport method
            if drive_data["duration"] < transit_data["duration"]:
                transport_choice = 1
                return_time = drive_data["duration"]
                return_cost = drive_data["price"]
            else:
                transport_choice = 0
                return_time = transit_data["duration"]
                return_cost = transit_data["price"]
            
            # Set route back to hotel
            x_var[day, transport_choice, last_loc_idx, 0] = 1
            
            # Update cost
            current_cost += return_cost
            
            # Update hotel return time
            return_finish_time = last_time + return_time
            
            # Make sure we don't get back too late
            if return_finish_time > problem.HARD_LIMIT_END_TIME:
                # Adjust departure time to guarantee arrival within limit
                u_var[day, 0] = max(u_var[day, 0], return_finish_time)
        except KeyError:
            # Missing transport data
            pass
    
    # Flatten and return
    solution[:problem.x_shape] = x_var.flatten()
    solution[problem.x_shape:] = u_var.flatten()
    
    return solution

def repair_regret(problem, solution):
    """
    Repair solution using a regret-based approach
    Tries to add routes with high opportunity cost
    
    Args:
        problem: TravelItineraryProblem instance
        solution: Partially destroyed solution vector
        
    Returns:
        np.ndarray: Repaired solution
    """
    # Reshape solution into x_var and u_var
    x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                            problem.num_locations, problem.num_locations)
    u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
    
    # Get potential locations
    attractions = [i for i in range(problem.num_locations) 
                  if problem.locations[i]["type"] == "attraction"]
    hawkers = [i for i in range(problem.num_locations) 
              if problem.locations[i]["type"] == "hawker"]
    
    # Track visited locations
    visited_attractions = set()
    visited_hawkers_by_day = defaultdict(set)
    
    # Find currently visited locations
    for day in range(problem.NUM_DAYS):
        for j in range(problem.num_transport_types):
            for k in range(problem.num_locations):
                for l in range(problem.num_locations):
                    if x_var[day, j, k, l] > 0:
                        if problem.locations[l]["type"] == "attraction":
                            visited_attractions.add(l)
                        elif problem.locations[l]["type"] == "hawker":
                            visited_hawkers_by_day[day].add(l)
    
    # Calculate value of unvisited locations
    attraction_values = []
    for attr_idx in attractions:
        if attr_idx not in visited_attractions:
            attr = problem.locations[attr_idx]
            satisfaction = attr.get("satisfaction", 0)
            cost = attr.get("entrance_fee", 1)
            duration = attr.get("duration", 60)
            
            # Calculate value ratio
            value_ratio = satisfaction / (cost + duration/10)
            attraction_values.append((attr_idx, value_ratio))
    
    # Sort attractions by value
    attraction_values.sort(key=lambda x: x[1], reverse=True)
    
    # Repair process for each day
    for day in range(problem.NUM_DAYS):
        # Track day's status
        current_cost = problem.NUM_DAYS * problem.HOTEL_COST
        current_time = u_var[day, 0]  # Start from hotel time
        last_location = 0  # Start at hotel
        
        # Lunch and dinner check
        lunch_visit = any(hawker_time >= problem.LUNCH_START and hawker_time <= problem.LUNCH_END 
                          for hawker_time in [u_var[day, h] for h in visited_hawkers_by_day[day]])
        dinner_visit = any(hawker_time >= problem.DINNER_START and hawker_time <= problem.DINNER_END 
                           for hawker_time in [u_var[day, h] for h in visited_hawkers_by_day[day]])
        
        # Try to add lunch if missing
        if not lunch_visit:
            available_lunch_hawkers = [h for h in hawkers if h not in visited_hawkers_by_day[day]]
            if available_lunch_hawkers:
                # Choose best lunch hawker (highest rating)
                lunch_hawker = max(available_lunch_hawkers, 
                                   key=lambda x: problem.locations[x].get("rating", 0))
                
                # Add lunch route
                try:
                    # Find best transport to lunch hawker
                    transport_hour = problem.get_transport_hour(current_time)
                    transport_key = (problem.locations[last_location]["name"], 
                                     problem.locations[lunch_hawker]["name"], 
                                     transport_hour)
                    
                    transit_data = problem.transport_matrix[transport_key]["transit"]
                    drive_data = problem.transport_matrix[transport_key]["drive"]
                    
                    # Choose transport
                    transport_choice = 0  # Default transit
                    if drive_data["duration"] < transit_data["duration"] * 0.7:
                        transport_choice = 1
                    
                    transport_data = transit_data if transport_choice == 0 else drive_data
                    
                    # Update time and route
                    current_time += transport_data["duration"]
                    current_time = max(current_time, problem.LUNCH_START)
                    
                    x_var[day, transport_choice, last_location, lunch_hawker] = 1
                    u_var[day, lunch_hawker] = current_time
                    
                    last_location = lunch_hawker
                    current_cost += transport_data["price"] + 10  # Meal cost
                    
                    visited_hawkers_by_day[day].add(lunch_hawker)
                except KeyError:
                    pass
        
        # Add attractions based on value
        for attr_idx, _ in attraction_values:
            # Skip if already visited
            if attr_idx in visited_attractions:
                continue
            
            try:
                # Find transport to attraction
                transport_hour = problem.get_transport_hour(current_time)
                transport_key = (problem.locations[last_location]["name"], 
                                 problem.locations[attr_idx]["name"], 
                                 transport_hour)
                
                transit_data = problem.transport_matrix[transport_key]["transit"]
                drive_data = problem.transport_matrix[transport_key]["drive"]
                
                # Choose transport
                transport_choice = 0  # Default transit
                if drive_data["duration"] < transit_data["duration"] * 0.7:
                    transport_choice = 1
                
                transport_data = transit_data if transport_choice == 0 else drive_data
                attraction_cost = problem.locations[attr_idx].get("entrance_fee",0)
                
                # Check time and budget constraints
                time_needed = transport_data["duration"] + problem.locations[attr_idx]["duration"]
                max_time_before_dinner = problem.DINNER_START - current_time
                
                if time_needed > max_time_before_dinner or current_cost + transport_data["price"] + attraction_cost > problem.budget:
                    continue
                
                # Add attraction route
                current_time += transport_data["duration"]
                x_var[day, transport_choice, last_location, attr_idx] = 1
                u_var[day, attr_idx] = current_time
                
                last_location = attr_idx
                current_cost += transport_data["price"] + attraction_cost
                visited_attractions.add(attr_idx)
                
                # Limit to 2 attractions per day
                if len([a for a in visited_attractions if u_var[day, a] > 0]) >= 2:
                    break
            except KeyError:
                pass
        
        # Add dinner if missing
        if not dinner_visit:
            available_dinner_hawkers = [h for h in hawkers if h not in visited_hawkers_by_day[day]]
            if available_dinner_hawkers:
                # Choose best dinner hawker (highest rating)
                dinner_hawker = max(available_dinner_hawkers, 
                                    key=lambda x: problem.locations[x].get("rating", 0))
                
                # Add dinner route
                try:
                    # Find best transport to dinner hawker
                    transport_hour = problem.get_transport_hour(current_time)
                    transport_key = (problem.locations[last_location]["name"], 
                                     problem.locations[dinner_hawker]["name"], 
                                     transport_hour)
                    
                    transit_data = problem.transport_matrix[transport_key]["transit"]
                    drive_data = problem.transport_matrix[transport_key]["drive"]
                    
                    # Choose transport
                    transport_choice = 0  # Default transit
                    if drive_data["duration"] < transit_data["duration"] * 0.7:
                        transport_choice = 1
                    
                    transport_data = transit_data if transport_choice == 0 else drive_data
                    
                    # Update time and route
                    current_time += transport_data["duration"]
                    current_time = max(current_time, problem.DINNER_START)
                    
                    x_var[day, transport_choice, last_location, dinner_hawker] = 1
                    u_var[day, dinner_hawker] = current_time
                    
                    last_location = dinner_hawker
                    current_cost += transport_data["price"] + 10  # Meal cost
                    
                    visited_hawkers_by_day[day].add(dinner_hawker)
                except KeyError:
                    pass
        
        # Return to hotel
        try:
            transport_hour = problem.get_transport_hour(current_time)
            transport_key = (problem.locations[last_location]["name"], 
                             problem.locations[0]["name"], 
                             transport_hour)
            
            transit_data = problem.transport_matrix[transport_key]["transit"]
            drive_data = problem.transport_matrix[transport_key]["drive"]
            
            # Choose fastest transport method
            transport_choice = 0  # Default transit
            if drive_data["duration"] < transit_data["duration"]:
                transport_choice = 1
            
            transport_data = transit_data if transport_choice == 0 else drive_data
            
            # Add route back to hotel
            x_var[day, transport_choice, last_location, 0] = 1
            
            # Update cost and time
            current_time += transport_data["duration"]
            current_cost += transport_data["price"]
            
            u_var[day, 0] = current_time
        except KeyError:
            pass
    
    # Flatten and return
    solution[:problem.x_shape] = x_var.flatten()
    solution[problem.x_shape:] = u_var.flatten()
    
    return solution

def repair_satisfaction_based(problem, solution):
    """
    Repair solution by prioritizing attractions with high satisfaction
    
    Args:
        problem: TravelItineraryProblem instance
        solution: Partially destroyed solution vector
        
    Returns:
        np.ndarray: Repaired solution
    """
    # Reshape solution into x_var and u_var
    x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                            problem.num_locations, problem.num_locations)
    u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
    
    # Get potential locations
    attractions = [i for i in range(problem.num_locations) 
                  if problem.locations[i]["type"] == "attraction"]
    hawkers = [i for i in range(problem.num_locations) 
              if problem.locations[i]["type"] == "hawker"]
    
    # Rank attractions by satisfaction
    attraction_rankings = sorted(
        [(i, problem.locations[i].get("satisfaction", 0)) for i in attractions],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Rank hawkers by rating
    hawker_rankings = sorted(
        [(i, problem.locations[i].get("rating", 0)) for i in hawkers],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Track currently visited attractions and hawkers
    visited_attractions = set()
    visited_hawkers_by_day = defaultdict(set)
    
    # Find currently visited locations
    for day in range(problem.NUM_DAYS):
        for j in range(problem.num_transport_types):
            for k in range(problem.num_locations):
                for l in range(problem.num_locations):
                    if x_var[day, j, k, l] > 0:
                        if problem.locations[l]["type"] == "attraction":
                            visited_attractions.add(l)
                        elif problem.locations[l]["type"] == "hawker":
                            visited_hawkers_by_day[day].add(l)
    
    # Repair for each day
    total_cost = problem.NUM_DAYS * problem.HOTEL_COST
    
    for day in range(problem.NUM_DAYS):
        # Ensure each day starts from hotel
        current_location = 0
        current_time = problem.START_TIME
        u_var[day, 0] = current_time
        
        # Lunch check
        lunch_visit = any(
            problem.LUNCH_START <= u_var[day, h] <= problem.LUNCH_END 
            for h in visited_hawkers_by_day[day]
        )
        
        # Dinner check
        dinner_visit = any(
            problem.DINNER_START <= u_var[day, h] <= problem.DINNER_END 
            for h in visited_hawkers_by_day[day]
        )
        
        # Add lunch if missing
        if not lunch_visit:
            best_lunch_hawker = next(
                (h for h, _ in hawker_rankings if h not in visited_hawkers_by_day[day]), 
                None
            )
            
            if best_lunch_hawker is not None:
                try:
                    # Find transport to lunch hawker
                    transport_hour = problem.get_transport_hour(current_time)
                    transport_key = (problem.locations[current_location]["name"], 
                                     problem.locations[best_lunch_hawker]["name"], 
                                     transport_hour)
                    
                    transit_data = problem.transport_matrix[transport_key]["transit"]
                    drive_data = problem.transport_matrix[transport_key]["drive"]
                    
                    # Choose optimal transport
                    transport_choice = 0
                    transport_data = transit_data
                    if drive_data["duration"] < transit_data["duration"] * 0.7:
                        transport_choice = 1
                        transport_data = drive_data
                    
                    # Update time and route
                    current_time += transport_data["duration"]
                    current_time = max(current_time, problem.LUNCH_START)
                    
                    x_var[day, transport_choice, current_location, best_lunch_hawker] = 1
                    u_var[day, best_lunch_hawker] = current_time
                    
                    current_location = best_lunch_hawker
                    total_cost += transport_data["price"] + 10  # Meal cost
                    visited_hawkers_by_day[day].add(best_lunch_hawker)
                except KeyError:
                    pass
        
        # Add top satisfaction attractions
        for attr_idx, _ in attraction_rankings:
            # Skip if already visited
            if attr_idx in visited_attractions:
                continue
            
            try:
                # Find transport to attraction
                transport_hour = problem.get_transport_hour(current_time)
                transport_key = (problem.locations[current_location]["name"], 
                                 problem.locations[attr_idx]["name"], 
                                 transport_hour)
                
                transit_data = problem.transport_matrix[transport_key]["transit"]
                drive_data = problem.transport_matrix[transport_key]["drive"]
                
                # Choose optimal transport
                transport_choice = 0
                transport_data = transit_data
                if drive_data["duration"] < transit_data["duration"] * 0.7:
                    transport_choice = 1
                    transport_data = drive_data
                
                attraction_cost = problem.locations[attr_idx].get("entrance_fee",0)
                attraction_duration = problem.locations[attr_idx]["duration"]
                
                # Check time and budget constraints
                time_needed = transport_data["duration"] + attraction_duration
                max_time_before_dinner = problem.DINNER_START - current_time
                
                if (time_needed > max_time_before_dinner or 
                    total_cost + transport_data["price"] + attraction_cost > problem.budget):
                    continue
                
                # Add attraction
                current_time += transport_data["duration"]
                x_var[day, transport_choice, current_location, attr_idx] = 1
                u_var[day, attr_idx] = current_time
                
                current_location = attr_idx
                total_cost += transport_data["price"] + attraction_cost
                visited_attractions.add(attr_idx)
                
                # Limit to 2 attractions per day
                if len([a for a in visited_attractions if u_var[day, a] > 0]) >= 2:
                    break
            except KeyError:
                pass
        
        # Add dinner if missing
        if not dinner_visit:
            best_dinner_hawker = next(
                (h for h, _ in hawker_rankings if h not in visited_hawkers_by_day[day]), 
                None
            )
            
            if best_dinner_hawker is not None:
                try:
                    # Find transport to dinner hawker
                    transport_hour = problem.get_transport_hour(current_time)
                    transport_key = (problem.locations[current_location]["name"], 
                                     problem.locations[best_dinner_hawker]["name"], 
                                     transport_hour)
                    
                    transit_data = problem.transport_matrix[transport_key]["transit"]
                    drive_data = problem.transport_matrix[transport_key]["drive"]
                    
                    # Choose optimal transport
                    transport_choice = 0
                    transport_data = transit_data
                    if drive_data["duration"] < transit_data["duration"] * 0.7:
                        transport_choice = 1
                        transport_data = drive_data
                    
                    # Update time and route
                    current_time += transport_data["duration"]
                    current_time = max(current_time, problem.DINNER_START)
                    
                    x_var[day, transport_choice, current_location, best_dinner_hawker] = 1
                    u_var[day, best_dinner_hawker] = current_time
                    
                    current_location = best_dinner_hawker
                    total_cost += transport_data["price"] + 10  # Meal cost
                    visited_hawkers_by_day[day].add(best_dinner_hawker)
                except KeyError:
                    pass
        
        # Return to hotel
        try:
            transport_hour = problem.get_transport_hour(current_time)
            transport_key = (problem.locations[current_location]["name"], 
                             problem.locations[0]["name"], 
                             transport_hour)
            
            transit_data = problem.transport_matrix[transport_key]["transit"]
            drive_data = problem.transport_matrix[transport_key]["drive"]
            
            # Choose fastest transport method
            transport_choice = 0
            transport_data = transit_data
            if drive_data["duration"] < transit_data["duration"]:
                transport_choice = 1
                transport_data = drive_data
            
            # Add route back to hotel
            x_var[day, transport_choice, current_location, 0] = 1
            
            # Update cost and time
            current_time += transport_data["duration"]
            total_cost += transport_data["price"]
            
            u_var[day, 0] = current_time
        except KeyError:
            pass
    
    # Flatten and return
    solution[:problem.x_shape] = x_var.flatten()
    solution[problem.x_shape:] = u_var.flatten()
    
    return solution

def repair_time_based(problem, solution):
    """
    Repair solution by redistributing visits based on time efficiency
    
    Args:
        problem: TravelItineraryProblem instance
        solution: Partially destroyed solution vector
        
    Returns:
        np.ndarray: Repaired solution
    """
    # Reshape solution into x_var and u_var
    x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                            problem.num_locations, problem.num_locations)
    u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
    
    # Get potential locations
    attractions = [i for i in range(problem.num_locations) 
                  if problem.locations[i]["type"] == "attraction"]
    hawkers = [i for i in range(problem.num_locations) 
              if problem.locations[i]["type"] == "hawker"]
    
    # Rank locations by time efficiency
    def calculate_time_efficiency(location_idx):
        """Calculate time efficiency for a location"""
        if problem.locations[location_idx]["type"] == "attraction":
            satisfaction = problem.locations[location_idx].get("satisfaction", 0)
            duration = problem.locations[location_idx].get("duration", 60)
            return satisfaction / duration
        elif problem.locations[location_idx]["type"] == "hawker":
            rating = problem.locations[location_idx].get("rating", 0)
            duration = problem.locations[location_idx].get("duration", 60)
            return rating / duration
        return 0
    
    attraction_efficiency = sorted(
        [(i, calculate_time_efficiency(i)) for i in attractions],
        key=lambda x: x[1],
        reverse=True
    )
    
    hawker_efficiency = sorted(
        [(i, calculate_time_efficiency(i)) for i in hawkers],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Track currently visited locations
    visited_attractions = set()
    visited_hawkers_by_day = defaultdict(set)
    
    # Find currently visited locations
    for day in range(problem.NUM_DAYS):
        for j in range(problem.num_transport_types):
            for k in range(problem.num_locations):
                for l in range(problem.num_locations):
                    if x_var[day, j, k, l] > 0:
                        if problem.locations[l]["type"] == "attraction":
                            visited_attractions.add(l)
                        elif problem.locations[l]["type"] == "hawker":
                            visited_hawkers_by_day[day].add(l)
    
    # Track total trip cost
    total_cost = problem.NUM_DAYS * problem.HOTEL_COST
    
    # Repair for each day
    for day in range(problem.NUM_DAYS):
        # Reset to hotel start
        current_location = 0
        current_time = problem.START_TIME
        u_var[day, 0] = current_time
        
        # Check current lunch and dinner status
        lunch_visit = any(
            problem.LUNCH_START <= u_var[day, h] <= problem.LUNCH_END 
            for h in visited_hawkers_by_day[day]
        )
        
        dinner_visit = any(
            problem.DINNER_START <= u_var[day, h] <= problem.DINNER_END 
            for h in visited_hawkers_by_day[day]
        )
        
        # Add lunch if missing
        if not lunch_visit:
            # Choose most time-efficient hawker not yet visited
            best_lunch_hawker = next(
                (h for h, _ in hawker_efficiency if h not in visited_hawkers_by_day[day]), 
                None
            )
            
            if best_lunch_hawker is not None:
                try:
                    # Find transport to lunch hawker
                    transport_hour = problem.get_transport_hour(current_time)
                    transport_key = (problem.locations[current_location]["name"], 
                                     problem.locations[best_lunch_hawker]["name"], 
                                     transport_hour)
                    
                    transit_data = problem.transport_matrix[transport_key]["transit"]
                    drive_data = problem.transport_matrix[transport_key]["drive"]
                    
                    # Choose optimal transport based on time efficiency
                    transport_choice = 0
                    transport_data = transit_data
                    if drive_data["duration"] < transit_data["duration"] * 0.7:
                        transport_choice = 1
                        transport_data = drive_data
                    
                    # Update time and route
                    current_time += transport_data["duration"]
                    current_time = max(current_time, problem.LUNCH_START)
                    
                    x_var[day, transport_choice, current_location, best_lunch_hawker] = 1
                    u_var[day, best_lunch_hawker] = current_time
                    
                    current_location = best_lunch_hawker
                    total_cost += transport_data["price"] + 10  # Meal cost
                    visited_hawkers_by_day[day].add(best_lunch_hawker)
                except KeyError:
                    pass
        
        # Add most time-efficient attractions
        for attr_idx, _ in attraction_efficiency:
            # Skip if already visited
            if attr_idx in visited_attractions:
                continue
            
            try:
                # Find transport to attraction
                transport_hour = problem.get_transport_hour(current_time)
                transport_key = (problem.locations[current_location]["name"], 
                                 problem.locations[attr_idx]["name"], 
                                 transport_hour)
                
                transit_data = problem.transport_matrix[transport_key]["transit"]
                drive_data = problem.transport_matrix[transport_key]["drive"]
                
                # Choose optimal transport based on time efficiency
                transport_choice = 0
                transport_data = transit_data
                if drive_data["duration"] < transit_data["duration"] * 0.7:
                    transport_choice = 1
                    transport_data = drive_data
                
                attraction_cost = problem.locations[attr_idx].get("entrance_fee",0)
                attraction_duration = problem.locations[attr_idx]["duration"]
                
                # Check time and budget constraints
                time_needed = transport_data["duration"] + attraction_duration
                max_time_before_dinner = problem.DINNER_START - current_time
                
                if (time_needed > max_time_before_dinner or 
                    total_cost + transport_data["price"] + attraction_cost > problem.budget):
                    continue
                
                # Add attraction
                current_time += transport_data["duration"]
                x_var[day, transport_choice, current_location, attr_idx] = 1
                u_var[day, attr_idx] = current_time
                
                current_location = attr_idx
                total_cost += transport_data["price"] + attraction_cost
                visited_attractions.add(attr_idx)
                
                # Limit to 2 attractions per day
                if len([a for a in visited_attractions if u_var[day, a] > 0]) >= 2:
                    break
            except KeyError:
                pass
        
        # Add dinner if missing
        if not dinner_visit:
            # Choose most time-efficient hawker not yet visited for dinner
            best_dinner_hawker = next(
                (h for h, _ in hawker_efficiency if h not in visited_hawkers_by_day[day]), 
                None
            )
            
            if best_dinner_hawker is not None:
                try:
                    # Find transport to dinner hawker
                    transport_hour = problem.get_transport_hour(current_time)
                    transport_key = (problem.locations[current_location]["name"], 
                                     problem.locations[best_dinner_hawker]["name"], 
                                     transport_hour)
                    
                    transit_data = problem.transport_matrix[transport_key]["transit"]
                    drive_data = problem.transport_matrix[transport_key]["drive"]
                    
                    # Choose optimal transport based on time efficiency
                    transport_choice = 0
                    transport_data = transit_data
                    if drive_data["duration"] < transit_data["duration"] * 0.7:
                        transport_choice = 1
                        transport_data = drive_data
                    
                    # Update time and route
                    current_time += transport_data["duration"]
                    current_time = max(current_time, problem.DINNER_START)
                    
                    x_var[day, transport_choice, current_location, best_dinner_hawker] = 1
                    u_var[day, best_dinner_hawker] = current_time
                    
                    current_location = best_dinner_hawker
                    total_cost += transport_data["price"] + 10  # Meal cost
                    visited_hawkers_by_day[day].add(best_dinner_hawker)
                except KeyError:
                    pass
        
        # Return to hotel
        try:
            transport_hour = problem.get_transport_hour(current_time)
            transport_key = (problem.locations[current_location]["name"], 
                             problem.locations[0]["name"], 
                             transport_hour)
            
            transit_data = problem.transport_matrix[transport_key]["transit"]
            drive_data = problem.transport_matrix[transport_key]["drive"]
            
            # Choose fastest transport method
            transport_choice = 0
            transport_data = transit_data
            if drive_data["duration"] < transit_data["duration"]:
                transport_choice = 1
                transport_data = drive_data
            
            # Add route back to hotel
            x_var[day, transport_choice, current_location, 0] = 1
            
            # Update cost and time
            current_time += transport_data["duration"]
            total_cost += transport_data["price"]
            
            u_var[day, 0] = current_time
        except KeyError:
            pass
    
    # Flatten and return
    solution[:problem.x_shape] = x_var.flatten()
    solution[problem.x_shape:] = u_var.flatten()
    
    return solution

def repair_random(problem, solution):
    """
    Repair the solution using a randomized approach

    Args:
        problem: TravelItineraryProblem instance
        solution: Partially destroyed solution vector

    Returns:
        np.ndarray: Repaired solution
    """
    # Reshape solution into x_var and u_var
    x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                            problem.num_locations, problem.num_locations)
    u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)

    # Get all locations by type
    hotel_idx = 0  # Assuming hotel is at index 0

    attractions = [i for i in range(problem.num_locations) 
                 if problem.locations[i]["type"] == "attraction"]

    hawkers = [i for i in range(problem.num_locations) 
              if problem.locations[i]["type"] == "hawker"]

    # Shuffle attractions and hawkers for randomized selection
    random.shuffle(attractions)
    random.shuffle(hawkers)

    # Calculate current cost
    current_cost = problem.NUM_DAYS * problem.HOTEL_COST

    # Track visited attractions
    visited_attractions = set()
    for day in range(problem.NUM_DAYS):
        for attr_idx in attractions:
            if np.sum(x_var[day, :, :, attr_idx]) > 0:
                visited_attractions.add(attr_idx)

                # Add attraction cost
                current_cost += problem.locations[attr_idx].get("entrance_fee",0)

    # For each day
    for day in range(problem.NUM_DAYS):
        # Check if the day is empty or needs fixing
        lunch_visit = False
        dinner_visit = False

        # Track visited hawkers for this day
        day_hawkers = []
        for hawker_idx in hawkers:
            if np.sum(x_var[day, :, :, hawker_idx]) > 0:
                day_hawkers.append(hawker_idx)

                # Check time of visit for lunch/dinner
                arrival_time = u_var[day, hawker_idx]
                if arrival_time >= problem.LUNCH_START and arrival_time <= problem.LUNCH_END:
                    lunch_visit = True
                if arrival_time >= problem.DINNER_START and arrival_time <= problem.DINNER_END:
                    dinner_visit = True

                # Add hawker cost
                current_cost += 10  # Assumed meal cost

        # Reset the day if it's not properly structured
        needs_reset = False

        # Check if hotel is the starting point
        if np.sum(x_var[day, :, hotel_idx, :]) != 1:
            needs_reset = True

        # Check if we return to the hotel
        returns_to_hotel = False
        for j in range(problem.num_transport_types):
            for k in range(problem.num_locations):
                if x_var[day, j, k, hotel_idx] == 1:
                    returns_to_hotel = True
        
        # If day needs reset or constraints are not met
        if needs_reset or not lunch_visit or not dinner_visit or not returns_to_hotel:
            # Reset this day's routes
            x_var[day, :, :, :] = 0
            u_var[day, :] = 0
            
            # Always start at hotel
            u_var[day, hotel_idx] = problem.START_TIME
            current_location = hotel_idx
            current_time = problem.START_TIME
            
            # Randomly select lunch hawker
            lunch_candidates = hawkers.copy()
            random.shuffle(lunch_candidates)
            
            # Find lunch hawker
            lunch_hawker = None
            for potential_hawker in lunch_candidates:
                try:
                    # Get transport options
                    transport_hour = problem.get_transport_hour(current_time)
                    transport_key = (problem.locations[current_location]["name"], 
                                     problem.locations[potential_hawker]["name"], 
                                     transport_hour)
                    
                    transit_data = problem.transport_matrix[transport_key]["transit"]
                    drive_data = problem.transport_matrix[transport_key]["drive"]
                    
                    # Choose transport method
                    transport_choice = 0  # Default transit
                    if drive_data["duration"] < transit_data["duration"] * 0.7 and current_cost + drive_data["price"] <= problem.budget:
                        transport_choice = 1  # Drive
                    
                    transport_data = transit_data if transport_choice == 0 else drive_data
                    
                    # Update time and cost
                    current_time += transport_data["duration"]
                    current_time = max(current_time, problem.LUNCH_START)
                    current_cost += transport_data["price"]
                    
                    # Set lunch route
                    x_var[day, transport_choice, current_location, potential_hawker] = 1
                    u_var[day, potential_hawker] = current_time
                    
                    # Add meal cost
                    current_cost += 10
                    
                    # Update location
                    current_location = potential_hawker
                    lunch_hawker = potential_hawker
                    break
                except KeyError:
                    continue
            
            # Add attractions
            max_attractions = min(2, len(attractions))
            added_attractions = 0
            
            for attr_idx in attractions:
                # Skip if already visited
                if attr_idx in visited_attractions:
                    continue
                
                try:
                    # Get transport options
                    transport_hour = problem.get_transport_hour(current_time)
                    transport_key = (problem.locations[current_location]["name"], 
                                     problem.locations[attr_idx]["name"], 
                                     transport_hour)
                    
                    transit_data = problem.transport_matrix[transport_key]["transit"]
                    drive_data = problem.transport_matrix[transport_key]["drive"]
                    
                    # Choose transport method
                    transport_choice = 0  # Default transit
                    if drive_data["duration"] < transit_data["duration"] * 0.7 and current_cost + drive_data["price"] <= problem.budget:
                        transport_choice = 1  # Drive
                    
                    transport_data = transit_data if transport_choice == 0 else drive_data
                    
                    # Check time and budget constraints
                    attraction_cost = problem.locations[attr_idx].get("entrance_fee",0)
                    attraction_duration = problem.locations[attr_idx]["duration"]
                    
                    # Check if we have enough time before dinner
                    time_needed = transport_data["duration"] + attraction_duration
                    if current_time + time_needed >= problem.DINNER_START:
                        continue
                    
                    # Check budget
                    if current_cost + transport_data["price"] + attraction_cost > problem.budget:
                        continue
                    
                    # Update time and cost
                    current_time += transport_data["duration"]
                    current_cost += transport_data["price"] + attraction_cost
                    
                    # Set attraction route
                    x_var[day, transport_choice, current_location, attr_idx] = 1
                    u_var[day, attr_idx] = current_time
                    
                    # Update location
                    current_location = attr_idx
                    visited_attractions.add(attr_idx)
                    
                    # Increment attraction count
                    added_attractions += 1
                    
                    # Stop if we've added max attractions
                    if added_attractions >= max_attractions:
                        break
                except KeyError:
                    continue
            
            # Add dinner hawker
            dinner_candidates = [h for h in hawkers if h != lunch_hawker]
            random.shuffle(dinner_candidates)
            
            for potential_hawker in dinner_candidates:
                try:
                    # Get transport options
                    transport_hour = problem.get_transport_hour(current_time)
                    transport_key = (problem.locations[current_location]["name"], 
                                     problem.locations[potential_hawker]["name"], 
                                     transport_hour)
                    
                    transit_data = problem.transport_matrix[transport_key]["transit"]
                    drive_data = problem.transport_matrix[transport_key]["drive"]
                    
                    # Choose transport method
                    transport_choice = 0  # Default transit
                    if drive_data["duration"] < transit_data["duration"] * 0.7 and current_cost + drive_data["price"] <= problem.budget:
                        transport_choice = 1  # Drive
                    
                    transport_data = transit_data if transport_choice == 0 else drive_data
                    
                    # Update time and cost
                    current_time += transport_data["duration"]
                    current_time = max(current_time, problem.DINNER_START)
                    current_cost += transport_data["price"]
                    
                    # Set dinner route
                    x_var[day, transport_choice, current_location, potential_hawker] = 1
                    u_var[day, potential_hawker] = current_time
                    
                    # Add meal cost
                    current_cost += 10
                    
                    # Update location
                    current_location = potential_hawker
                    break
                except KeyError:
                    continue
            
            # Return to hotel
            try:
                # Get transport options
                transport_hour = problem.get_transport_hour(current_time)
                transport_key = (problem.locations[current_location]["name"], 
                                 problem.locations[hotel_idx]["name"], 
                                 transport_hour)
                
                transit_data = problem.transport_matrix[transport_key]["transit"]
                drive_data = problem.transport_matrix[transport_key]["drive"]
                
                # Choose transport method
                transport_choice = 0  # Default transit
                if drive_data["duration"] < transit_data["duration"] and current_cost + drive_data["price"] <= problem.budget:
                    transport_choice = 1  # Drive
                
                transport_data = transit_data if transport_choice == 0 else drive_data
                
                # Set route back to hotel
                x_var[day, transport_choice, current_location, hotel_idx] = 1
                
                # Update time and cost
                current_time += transport_data["duration"]
                current_cost += transport_data["price"]
                
                # Set hotel return time
                u_var[day, hotel_idx] = current_time
            except KeyError:
                pass
    
    # Flatten and return solution
    solution[:problem.x_shape] = x_var.flatten()
    solution[problem.x_shape:] = u_var.flatten()
    
    return solution