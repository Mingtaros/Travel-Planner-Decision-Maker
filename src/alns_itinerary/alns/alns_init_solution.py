import numpy as np
import logging

logger = logging.getLogger(__name__)

def check_budget_for_attraction(problem, current_location, current_cost, current_time, attr_idx, day):
    """
    Check if adding this attraction leaves enough budget for remaining trip needs
    
    Args:
        current_cost (float): Current total cost
        current_time (float): Current time of day
        attr_idx (int): Index of the attraction
        day (int): Current day
    
    Returns:
        bool: True if adding the attraction is budget-feasible
    """
    # Estimate remaining costs
    # 1. Attraction entrance fee
    attraction_cost = problem.locations[attr_idx]["entrance_fee"]
    
    # 2. Transport to attraction
    try:
        transport_hour = problem.get_transport_hour(current_time)
        transit_data = problem.transport_matrix[
            (problem.locations[current_location]["name"], 
            problem.locations[attr_idx]["name"], 
            transport_hour)]["transit"]
        
        drive_data = problem.transport_matrix[
            (problem.locations[current_location]["name"], 
            problem.locations[attr_idx]["name"], 
            transport_hour)]["drive"]
        
        # Choose cheapest transport
        transport_cost = min(transit_data["price"], drive_data["price"])
    except KeyError:
        # Fallback transport cost
        transport_cost = 5
    
    # 3. Budget for remaining meals (lunch and dinner)
    # Assume 10 SGD per meal
    remaining_meals_cost = 20
    
    # 4. Transport back to hotel
    try:
        transport_hour = problem.get_transport_hour(current_time)
        transit_data = problem.transport_matrix[
            (problem.locations[attr_idx]["name"], 
            problem.locations[0]["name"], 
            transport_hour)]["transit"]
        
        drive_data = problem.transport_matrix[
            (problem.locations[attr_idx]["name"], 
            problem.locations[0]["name"], 
            transport_hour)]["drive"]
        
        # Choose cheapest transport
        hotel_return_cost = min(transit_data["price"], drive_data["price"])
    except KeyError:
        # Fallback hotel return cost
        hotel_return_cost = 5
    
    # Total estimated additional cost
    total_additional_cost = (
        attraction_cost + 
        transport_cost + 
        remaining_meals_cost + 
        hotel_return_cost
    )
    
    # Check if adding this attraction keeps us within budget
    BUDGET_TOLERANCE = 1.05
    return (current_cost + total_additional_cost) <= (problem.budget * BUDGET_TOLERANCE)

def create_initial_solution(problem, memory_efficient, optimize_numpy_arrays):
    """
    Create an initial solution using an enhanced greedy heuristic approach
    
    Returns:
        np.ndarray: Initial solution vector
    """
    logger.info("Creating greedy heuristic initial solution...")
    
    BUDGET_TOLERANCE = 1.05
    MEAL_COST = 10  # Cost of a meal
    HOTEL_RETURN_BUFFER = 30  # Estimated cost to return to hotel

    # Initialize solution vectors
    solution = np.zeros(problem.n_var, dtype=int)
    x_var = np.zeros((problem.NUM_DAYS, problem.num_transport_types, 
                    problem.num_locations, problem.num_locations), dtype=int)
    u_var = np.zeros((problem.NUM_DAYS, problem.num_locations), dtype=float)
    
    # Set the hotel as always starting from 9 AM
    hotel_index = 0  # Assuming hotel is at index 0
    
    # Track attractions already visited to ensure each is visited at most once
    attractions_visited = set()
    total_cost = problem.NUM_DAYS * problem.HOTEL_COST
    
    # Get all hawkers for meal planning
    hawkers = [i for i in range(problem.num_locations) 
            if problem.locations[i]["type"] == "hawker"]
    
    # Rank hawkers by rating for better meal experiences
    hawker_rankings = [(i, problem.locations[i].get("rating", 0)) 
                    for i in hawkers]
    hawker_rankings.sort(key=lambda x: x[1], reverse=True)  # Sort by rating (highest first)
    
    # Get all attractions
    attractions = [i for i in range(problem.num_locations) 
                if problem.locations[i]["type"] == "attraction"]
    
    # Rank attractions by value (satisfaction / (cost + time))
    attraction_rankings = []
    for attr_idx in attractions:
        attr = problem.locations[attr_idx]
        satisfaction = attr.get("satisfaction", 0)
        cost = max(1, attr.get("entrance_fee", 1))  # Avoid division by zero
        duration = attr.get("duration", 60)
        
        # Calculate value ratio (higher is better)
        value_ratio = satisfaction / (cost + duration/60)
        attraction_rankings.append((attr_idx, value_ratio))
    
    # Sort attractions by value ratio (highest first)
    attraction_rankings.sort(key=lambda x: x[1], reverse=True)
    
    # Track total cost of the itinerary
    total_cost = problem.NUM_DAYS * problem.HOTEL_COST
    
    # For each day, build an optimal schedule
    for day in range(problem.NUM_DAYS):
        
        # Start at hotel
        current_location = hotel_index
        current_time = problem.START_TIME
        
        # Set the initial time for hotel
        u_var[day, hotel_index] = current_time
        
        # Plan lunch (greedy: choose highest rated available hawker)
        lunch_time = problem.LUNCH_START + 30  # Target lunch around 11:30 AM
        lunch_hawker = None
        
        # Try each hawker in order of rating until we find a feasible one
        for hawker_idx, _ in hawker_rankings:
            try:
                # Calculate transport from current location to hawker
                transport_hour = problem.get_transport_hour(current_time)
                transit_data = problem.transport_matrix[
                    (problem.locations[current_location]["name"], 
                    problem.locations[hawker_idx]["name"], 
                    transport_hour)]["transit"]
                
                drive_data = problem.transport_matrix[
                    (problem.locations[current_location]["name"], 
                    problem.locations[hawker_idx]["name"], 
                    transport_hour)]["drive"]
                
                # Choose faster/cheaper transport based on budget
                if drive_data["duration"] < transit_data["duration"] * 0.7 and total_cost + drive_data["price"] <= problem.budget:
                    # Driving saves significant time and is within budget
                    transport_choice = 1  # drive
                    transport_time = drive_data["duration"]
                    transport_cost = drive_data["price"]
                else:
                    # Default to transit
                    transport_choice = 0  # transit
                    transport_time = transit_data["duration"]
                    transport_cost = transit_data["price"]
                
                # Calculate arrival time at hawker
                hawker_arrival_time = current_time + transport_time
                
                # Check if arrival is before end of lunch window
                if hawker_arrival_time < problem.LUNCH_END:
                    # This hawker works for lunch!
                    lunch_hawker = hawker_idx
                    
                    # Set the route in the solution
                    x_var[day, transport_choice, current_location, lunch_hawker] = 1
                    
                    # Calculate meal time (make sure it's within lunch window)
                    lunch_time = max(hawker_arrival_time, problem.LUNCH_START)
                    
                    # Add costs
                    total_cost += transport_cost + 10  # Transit + meal cost
                    
                    # Update time and location
                    meal_duration = problem.locations[lunch_hawker]["duration"]
                    current_time = lunch_time + meal_duration
                    current_location = lunch_hawker
                    
                    # Set the time in the solution
                    u_var[day, lunch_hawker] = lunch_time
                    
                    # Found a valid lunch hawker, so break
                    break
            except KeyError:
                # Missing transport data, try next hawker
                continue
        
        if lunch_hawker is None:
            # If no feasible lunch hawker found, this is a problem!
            logger.warning(f"Day {day}: Could not find a feasible lunch hawker")
            # Just choose first hawker as a fallback
            if hawkers:
                lunch_hawker = hawkers[0]
                lunch_time = problem.LUNCH_START + 30
                u_var[day, lunch_hawker] = lunch_time
                
                # Assume default transit time and cost
                x_var[day, 0, current_location, lunch_hawker] = 1
                total_cost += 5  # Estimate transit cost
                total_cost += 10  # Meal cost
                
                current_time = lunch_time + problem.locations[lunch_hawker]["duration"]
                current_location = lunch_hawker
        
        # Visit attractions between lunch and dinner
        remaining_attractions = [(idx, val) for idx, val in attraction_rankings 
                            if idx not in attractions_visited]
        
        # Available time between lunch and dinner
        available_time = problem.DINNER_START - current_time
        
        while available_time > 0 and remaining_attractions:
            # Get highest value unvisited attraction
            attr_idx, _ = remaining_attractions[0]
            
            # Check budget feasibility before adding attraction
            if not check_budget_for_attraction(problem, current_location, total_cost, current_time, attr_idx, day):
                # Remove this attraction if budget doesn't allow
                remaining_attractions = remaining_attractions[1:]
                continue
            
            remaining_attractions = remaining_attractions[1:]  # Remove from candidates
            
            try:
                # Check if we can reach and visit this attraction before dinner
                transport_hour = problem.get_transport_hour(current_time)
                transit_data = problem.transport_matrix[
                    (problem.locations[current_location]["name"], 
                    problem.locations[attr_idx]["name"], 
                    transport_hour)]["transit"]
                
                drive_data = problem.transport_matrix[
                    (problem.locations[current_location]["name"], 
                    problem.locations[attr_idx]["name"], 
                    transport_hour)]["drive"]
                
                # Choose transit vs drive based on time savings and budget
                if drive_data["duration"] < transit_data["duration"] * 0.7 and total_cost + drive_data["price"] <= problem.budget:
                    transport_choice = 1  # drive
                    transport_time = drive_data["duration"]
                    transport_cost = drive_data["price"]
                else:
                    transport_choice = 0  # transit
                    transport_time = transit_data["duration"]
                    transport_cost = transit_data["price"]
                
                # Calculate total time needed including transport and visiting the attraction
                attraction_duration = problem.locations[attr_idx]["duration"]
                total_time_needed = transport_time + attraction_duration
                
                # Check if we have enough time before dinner
                if total_time_needed <= available_time:
                    # This attraction works! Add it to the itinerary
                    x_var[day, transport_choice, current_location, attr_idx] = 1
                    
                    # Calculate arrival time at the attraction
                    attr_arrival_time = current_time + transport_time
                    
                    # Update time, costs, and location
                    current_time = attr_arrival_time + attraction_duration
                    total_cost += transport_cost + problem.locations[attr_idx]["entrance_fee"]
                    current_location = attr_idx
                    
                    # Set the time in the solution
                    u_var[day, attr_idx] = attr_arrival_time
                    
                    # Mark as visited to prevent revisiting in later days
                    attractions_visited.add(attr_idx)
                    
                    # Update available time
                    available_time = problem.DINNER_START - current_time
            except KeyError:
                # Missing transport data, skip this attraction
                continue
        
        # Plan dinner (exclude lunch hawker if possible)
        dinner_time = problem.DINNER_START + 30  # Target dinner around 5:30 PM
        dinner_hawker = None
        
        # Available dinner hawkers (prioritize different from lunch)
        dinner_candidates = [(idx, rating) for idx, rating in hawker_rankings 
                            if idx != lunch_hawker]
        
        # If no other options, include lunch hawker
        if not dinner_candidates and hawker_rankings:
            dinner_candidates = hawker_rankings
        
        # Try each hawker for dinner
        for hawker_idx, _ in dinner_candidates:
            try:
                # Calculate transport to dinner hawker
                transport_hour = problem.get_transport_hour(current_time)
                transit_data = problem.transport_matrix[
                    (problem.locations[current_location]["name"], 
                    problem.locations[hawker_idx]["name"], 
                    transport_hour)]["transit"]
                
                drive_data = problem.transport_matrix[
                    (problem.locations[current_location]["name"], 
                    problem.locations[hawker_idx]["name"], 
                    transport_hour)]["drive"]
                
                # Choose optimal transport
                if drive_data["duration"] < transit_data["duration"] * 0.7 and total_cost + drive_data["price"] <= problem.budget:
                    transport_choice = 1  # drive
                    transport_time = drive_data["duration"]
                    transport_cost = drive_data["price"]
                else:
                    transport_choice = 0  # transit
                    transport_time = transit_data["duration"]
                    transport_cost = transit_data["price"]
                
                # Calculate dinner arrival time
                hawker_arrival_time = current_time + transport_time
                
                # Check if arrival is before end of dinner window
                if hawker_arrival_time < problem.DINNER_END:
                    # This hawker works for dinner!
                    dinner_hawker = hawker_idx
                    
                    # Set the route in the solution
                    x_var[day, transport_choice, current_location, dinner_hawker] = 1
                    
                    # Calculate meal time (ensure it's within dinner window)
                    dinner_time = max(hawker_arrival_time, problem.DINNER_START)
                    
                    # Add costs
                    total_cost += transport_cost + 10  # Transit + meal cost
                    
                    # Update time and location
                    meal_duration = problem.locations[dinner_hawker]["duration"]
                    current_time = dinner_time + meal_duration
                    current_location = dinner_hawker
                    
                    # Set the time in the solution
                    u_var[day, dinner_hawker] = dinner_time
                    
                    # Found a valid dinner hawker, so break
                    break
            except KeyError:
                # Missing transport data, try next hawker
                continue
        
        if dinner_hawker is None:
            # If no feasible dinner hawker found, this is a problem!
            logger.warning(f"Day {day}: Could not find a feasible dinner hawker")
            # Just choose first hawker as a fallback
            if hawkers:
                dinner_hawker = hawkers[0]
                dinner_time = problem.DINNER_START + 30
                u_var[day, dinner_hawker] = dinner_time
                
                # Assume default transit time and cost
                x_var[day, 0, current_location, dinner_hawker] = 1
                total_cost += 5  # Estimate transit cost
                total_cost += 10  # Meal cost
                
                current_time = dinner_time + problem.locations[dinner_hawker]["duration"]
                current_location = dinner_hawker
        
        # After dinner, visit one more attraction if time permits
        available_time = problem.HARD_LIMIT_END_TIME - 60 - current_time  # Leave 60 min buffer to return to hotel
        
        if available_time > 60:  # Only if we have at least an hour
            # Try to find an evening attraction
            remaining_attractions = [(idx, val) for idx, val in attraction_rankings 
                                if idx not in attractions_visited]
            
            for attr_idx, _ in remaining_attractions:
                try:
                    # Check if we can reach and visit this attraction within available time
                    transport_hour = problem.get_transport_hour(current_time)
                    transit_data = problem.transport_matrix[
                        (problem.locations[current_location]["name"], 
                        problem.locations[attr_idx]["name"], 
                        transport_hour)]["transit"]
                    
                    drive_data = problem.transport_matrix[
                        (problem.locations[current_location]["name"], 
                        problem.locations[attr_idx]["name"], 
                        transport_hour)]["drive"]
                    
                    # Choose optimal transport
                    if drive_data["duration"] < transit_data["duration"] * 0.7 and total_cost + drive_data["price"] <= problem.budget:
                        transport_choice = 1  # drive
                        transport_time = drive_data["duration"]
                        transport_cost = drive_data["price"]
                    else:
                        transport_choice = 0  # transit
                        transport_time = transit_data["duration"]
                        transport_cost = transit_data["price"]
                    
                    # Calculate total time needed
                    attraction_duration = problem.locations[attr_idx]["duration"]
                    total_time_needed = transport_time + attraction_duration
                    
                    # Check if we have enough time
                    if total_time_needed <= available_time:
                        # This attraction works for evening
                        x_var[day, transport_choice, current_location, attr_idx] = 1
                        
                        # Calculate arrival time
                        attr_arrival_time = current_time + transport_time
                        
                        # Update time, costs, and location
                        current_time = attr_arrival_time + attraction_duration
                        total_cost += transport_cost + problem.locations[attr_idx]["entrance_fee"]
                        current_location = attr_idx
                        
                        # Set the time in the solution
                        u_var[day, attr_idx] = attr_arrival_time
                        
                        # Mark as visited
                        attractions_visited.add(attr_idx)
                        
                        # We found an evening attraction, so break
                        break
                except KeyError:
                    # Missing transport data, skip this attraction
                    continue
        
        # Return to hotel
        try:
            # Calculate transport back to hotel
            transport_hour = problem.get_transport_hour(current_time)
            transit_data = problem.transport_matrix[
                (problem.locations[current_location]["name"], 
                problem.locations[hotel_index]["name"], 
                transport_hour)]["transit"]
            
            drive_data = problem.transport_matrix[
                (problem.locations[current_location]["name"], 
                problem.locations[hotel_index]["name"], 
                transport_hour)]["drive"]
            
            if drive_data["duration"] < transit_data["duration"] * 0.7 and total_cost + drive_data["price"] <= problem.budget * BUDGET_TOLERANCE:
                # Driving saves significant time and is within budget
                transport_choice = 1  # drive
                transport_time = drive_data["duration"]
                transport_cost = drive_data["price"]
            else:
                # Default to transit
                transport_choice = 0  # transit
                transport_time = transit_data["duration"]
                transport_cost = transit_data["price"]
            
            # Set the route back to hotel
            x_var[day, transport_choice, current_location, hotel_index] = 1
            
            # Calculate return time
            return_time = current_time + transport_time
            
            # Check if return is before hard limit
            if return_time > problem.HARD_LIMIT_END_TIME:
                logger.warning(f"Day {day}: Return to hotel time ({return_time}) exceeds hard limit ({problem.HARD_LIMIT_END_TIME})")
            
            # Update total cost and set return time
            total_cost += transport_cost
            u_var[day, hotel_index] = max(u_var[day, hotel_index], return_time)
        except KeyError:
            # Missing transport data, use a fallback
            logger.warning(f"Day {day}: Missing transport data for return to hotel")
            
            # Fallback: set return with default transit
            x_var[day, 0, current_location, hotel_index] = 1
            return_time = current_time + 30  # Assume 30 minutes
            u_var[day, hotel_index] = max(u_var[day, hotel_index], return_time)
            total_cost += 5  # Estimate transit cost
    
    # Check if the solution is over budget
    if total_cost > problem.budget:
        logger.warning(f"Initial greedy solution exceeds budget: ${total_cost:.2f} vs ${problem.budget:.2f}")
    
    if memory_efficient:
        solution = optimize_numpy_arrays(solution)
    
    solution[:problem.x_shape] = x_var.flatten()
    solution[problem.x_shape:] = u_var.flatten()
    
    return solution