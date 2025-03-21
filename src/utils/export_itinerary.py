import os

def export_itinerary(problem, solution, filename="results/final_itinerary.txt"):
    """
    Export the optimized solution as a readable itinerary
    
    Args:
        problem: The TravelItineraryProblem instance
        solution: The optimal solution vector
        filename: Path to the output file
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Reshape solution into x_var and u_var
    x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                               problem.num_locations, problem.num_locations)
    u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
    
    # Calculate total cost, travel time, and satisfaction
    total_cost = problem.NUM_DAYS * problem.HOTEL_COST
    total_travel_time = 0
    total_satisfaction = 0
    
    # Trace daily routes
    daily_routes = []
    
    for day in range(problem.NUM_DAYS):
        route = trace_daily_routes(problem, x_var, u_var, day)
        daily_routes.append(route)
    
    # Calculate detailed metrics
    for day in range(problem.NUM_DAYS):
        for j in range(problem.num_transport_types):
            for k in range(problem.num_locations):
                for l in range(problem.num_locations):
                    if k == l:
                        continue
                    
                    if x_var[day, j, k, l] == 1:
                        try:
                            transport_hour = problem.get_transport_hour(u_var[day, k])
                            transport_value = problem.transport_matrix[(problem.locations[k]["name"], 
                                                                     problem.locations[l]["name"], 
                                                                     transport_hour)][problem.transport_types[j]]
                            
                            total_travel_time += transport_value["duration"]
                            total_cost += transport_value["price"]
                            
                            # Add location costs
                            if problem.locations[l]["type"] == "attraction":
                                total_cost += problem.locations[l]["entrance_fee"]
                                total_satisfaction += problem.locations[l]["satisfaction"]
                            elif problem.locations[l]["type"] == "hawker":
                                total_cost += 10  # Assumed meal cost
                                total_satisfaction += problem.locations[l]["rating"]
                        except KeyError:
                            # Missing transport data
                            pass
    
    # Open file for writing
    with open(filename, 'w') as f:
        f.write("=================================================\n")
        f.write("           SINGAPORE TRAVEL ITINERARY            \n")
        f.write("=================================================\n\n")
        
        # Write summary
        f.write("TRIP SUMMARY\n")
        f.write("-------------------------------------------------\n")
        f.write(f"Duration: {problem.NUM_DAYS} days\n")
        f.write(f"Total Cost: ${total_cost:.2f} SGD\n")
        f.write(f"Total Travel Time: {total_travel_time:.1f} minutes ({total_travel_time/60:.1f} hours)\n")
        f.write(f"Satisfaction Rating: {total_satisfaction:.1f}\n")
        f.write(f"Hotel: {problem.locations[0]['name']}\n\n")
        
        # Write attractions visited
        attractions_visited = []
        for day in range(problem.NUM_DAYS):
            for step in daily_routes[day]:
                if step["type"] == "attraction" and step["name"] not in attractions_visited:
                    attractions_visited.append(step["name"])
        
        f.write(f"Attractions Visited ({len(attractions_visited)}):\n")
        for i, attr in enumerate(attractions_visited):
            f.write(f"  {i+1}. {attr}\n")
        f.write("\n")
        
        # Write daily itinerary
        for day in range(problem.NUM_DAYS):
            f.write(f"DAY {day+1}\n")
            f.write("-------------------------------------------------\n")
            
            prev_time = problem.START_TIME
            
            for i, step in enumerate(daily_routes[day]):
                # Format time as hours:minutes
                arrival_time = step["time"]
                location_type = step["type"]
                location_name = step["name"]
                
                hours = int(arrival_time // 60)
                minutes = int(arrival_time % 60)
                time_str = f"{hours:02d}:{minutes:02d}"
                
                # Skip printing transport for first location
                if i == 0:
                    f.write(f"{time_str} - Start at {location_name}\n")
                    prev_time = arrival_time
                else:
                    transport = step["transport_from_prev"]
                    transport_cost = 0
                    
                    # Calculate travel time and cost
                    travel_time = arrival_time - prev_time
                    
                    # Try to get actual travel cost from transport matrix
                    try:
                        prev_location = daily_routes[day][i-1]
                        prev_name = prev_location["name"]
                        transport_hour = problem.get_transport_hour(prev_location["time"])
                        
                        transport_data = problem.transport_matrix.get(
                            (prev_name, location_name, transport_hour), {}
                        ).get(transport, {})
                        
                        if transport_data:
                            transport_cost = transport_data.get("price", 0)
                    except:
                        pass
                    
                    f.write(f"{time_str} - Arrive at {location_name}")
                    if location_type != "hotel":
                        f.write(f" ({location_type})")
                    f.write("\n")
                    
                    f.write(f"       Transport: {transport} ({travel_time:.0f} min, ${transport_cost:.2f})\n")
                    
                    # Add activity details
                    if location_type == "attraction":
                        loc_idx = step["location"]
                        entrance_fee = problem.locations[loc_idx].get("entrance_fee", 0)
                        duration = problem.locations[loc_idx].get("duration", 60)
                        satisfaction = problem.locations[loc_idx].get("satisfaction", 0)
                        
                        f.write(f"       Activity: Visit attraction ({duration:.0f} min)\n")
                        f.write(f"       Cost: ${entrance_fee:.2f}\n")
                        f.write(f"       Satisfaction Rating: {satisfaction:.1f}/10\n")
                        
                        # Update previous time
                        prev_time = arrival_time + duration
                        
                    elif location_type == "hawker":
                        loc_idx = step["location"]
                        duration = problem.locations[loc_idx].get("duration", 60)
                        rating = problem.locations[loc_idx].get("rating", 0)
                        
                        # Determine if lunch or dinner
                        meal_type = "Meal"
                        if arrival_time >= problem.LUNCH_START and arrival_time <= problem.LUNCH_END:
                            meal_type = "Lunch"
                        elif arrival_time >= problem.DINNER_START and arrival_time <= problem.DINNER_END:
                            meal_type = "Dinner"
                        
                        f.write(f"       Activity: {meal_type} ({duration:.0f} min)\n")
                        f.write(f"       Cost: $10.00 (estimated)\n")
                        f.write(f"       Food Rating: {rating:.1f}/5\n")
                        
                        # Update previous time
                        prev_time = arrival_time + duration
                
                f.write("\n")
            
            f.write("\n")
        
        # Write budget breakdown
        f.write("BUDGET BREAKDOWN\n")
        f.write("-------------------------------------------------\n")
        f.write(f"Hotel Accommodation ({problem.NUM_DAYS} nights): ${problem.NUM_DAYS * problem.HOTEL_COST:.2f}\n")
        
        # Calculate transport costs
        transport_costs = {"transit": 0, "drive": 0}
        for day in range(problem.NUM_DAYS):
            for j, transport_type in enumerate(problem.transport_types):
                for k in range(problem.num_locations):
                    for l in range(problem.num_locations):
                        if k == l:
                            continue
                        
                        if x_var[day, j, k, l] == 1:
                            try:
                                transport_hour = problem.get_transport_hour(u_var[day, k])
                                transport_data = problem.transport_matrix[(problem.locations[k]["name"], 
                                                                        problem.locations[l]["name"], 
                                                                        transport_hour)][problem.transport_types[j]]
                                
                                transport_costs[problem.transport_types[j]] += transport_data["price"]
                            except KeyError:
                                pass
        
        f.write(f"Public Transport: ${transport_costs['transit']:.2f}\n")
        f.write(f"Taxi/Ride-sharing: ${transport_costs['drive']:.2f}\n")
        
        # Calculate attraction costs
        attraction_cost = 0
        for day in range(problem.NUM_DAYS):
            for k in range(problem.num_locations):
                for l in range(problem.num_locations):
                    if x_var[day, 0, k, l] == 1 or x_var[day, 1, k, l] == 1:  # Using either transport mode
                        if problem.locations[l]["type"] == "attraction":
                            attraction_cost += problem.locations[l]["entrance_fee"]
        
        f.write(f"Attraction Entrance Fees: ${attraction_cost:.2f}\n")
        
        # Calculate food costs (assume $10 per hawker visit)
        hawker_visits = 0
        for day in range(problem.NUM_DAYS):
            for k in range(problem.num_locations):
                for l in range(problem.num_locations):
                    if x_var[day, 0, k, l] == 1 or x_var[day, 1, k, l] == 1:  # Using either transport mode
                        if problem.locations[l]["type"] == "hawker":
                            hawker_visits += 1
        
        food_cost = hawker_visits * 10
        f.write(f"Food (Hawker Centers): ${food_cost:.2f}\n")
        
        # Write total
        f.write("-------------------------------------------------\n")
        f.write(f"TOTAL: ${total_cost:.2f}\n")
    
    return filename

def trace_daily_routes(problem, x_var, u_var, day):
    """
    Trace the route for a specific day, avoiding cycles and repetition
    
    Args:
        problem: The problem instance
        x_var: The binary decision variables reshaped to 4D
        u_var: The time variables reshaped to 2D
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
        "name": problem.locations[current_loc]["name"],
        "type": problem.locations[current_loc]["type"],
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
        for j in range(problem.num_transport_types):
            for l in range(problem.num_locations):
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
            "name": problem.locations[next_loc]["name"],
            "type": problem.locations[next_loc]["type"],
            "time": float(u_var[day, next_loc]),
            "transport_from_prev": problem.transport_types[next_transport]
        })
        
        # Mark location as visited and move to it
        visited.add(next_loc)
        current_loc = next_loc
        
        # Safety check to prevent infinite loops
        if len(route) > problem.num_locations:
            break
    
    return route