import numpy as np
import random
import logging

logger = logging.getLogger("destroy_operators")

def destroy_random_days(problem, solution, removal_percentage=0.5):
    """
    Destroy random days from the solution
    
    Args:
        problem: TravelItineraryProblem instance
        solution: Current solution vector
        removal_percentage: Percentage of days to remove (0.0 to 1.0)
    
    Returns:
        np.ndarray: Partially destroyed solution
    """
    # Reshape solution into x_var and u_var
    x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                            problem.num_locations, problem.num_locations)
    u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
    
    # Determine number of days to remove
    num_days_to_remove = max(1, int(problem.NUM_DAYS * removal_percentage))
    
    # Randomly select days to remove
    days_to_remove = random.sample(range(problem.NUM_DAYS), num_days_to_remove)
    
    # Remove selected days (set to zero)
    for day in days_to_remove:
        x_var[day, :, :, :] = 0
        u_var[day, :] = 0
        
        # Ensure hotel start time is maintained
        u_var[day, 0] = problem.START_TIME
    
    # Flatten and return
    solution[:problem.x_shape] = x_var.flatten()
    solution[problem.x_shape:] = u_var.flatten()
    
    return solution

def destroy_random_attractions(problem, solution, removal_percentage=0.6):
    """
    Destroy random attraction visits from the solution
    
    Args:
        problem: TravelItineraryProblem instance
        solution: Current solution vector
        removal_percentage: Percentage of attraction visits to remove (0.0 to 1.0)
        
    Returns:
        np.ndarray: Partially destroyed solution
    """
    # Reshape solution into x_var and u_var
    x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                            problem.num_locations, problem.num_locations)
    u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
    
    # Get attraction indices
    attraction_indices = [i for i in range(problem.num_locations) 
                       if problem.locations[i]["type"] == "attraction"]
    
    # Find visited attractions
    visited_attractions = []
    for day in range(problem.NUM_DAYS):
        for i in attraction_indices:
            # Check if this attraction is visited (as destination)
            if np.sum(x_var[day, :, :, i]) > 0:
                visited_attractions.append((day, i))
    
    # Determine number of attractions to remove
    num_to_remove = max(1, int(len(visited_attractions) * removal_percentage))
    num_to_remove = min(num_to_remove, len(visited_attractions))
    
    # Randomly select attractions to remove
    if visited_attractions:
        to_remove = random.sample(visited_attractions, num_to_remove)
        
        # Remove selected attractions
        for day, attr_idx in to_remove:
            # Find inbound and outbound routes for this attraction
            for j in range(problem.num_transport_types):
                for k in range(problem.num_locations):
                    # Remove routes to this attraction
                    if x_var[day, j, k, attr_idx] > 0:
                        x_var[day, j, k, attr_idx] = 0
                        
                    # Remove routes from this attraction
                    if x_var[day, j, attr_idx, k] > 0:
                        x_var[day, j, attr_idx, k] = 0
            
            # Reset visit time
            u_var[day, attr_idx] = 0
    
    # Flatten and return
    solution[:problem.x_shape] = x_var.flatten()
    solution[problem.x_shape:] = u_var.flatten()
    
    return solution

def destroy_worst_attractions(problem, solution, removal_percentage=0.7):
    """
    Destroy attraction visits with the worst satisfaction-to-cost ratio
    
    Args:
        problem: TravelItineraryProblem instance
        solution: Current solution vector
        removal_percentage: Percentage of attraction visits to remove (0.0 to 1.0)
        
    Returns:
        np.ndarray: Partially destroyed solution
    """
    # Reshape solution into x_var and u_var
    x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                            problem.num_locations, problem.num_locations)
    u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
    
    # Get attraction indices
    attraction_indices = [i for i in range(problem.num_locations) 
                       if problem.locations[i]["type"] == "attraction"]
    
    # Find visited attractions with their value ratio
    visited_attractions = []
    for day in range(problem.NUM_DAYS):
        for i in attraction_indices:
            # Check if this attraction is visited (as destination)
            if np.sum(x_var[day, :, :, i]) > 0:
                attraction = problem.locations[i]
                satisfaction = attraction.get("satisfaction", 0)
                cost = attraction.get("entrance_fee", 1)
                duration = attraction.get("duration", 60)
                
                # Calculate value ratio (lower is worse)
                value_ratio = satisfaction / (cost + duration/10)
                
                visited_attractions.append((day, i, value_ratio))
    
    # Sort by value ratio (ascending)
    visited_attractions.sort(key=lambda x: x[2])
    
    # Determine number of attractions to remove
    num_to_remove = max(1, int(len(visited_attractions) * removal_percentage))
    num_to_remove = min(num_to_remove, len(visited_attractions))
    
    # Select worst attractions to remove
    if visited_attractions:
        to_remove = visited_attractions[:num_to_remove]
        
        # Remove selected attractions
        for day, attr_idx, _ in to_remove:
            # Find inbound and outbound routes for this attraction
            for j in range(problem.num_transport_types):
                for k in range(problem.num_locations):
                    # Remove routes to this attraction
                    if x_var[day, j, k, attr_idx] > 0:
                        x_var[day, j, k, attr_idx] = 0
                        
                    # Remove routes from this attraction
                    if x_var[day, j, attr_idx, k] > 0:
                        x_var[day, j, attr_idx, k] = 0
            
            # Reset visit time
            u_var[day, attr_idx] = 0
    
    # Flatten and return
    solution[:problem.x_shape] = x_var.flatten()
    solution[problem.x_shape:] = u_var.flatten()
    
    return solution

def destroy_random_meals(problem, solution, removal_percentage=0.5):
    """
    Destroy random meal visits (hawker centers) from the solution
    
    Args:
        problem: TravelItineraryProblem instance
        solution: Current solution vector
        removal_percentage: Percentage of meal visits to remove (0.0 to 1.0)
        
    Returns:
        np.ndarray: Partially destroyed solution
    """
    # Reshape solution into x_var and u_var
    x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                            problem.num_locations, problem.num_locations)
    u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
    
    # Get hawker indices
    hawker_indices = [i for i in range(problem.num_locations) 
                    if problem.locations[i]["type"] == "hawker"]
    
    # Find hawker visits
    visited_hawkers = []
    for day in range(problem.NUM_DAYS):
        daily_hawkers = []
        for i in hawker_indices:
            # Check if this hawker is visited (as destination)
            if np.sum(x_var[day, :, :, i]) > 0:
                daily_hawkers.append(i)
        
        # We need to keep at least one hawker per day for lunch
        if len(daily_hawkers) > 1:
            # Keep one hawker (preferably lunch time) and consider the rest for removal
            daily_hawkers_with_time = [(i, u_var[day, i]) for i in daily_hawkers]
            # Sort by time
            daily_hawkers_with_time.sort(key=lambda x: x[1])
            
            # Add all but one hawker to the removal candidates
            for i, _ in daily_hawkers_with_time[1:]:
                visited_hawkers.append((day, i))
    
    # Determine number of hawkers to remove
    num_to_remove = max(1, int(len(visited_hawkers) * removal_percentage))
    num_to_remove = min(num_to_remove, len(visited_hawkers))
    
    # Randomly select hawkers to remove
    if visited_hawkers:
        to_remove = random.sample(visited_hawkers, num_to_remove)
        
        # Remove selected hawkers
        for day, hawker_idx in to_remove:
            # Find inbound and outbound routes for this hawker
            for j in range(problem.num_transport_types):
                for k in range(problem.num_locations):
                    # Remove routes to this hawker
                    if x_var[day, j, k, hawker_idx] > 0:
                        x_var[day, j, k, hawker_idx] = 0
                        
                    # Remove routes from this hawker
                    if x_var[day, j, hawker_idx, k] > 0:
                        x_var[day, j, hawker_idx, k] = 0
            
            # Reset visit time
            u_var[day, hawker_idx] = 0
    
    # Flatten and return
    solution[:problem.x_shape] = x_var.flatten()
    solution[problem.x_shape:] = u_var.flatten()
    
    return solution

def destroy_random_routes(problem, solution, removal_percentage=0.4):
    """
    Destroy random routes from the solution
    
    Args:
        problem: TravelItineraryProblem instance
        solution: Current solution vector
        removal_percentage: Percentage of routes to remove (0.0 to 1.0)
        
    Returns:
        np.ndarray: Partially destroyed solution
    """
    # Reshape solution into x_var and u_var
    x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                            problem.num_locations, problem.num_locations)
    u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
    
    # Find all routes
    routes = []
    for day in range(problem.NUM_DAYS):
        for j in range(problem.num_transport_types):
            for k in range(problem.num_locations):
                for l in range(problem.num_locations):
                    if k != l and x_var[day, j, k, l] > 0:
                        routes.append((day, j, k, l))
    
    # Count routes with hotel as origin (we'll preserve these)
    hotel_start_routes = []
    for day in range(problem.NUM_DAYS):
        for j in range(problem.num_transport_types):
            for l in range(problem.num_locations):
                if l != 0 and x_var[day, j, 0, l] > 0:
                    hotel_start_routes.append((day, j, 0, l))
    
    # Only consider non-hotel-start routes for removal
    removable_routes = [r for r in routes if r not in hotel_start_routes]
    
    # Determine number of routes to remove
    num_to_remove = max(1, int(len(removable_routes) * removal_percentage))
    num_to_remove = min(num_to_remove, len(removable_routes))
    
    # Randomly select routes to remove
    if removable_routes:
        to_remove = random.sample(removable_routes, num_to_remove)
        
        # Remove selected routes
        for day, j, k, l in to_remove:
            x_var[day, j, k, l] = 0
            
            # Reset destination visit time if this was the only route to it
            if np.sum(x_var[day, :, :, l]) == 0:
                u_var[day, l] = 0
    
    # Flatten and return
    solution[:problem.x_shape] = x_var.flatten()
    solution[problem.x_shape:] = u_var.flatten()
    
    return solution

def destroy_time_aware_routes(problem, solution):
    """
    Destroy routes while attempting to maintain time sequences
    """
    # Similar to existing destroy_random_routes, but:
    # 1. Prefer removing routes that don't break critical time constraints
    # 2. Maintain sequence of locations for each day
    x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                            problem.num_locations, problem.num_locations)
    u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
    
    # Identify routes with less critical time dependencies
    time_sensitive_routes = []
    less_sensitive_routes = []
    
    for day in range(problem.NUM_DAYS):
        for j in range(problem.num_transport_types):
            for k in range(problem.num_locations):
                for l in range(problem.num_locations):
                    if x_var[day, j, k, l] == 0:
                        continue
                    
                    # Assess time sensitivity
                    try:
                        transport_hour = problem.get_transport_hour(u_var[day, k])
                        transport_key = (problem.locations[k]["name"], 
                                         problem.locations[l]["name"], 
                                         transport_hour)
                        
                        transport_value = problem.transport_matrix[transport_key][problem.transport_types[j]]
                        
                        # Calculate time requirements
                        time_should_finish_l = (
                            u_var[day, k] + 
                            transport_value["duration"] + 
                            problem.locations[l]["duration"]
                        )
                        
                        # Classify routes
                        if abs(u_var[day, l] - time_should_finish_l) > 30:  # 30 minutes threshold
                            less_sensitive_routes.append((day, j, k, l))
                        else:
                            time_sensitive_routes.append((day, j, k, l))
                    
                    except KeyError:
                        # Less sensitive if transport data is missing
                        less_sensitive_routes.append((day, j, k, l))
    
    # Prioritize removing less time-sensitive routes
    routes_to_remove = less_sensitive_routes
    
    # Remove selected routes
    for day, j, k, l in routes_to_remove:
        x_var[day, j, k, l] = 0
        # Optionally reset time for this location
        u_var[day, l] = 0
    
    # Flatten and return
    solution[:problem.x_shape] = x_var.flatten()
    solution[problem.x_shape:] = u_var.flatten()
    
    return solution