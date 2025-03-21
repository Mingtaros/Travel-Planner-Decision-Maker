from pymoo.core.initialization import Initialization
from pymoo.operators.sampling.rnd import IntegerRandomSampling
import numpy as np
import logging
import os

os.makedirs("log", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/heuristic_init.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("heuristic_init")

class HeuristicInitialization(Initialization):
    def __init__(self, heuristic_solution, n_random=99):
        """
        Initialize with a heuristic solution and randomly generated solutions
        
        Args:
            heuristic_solution: The pre-computed heuristic solution(s)
            n_random: Number of random solutions to generate (total pop = n_random + len(heuristic_solution))
        """
        super().__init__(sampling=IntegerRandomSampling())
        self.heuristic_solution = heuristic_solution
        self.n_random = n_random
        
    def do(self, problem, n_samples, **kwargs):
        # Generate random solutions
        random_sols = np.random.random((self.n_random, problem.n_var))
        # Scale random solutions to problem bounds
        xl, xu = problem.xl, problem.xu
        random_sols = xl + random_sols * (xu - xl)
        
        # If heuristic solution is a single solution, reshape to 2D
        if self.heuristic_solution.ndim == 1:
            heuristic_sols = self.heuristic_solution.reshape(1, -1)
        else:
            heuristic_sols = self.heuristic_solution
            
        # Combine heuristic and random solutions
        pop = np.vstack([heuristic_sols, random_sols])
        
        return pop
    
    @staticmethod
    def save_solution_to_file(problem, solution, filename="log/heuristic_solution.log"):
        """
        Save the solution to a log file in a readable format
        
        Args:
            problem: The problem instance
            solution: The solution vector
            filename: Path to the log file
        """
        
        # np.save(filename.replace('log','results').replace('.log','.npy'), solution)
        
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Reshape solution into x_var and u_var
        x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                                problem.num_locations, problem.num_locations)
        u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
        
        # Open file for writing
        with open(filename, 'w') as f:
            f.write("=== HEURISTIC SOLUTION DETAILS ===\n\n")
            
            # Write solution metadata
            f.write(f"Problem dimensions: {problem.NUM_DAYS} days, {problem.num_locations} locations\n")
            f.write(f"Budget: ${problem.budget}\n")
            f.write(f"Solution vector shape: {solution.shape}, dtype: {solution.dtype}\n\n")
            
            # Write x_var (binary decision variables)
            f.write("=== BINARY DECISION VARIABLES (x_var) ===\n")
            f.write(f"Shape: {x_var.shape}\n\n")
            
            for day in range(problem.NUM_DAYS):
                f.write(f"DAY {day+1}:\n")
                
                # For each transport type
                for j, transport_type in enumerate(problem.transport_types):
                    f.write(f"  Transport mode: {transport_type}\n")
                    
                    # Create a matrix of routes
                    route_matrix = np.zeros((problem.num_locations, problem.num_locations), dtype=int)
                    for k in range(problem.num_locations):
                        for l in range(problem.num_locations):
                            route_matrix[k, l] = x_var[day, j, k, l]
                    
                    # Write route matrix with location names
                    f.write("    Origin \\ Destination")
                    for l in range(problem.num_locations):
                        name = problem.locations[l]["name"]
                        # Truncate long names
                        if len(name) > 15:
                            name = name[:12] + "..."
                        f.write(f" | {name:15s}")
                    f.write("\n")
                    
                    f.write("    " + "-" * (20 + 18 * problem.num_locations) + "\n")
                    
                    for k in range(problem.num_locations):
                        name = problem.locations[k]["name"]
                        if len(name) > 15:
                            name = name[:12] + "..."
                        f.write(f"    {name:15s} | ")
                        
                        for l in range(problem.num_locations):
                            f.write(f"{int(route_matrix[k, l]):15d} | ")
                        f.write("\n")
                    
                    f.write("\n")
                
                f.write("\n")
            
            # Write u_var (time variables)
            f.write("=== TIME VARIABLES (u_var) ===\n")
            f.write(f"Shape: {u_var.shape}\n\n")
            
            for day in range(problem.NUM_DAYS):
                f.write(f"DAY {day+1} (times in minutes since day start):\n")
                f.write("  Location ID | Location Name                  | Type       | Time\n")
                f.write("  " + "-" * 70 + "\n")
                
                # Sort by time for readability
                time_sorted_indices = np.argsort(u_var[day, :])
                
                for idx in time_sorted_indices:
                    time_val = u_var[day, idx]
                    if time_val > 0:  # Only show locations that are visited
                        location_name = problem.locations[idx]["name"]
                        location_type = problem.locations[idx]["type"]
                        
                        # Format time as HH:MM
                        hours = int(time_val // 60)
                        minutes = int(time_val % 60)
                        time_str = f"{hours:02d}:{minutes:02d}"
                        
                        f.write(f"  {idx:11d} | {location_name:30s} | {location_type:10s} | {time_str} ({time_val:.1f} min)\n")
                
                f.write("\n")
            
            # Write derived information
            f.write("=== DERIVED INFORMATION ===\n\n")
            
            # Calculate total visits
            total_visits = np.sum(x_var)
            f.write(f"Total visits: {total_visits}\n")
            
            # Count visits by location type
            attraction_visits = 0
            hawker_visits = 0
            hotel_visits = 0
            
            for i in range(problem.num_locations):
                loc_type = problem.locations[i]["type"]
                visit_count = np.sum(x_var[:, :, :, i])  # Count as destination
                
                if loc_type == "attraction":
                    attraction_visits += visit_count
                elif loc_type == "hawker":
                    hawker_visits += visit_count
                elif loc_type == "hotel":
                    hotel_visits += visit_count
            
            f.write(f"Attraction visits: {attraction_visits}\n")
            f.write(f"Hawker visits: {hawker_visits}\n")
            f.write(f"Hotel visits: {hotel_visits}\n\n")
            
            # Write raw solution vector if needed
            f.write("=== RAW SOLUTION VECTOR ===\n")
            f.write(f"First 20 elements: {solution[:20]}\n")
            f.write(f"Last 20 elements: {solution[-20:]}\n")
    
    @staticmethod
    def create_heuristic_solution(problem):
        """
        Create a greedy heuristic solution for the travel itinerary problem
        
        The heuristic works as follows:
        1. For each day, start at the hotel
        2. Visit a hawker for lunch during lunch time window
        3. Visit attractions with highest satisfaction/cost ratio
        4. Visit a different hawker for dinner during dinner time window
        5. Return to hotel before closing time
        
        Returns:
            np.ndarray: A heuristic solution vector
        """
        logger.info("Creating greedy heuristic initial solution...")
        
        # Initialize solution vectors
        solution = np.zeros(problem.n_var, dtype=problem.xl.dtype)
        x_var = np.zeros((problem.NUM_DAYS, problem.num_transport_types, problem.num_locations, problem.num_locations), 
                          dtype=problem.xl.dtype)
        u_var = np.zeros((problem.NUM_DAYS, problem.num_locations), dtype=problem.xl.dtype)
        
        # Set the hotel as always starting from 9 AM
        hotel_index = 0  # Assuming hotel is at index 0
        
        # Set all attraction and hawker visits to be blocked initially
        attractions_visited = set()  # Track attractions already visited
        hawkers_available = list(range(problem.num_locations))
        hawkers_available = [i for i in hawkers_available if i != hotel_index and 
                             problem.locations[i]["type"] == "hawker"]
        
        # Get all attraction indices
        attractions = [i for i in range(problem.num_locations) 
                     if problem.locations[i]["type"] == "attraction"]
        
        # Sort attractions by satisfaction to cost+time ratio (greedy heuristic)
        attraction_values = []
        for attr_idx in attractions:
            attr = problem.locations[attr_idx]
            satisfaction = attr.get("satisfaction", 0)
            cost = attr.get("entrance_fee", 0)
            duration = attr.get("duration", 60)
            
            # Avoid division by zero
            if cost == 0:
                cost = 1
            
            # Higher ratio means better value
            value_ratio = satisfaction / (cost + duration/10)
            attraction_values.append((attr_idx, value_ratio))
        
        # Sort attractions by value (highest to lowest)
        attraction_values.sort(key=lambda x: x[1], reverse=True)
        
        # Keep track of total cost
        total_cost = problem.NUM_DAYS * problem.HOTEL_COST  # Start with hotel costs
        total_budget_used = 0
        
        # For each day
        for day in range(problem.NUM_DAYS):
            # Start at hotel
            current_location = hotel_index
            current_time = problem.START_TIME
            
            # Set the initial time for hotel
            u_var[day, hotel_index] = current_time
            
            # 1. Go to lunch hawker
            # Find available hawkers for lunch
            available_lunch_hawkers = hawkers_available.copy()
            np.random.shuffle(available_lunch_hawkers)  # Randomize selection
            
            if available_lunch_hawkers:
                lunch_hawker = available_lunch_hawkers[0]
                
                # Choose between transit and driving based on cost and time
                transport_choice = 0  # Default to transit (index 0)
                
                # Get transport time and cost using problem's transport matrix
                transport_hour = problem.get_transport_hour(current_time)
                
                transit_data = problem.transport_matrix[(problem.locations[current_location]["name"], 
                                                       problem.locations[lunch_hawker]["name"], 
                                                       transport_hour)]["transit"]
                drive_data = problem.transport_matrix[(problem.locations[current_location]["name"], 
                                                     problem.locations[lunch_hawker]["name"], 
                                                     transport_hour)]["drive"]
                
                # Choose driving if it saves significant time and budget allows
                if drive_data["duration"] < transit_data["duration"] * 0.7 and total_cost + drive_data["price"] <= problem.budget:
                    transport_choice = 1  # Drive (index 1)
                
                # Calculate arrival time
                if transport_choice == 0:
                    transit_time = transit_data["duration"]
                    transit_cost = transit_data["price"]
                    current_time += transit_time
                    total_cost += transit_cost
                else:
                    drive_time = drive_data["duration"]
                    drive_cost = drive_data["price"]
                    current_time += drive_time
                    total_cost += drive_cost
                
                # Set route in solution
                x_var[day, transport_choice, current_location, lunch_hawker] = 1
                
                # Add meal cost
                lunch_cost = 10  # Assumed fixed cost
                total_cost += lunch_cost
                
                # Set finish time at hawker (include duration)
                lunch_duration = problem.locations[lunch_hawker]["duration"]
                current_time += lunch_duration
                
                # Ensure lunch time is within the lunch window
                if current_time < problem.LUNCH_START:
                    current_time = problem.LUNCH_START + lunch_duration
                
                u_var[day, lunch_hawker] = current_time
                current_location = lunch_hawker
            
            # 2. Visit attractions based on value
            # Track attractions visited today
            todays_attractions = []
            
            # Calculate max time available before dinner
            max_time_before_dinner = problem.DINNER_START - current_time
            
            # Add attractions while time permits
            for attr_idx, _ in attraction_values:
                # Skip if already visited
                if attr_idx in attractions_visited:
                    continue
                
                # Check if we still have time before dinner
                transport_hour = problem.get_transport_hour(current_time)
                
                # Get transit data
                transit_data = problem.transport_matrix[(problem.locations[current_location]["name"], 
                                                       problem.locations[attr_idx]["name"], 
                                                       transport_hour)]["transit"]
                drive_data = problem.transport_matrix[(problem.locations[current_location]["name"], 
                                                     problem.locations[attr_idx]["name"], 
                                                     transport_hour)]["drive"]
                
                # Choose transport method
                transport_choice = 0
                transport_time = transit_data["duration"]
                transport_cost = transit_data["price"]
                
                # If driving saves a lot of time and budget allows, choose driving
                if drive_data["duration"] < transit_data["duration"] * 0.7 and total_cost + drive_data["price"] <= problem.budget:
                    transport_choice = 1
                    transport_time = drive_data["duration"]
                    transport_cost = drive_data["price"]
                
                # Calculate time needed
                attraction_duration = problem.locations[attr_idx]["duration"]
                total_time_needed = transport_time + attraction_duration
                
                # If there's not enough time before dinner, skip this attraction
                if total_time_needed > max_time_before_dinner:
                    continue
                
                # Check if adding this attraction would exceed budget
                attraction_cost = problem.locations[attr_idx]["entrance_fee"]
                if total_cost + transport_cost + attraction_cost > problem.budget:
                    continue
                
                # Add this attraction to today's itinerary
                todays_attractions.append(attr_idx)
                
                # Set route in solution
                x_var[day, transport_choice, current_location, attr_idx] = 1
                
                # Update time and cost
                current_time += transport_time + attraction_duration
                total_cost += transport_cost + attraction_cost
                
                # Update current location
                u_var[day, attr_idx] = current_time
                current_location = attr_idx
                
                # Update remaining time
                max_time_before_dinner = problem.DINNER_START - current_time
                
                # Mark as visited
                attractions_visited.add(attr_idx)
                
                # Limit to 2-3 attractions per day
                if len(todays_attractions) >= 2:
                    break
            
            # 3. Go to dinner hawker
            # Find hawkers other than the lunch one
            dinner_hawkers = [h for h in hawkers_available if h != lunch_hawker]
            np.random.shuffle(dinner_hawkers)
            
            if dinner_hawkers:
                dinner_hawker = dinner_hawkers[0]
                
                # Choose transport method
                transport_hour = problem.get_transport_hour(current_time)
                
                transit_data = problem.transport_matrix[(problem.locations[current_location]["name"], 
                                                       problem.locations[dinner_hawker]["name"], 
                                                       transport_hour)]["transit"]
                drive_data = problem.transport_matrix[(problem.locations[current_location]["name"], 
                                                     problem.locations[dinner_hawker]["name"], 
                                                     transport_hour)]["drive"]
                
                transport_choice = 0
                transport_time = transit_data["duration"]
                transport_cost = transit_data["price"]
                
                # If driving saves time and budget allows, choose driving
                if drive_data["duration"] < transit_data["duration"] * 0.7 and total_cost + drive_data["price"] <= problem.budget:
                    transport_choice = 1
                    transport_time = drive_data["duration"]
                    transport_cost = drive_data["price"]
                
                # Calculate arrival time
                current_time += transport_time
                total_cost += transport_cost
                
                # Ensure dinner time is within dinner window
                if current_time < problem.DINNER_START:
                    current_time = problem.DINNER_START
                elif current_time > problem.DINNER_END:
                    # If we're past dinner window, adjust
                    current_time = problem.DINNER_END - problem.locations[dinner_hawker]["duration"]
                
                # Set route in solution
                x_var[day, transport_choice, current_location, dinner_hawker] = 1
                
                # Add meal cost
                dinner_cost = 10  # Assumed fixed cost
                total_cost += dinner_cost
                
                # Set finish time at hawker
                dinner_duration = problem.locations[dinner_hawker]["duration"]
                current_time += dinner_duration
                u_var[day, dinner_hawker] = current_time
                
                # Update location
                current_location = dinner_hawker
            
            # 4. Return to hotel
            transport_hour = problem.get_transport_hour(current_time)
            
            transit_data = problem.transport_matrix[(problem.locations[current_location]["name"], 
                                                   problem.locations[hotel_index]["name"], 
                                                   transport_hour)]["transit"]
            drive_data = problem.transport_matrix[(problem.locations[current_location]["name"], 
                                                 problem.locations[hotel_index]["name"], 
                                                 transport_hour)]["drive"]
            
            # Choose fastest transport method to get back
            if drive_data["duration"] < transit_data["duration"]:
                transport_choice = 1
                return_time = drive_data["duration"]
                return_cost = drive_data["price"]
            else:
                transport_choice = 0
                return_time = transit_data["duration"]
                return_cost = transit_data["price"]
            
            # Set route back to hotel
            x_var[day, transport_choice, current_location, hotel_index] = 1
            
            # Update cost
            total_cost += return_cost
        
        # Reshape x_var and u_var into the flat solution vector
        solution[:problem.x_shape] = x_var.flatten()
        solution[problem.x_shape:] = u_var.flatten()
        
        # Log solution stats
        total_trips = np.sum(x_var)
        attraction_visits = len(attractions_visited)
        
        logger.info(f"Created heuristic solution with {total_trips} trips, visiting {attraction_visits} attractions")
        logger.info(f"Estimated total cost: ${total_cost:.2f}")
        # logger.info(f"Solution: {solution}")
        
        return solution

    @staticmethod
    def print_daily_routes(problem, results):
        """
        Print the daily routes in a clear, readable format
        
        Args:
            results: The validation results dictionary
        """
        # Print feasibility status
        if results["is_feasible"]:
            print("\n✅ Solution is FEASIBLE")
        else:
            print("\n❌ Solution is INFEASIBLE")
            print(f"   - Inequality violations: {len(results['inequality_violations'])}")
            print(f"   - Equality violations: {len(results['equality_violations'])}")
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"   - Total cost: ${results['total_cost']:.2f}")
        print(f"   - Total travel time: {results['total_travel_time']:.1f} minutes")
        print(f"   - Total satisfaction: {results['total_satisfaction']:.1f}")
        
        # Print daily routes
        for day, route in enumerate(results["details"]["daily_routes"]):
            print(f"\n===== Day {day+1} Itinerary =====")
            
            prev_time = None
            
            for i, step in enumerate(route):
                location_type = step["type"]
                location_name = step["name"]
                arrival_time = step["time"]
                transport = step["transport_from_prev"]
                
                # Format time as hours:minutes
                hours = int(arrival_time // 60)
                minutes = int(arrival_time % 60)
                time_str = f"{hours:02d}:{minutes:02d}"
                
                # Skip printing transport for first location
                if i == 0:
                    print(f"{time_str} - Start at {location_name}")
                    prev_time = arrival_time
                else:
                    # Calculate travel time from previous location
                    travel_time = arrival_time - prev_time
                    
                    print(f"{time_str} - Arrive at {location_name} ({location_type}) by {transport}")
                    print(f"       - Travel time: {travel_time:.1f} minutes")
                    
                    # Calculate duration at location
                    if location_type in ["attraction", "hawker"]:
                        duration = problem.locations[step["location"]].get("duration", 60)
                        print(f"       - Stay for {duration:.1f} minutes")
                        # Update previous time to include stay duration
                        prev_time = arrival_time + duration
                    else:
                        prev_time = arrival_time
        
        # Print constraint violations if any
        if not results["is_feasible"]:
            print("\n⚠️ Constraint Violations:")
            
            if results["inequality_violations"]:
                print("\nInequality Violations:")
                for i, violation in enumerate(results["inequality_violations"]):
                    print(f"{i+1}. {violation['type']}: {violation.get('value', 'N/A')} (limit: {violation.get('limit', 'N/A')})")
            
            if results["equality_violations"]:
                print("\nEquality Violations:")
                for i, violation in enumerate(results["equality_violations"]):
                    print(f"{i+1}. {violation['type']}: {violation.get('value', 'N/A')} != {violation.get('required', 'N/A')}")
    
    @staticmethod
    def validate_heuristic_solution(problem, solution):
        """
        Validate a solution against all constraints in the problem
        and trace daily routes without repetition
        
        Args:
            problem: The TravelItineraryProblem instance
            solution: The solution vector to validate
            
        Returns:
            dict: A dictionary with constraint violation information
        """
        # Reshape solution into x_var and u_var
        x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                                problem.num_locations, problem.num_locations)
        u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
        
        # Initialize results
        results = {
            "is_feasible": True,
            "inequality_violations": [],
            "equality_violations": [],
            "total_cost": 0,
            "total_travel_time": 0,
            "total_satisfaction": 0,
            "details": {}
        }
        
        # Calculate solution properties
        total_cost = problem.NUM_DAYS * problem.HOTEL_COST
        total_travel_time = 0
        total_satisfaction = 0
        
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
        
        # ===== Check Inequality Constraints =====
        
        # 1. For attractions, visit at most once
        for k in range(problem.num_locations):
            if problem.locations[k]["type"] == "attraction":
                visits_as_source = np.sum(x_var[:, :, k, :])
                visits_as_dest = np.sum(x_var[:, :, :, k])
                
                if visits_as_source > 1:
                    results["is_feasible"] = False
                    results["inequality_violations"].append({
                        "type": "attraction_max_once_source",
                        "location": k,
                        "name": problem.locations[k]["name"],
                        "value": float(visits_as_source),
                        "limit": 1
                    })
                
                if visits_as_dest > 1:
                    results["is_feasible"] = False
                    results["inequality_violations"].append({
                        "type": "attraction_max_once_dest",
                        "location": k,
                        "name": problem.locations[k]["name"],
                        "value": float(visits_as_dest),
                        "limit": 1
                    })
        
        # 2. For hawkers, visit at most once per day
        for day in range(problem.NUM_DAYS):
            for k in range(problem.num_locations):
                if problem.locations[k]["type"] == "hawker":
                    daily_visits_source = np.sum(x_var[day, :, k, :])
                    daily_visits_dest = np.sum(x_var[day, :, :, k])
                    
                    if daily_visits_source > 1:
                        results["is_feasible"] = False
                        results["inequality_violations"].append({
                            "type": "hawker_max_once_daily_source",
                            "day": day,
                            "location": k,
                            "name": problem.locations[k]["name"],
                            "value": float(daily_visits_source),
                            "limit": 1
                        })
                    
                    if daily_visits_dest > 1:
                        results["is_feasible"] = False
                        results["inequality_violations"].append({
                            "type": "hawker_max_once_daily_dest",
                            "day": day,
                            "location": k,
                            "name": problem.locations[k]["name"],
                            "value": float(daily_visits_dest),
                            "limit": 1
                        })
        
        # 3. Check start time constraint (must start from hotel at START_TIME)
        for day in range(problem.NUM_DAYS):
            if u_var[day, 0] < problem.START_TIME:
                results["is_feasible"] = False
                results["inequality_violations"].append({
                    "type": "start_time_before_minimum",
                    "day": day,
                    "value": float(u_var[day, 0]),
                    "limit": problem.START_TIME
                })
        
        # 4. Check time constraints
        for day in range(problem.NUM_DAYS):
            for j in range(problem.num_transport_types):
                for k in range(problem.num_locations):
                    for l in range(1, problem.num_locations):
                        if k == l:
                            continue
                        
                        # Skip if route not chosen
                        if x_var[day, j, k, l] == 0:
                            continue
                        
                        # Check time constraints
                        time_finish_l = u_var[day, l]
                        transport_hour = problem.get_transport_hour(u_var[day, k])
                        
                        try:
                            transport_value = problem.transport_matrix[(problem.locations[k]["name"], 
                                                                    problem.locations[l]["name"], 
                                                                    transport_hour)][problem.transport_types[j]]
                            
                            time_should_finish_l = u_var[day, k] + transport_value["duration"] + problem.locations[l]["duration"]
                            
                            if time_finish_l < time_should_finish_l:
                                results["is_feasible"] = False
                                results["inequality_violations"].append({
                                    "type": "time_constraint_violated",
                                    "day": day,
                                    "transport": j,
                                    "from": k,
                                    "to": l,
                                    "from_name": problem.locations[k]["name"],
                                    "to_name": problem.locations[l]["name"],
                                    "actual_finish": float(time_finish_l),
                                    "required_finish": float(time_should_finish_l)
                                })
                        except KeyError:
                            results["inequality_violations"].append({
                                "type": "missing_transport_data",
                                "day": day,
                                "transport": j,
                                "from": k,
                                "to": l,
                                "from_name": problem.locations[k]["name"],
                                "to_name": problem.locations[l]["name"],
                                "hour": transport_hour
                            })
        
        # 5. Check transport type constraints (can't use multiple transport types for same route)
        for day in range(problem.NUM_DAYS):
            for k in range(problem.num_locations):
                for l in range(problem.num_locations):
                    transport_sum = np.sum(x_var[day, :, k, l])
                    
                    if transport_sum > 1:
                        results["is_feasible"] = False
                        results["inequality_violations"].append({
                            "type": "multiple_transport_types",
                            "day": day,
                            "from": k,
                            "to": l,
                            "from_name": problem.locations[k]["name"],
                            "to_name": problem.locations[l]["name"],
                            "value": float(transport_sum),
                            "limit": 1
                        })
        
        # 6. Check lunch and dinner constraints
        for day in range(problem.NUM_DAYS):
            lunch_visits = 0
            dinner_visits = 0
            hawker_sum = 0
            
            for k in range(problem.num_locations):
                if problem.locations[k]["type"] == "hawker":
                    hawker_sum += np.sum(x_var[day, :, k, :])
                    
                    # Check if this hawker visit is during lunch or dinner
                    if u_var[day, k] >= problem.LUNCH_START and u_var[day, k] <= problem.LUNCH_END:
                        lunch_visits += 1
                    
                    if u_var[day, k] >= problem.DINNER_START and u_var[day, k] <= problem.DINNER_END:
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
        
        # 7. Calculate total cost
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
        
        # Check budget constraint
        if total_cost > problem.budget:
            results["is_feasible"] = False
            results["inequality_violations"].append({
                "type": "budget_exceeded",
                "value": float(total_cost),
                "limit": problem.budget
            })
        
        # 8. Check minimum and maximum visits
        min_visits = problem.NUM_DAYS * 2  # Minimum 2 visits per day (lunch & dinner)
        max_visits = problem.NUM_DAYS * 6  # Maximum reasonable visits per day
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
        
        # ===== Check Equality Constraints =====
        
        # 1. For each day, hotel must be the starting point
        for day in range(problem.NUM_DAYS):
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
        
        # 2. For each day, must return to hotel
        for day in range(problem.NUM_DAYS):
            # Find last location
            last_time = np.max(u_var[day, :])
            last_locations = np.where(u_var[day, :] == last_time)[0]
            
            # If last location is not hotel (0), check if there's a route back to hotel
            if len(last_locations) > 0 and last_locations[0] != 0:
                last_loc = last_locations[0]
                returns_to_hotel = np.sum(x_var[day, :, last_loc, 0])
                
                if returns_to_hotel != 1:
                    results["is_feasible"] = False
                    results["equality_violations"].append({
                        "type": "not_returning_to_hotel",
                        "day": day,
                        "last_location": last_loc,
                        "last_location_name": problem.locations[last_loc]["name"],
                        "value": float(returns_to_hotel),
                        "required": 1
                    })
        
        # 3. Flow conservation (in = out)
        for day in range(problem.NUM_DAYS):
            for k in range(problem.num_locations):
                incoming = np.sum(x_var[day, :, :, k])
                outgoing = np.sum(x_var[day, :, k, :])
                
                if incoming != outgoing:
                    results["is_feasible"] = False
                    results["equality_violations"].append({
                        "type": "flow_conservation_violated",
                        "day": day,
                        "location": k,
                        "location_name": problem.locations[k]["name"],
                        "incoming": float(incoming),
                        "outgoing": float(outgoing)
                    })
        
        # 4. If attraction is visited as source, it must be visited as destination
        for k in range(problem.num_locations):
            if problem.locations[k]["type"] == "attraction":
                as_source = np.sum(x_var[:, :, k, :])
                as_dest = np.sum(x_var[:, :, :, k])
                
                if as_source != as_dest:
                    results["is_feasible"] = False
                    results["equality_violations"].append({
                        "type": "attraction_source_dest_mismatch",
                        "location": k,
                        "name": problem.locations[k]["name"],
                        "as_source": float(as_source),
                        "as_dest": float(as_dest)
                    })
        
        # Store totals
        results["total_cost"] = float(total_cost)
        results["total_travel_time"] = float(total_travel_time)
        results["total_satisfaction"] = float(total_satisfaction)
        
        # ===== Generate non-repetitive daily routes =====
        
        results["details"] = {
            "visited_attractions": [problem.locations[i]["name"] for i in range(problem.num_locations) 
                                if problem.locations[i]["type"] == "attraction" and np.sum(x_var[:, :, i, :]) > 0],
            "daily_routes": [],
            "daily_timelines": []
        }
        
        # Generate daily routes using improved tracing
        for day in range(problem.NUM_DAYS):
            route = trace_daily_routes(problem, x_var, u_var, day)
            results["details"]["daily_routes"].append(route)
            
            # Also create a simpler timeline for easy viewing
            timeline = []
            for step in route:
                timeline.append({
                    "location": step["location"],
                    "name": step["name"],
                    "type": step["type"],
                    "time": step["time"],
                    "transport": step["transport_from_prev"]
                })
            
            results["details"]["daily_timelines"].append(timeline)
        
        return results

    