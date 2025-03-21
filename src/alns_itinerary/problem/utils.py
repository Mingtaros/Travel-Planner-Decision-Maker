import numpy as np
import logging
import json
import os

logger = logging.getLogger(__name__)

def generate_random_initial_solution(problem):
    """
    Generate a random initial solution for the travel itinerary problem
    
    Args:
        problem: TravelItineraryProblem instance
    
    Returns:
        np.ndarray: Randomly generated solution vector
    """
    # Create solution vector
    solution = np.zeros(problem.n_var, dtype=problem.xl.dtype)
    
    # Reshape solution into x_var and u_var
    x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                             problem.num_locations, problem.num_locations)
    u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
    
    # Get location indices by type
    hotel_idx = 0  # Assuming hotel is at index 0
    attractions = [i for i in range(problem.num_locations) 
                  if problem.locations[i]["type"] == "attraction"]
    hawkers = [i for i in range(problem.num_locations) 
              if problem.locations[i]["type"] == "hawker"]
    
    # For each day
    for day in range(problem.NUM_DAYS):
        # Always start at hotel
        u_var[day, hotel_idx] = problem.START_TIME
        
        # Randomly select lunch hawker
        if hawkers:
            lunch_hawker = np.random.choice(hawkers)
            
            # Find route to lunch hawker from hotel
            for j in range(problem.num_transport_types):
                try:
                    lunch_hour = problem.get_transport_hour(problem.START_TIME)
                    transport_key = (problem.locations[hotel_idx]["name"], 
                                     problem.locations[lunch_hawker]["name"], 
                                     lunch_hour)
                    
                    # Randomly choose transport and add route
                    if np.random.random() < 0.5:
                        x_var[day, j, hotel_idx, lunch_hawker] = 1
                        u_var[day, lunch_hawker] = problem.LUNCH_START
                        break
                except KeyError:
                    continue
        
        # Randomly select 1-2 attractions
        num_attractions = np.random.randint(1, 3)
        for _ in range(num_attractions):
            if attractions:
                selected_attr = np.random.choice(attractions)
                
                # Find route from last location
                current_location = lunch_hawker if 'lunch_hawker' in locals() else hotel_idx
                for j in range(problem.num_transport_types):
                    try:
                        attr_hour = problem.get_transport_hour(u_var[day, current_location])
                        transport_key = (problem.locations[current_location]["name"], 
                                         problem.locations[selected_attr]["name"], 
                                         attr_hour)
                        
                        # Randomly choose transport and add route
                        if np.random.random() < 0.5:
                            x_var[day, j, current_location, selected_attr] = 1
                            u_var[day, selected_attr] = u_var[day, current_location] + \
                                problem.transport_matrix[transport_key][problem.transport_types[j]]["duration"]
                            current_location = selected_attr
                            break
                    except KeyError:
                        continue
        
        # Randomly select dinner hawker
        if hawkers:
            # Ensure different from lunch hawker
            dinner_hawkers = [h for h in hawkers if h != lunch_hawker] if 'lunch_hawker' in locals() else hawkers
            
            if dinner_hawkers:
                dinner_hawker = np.random.choice(dinner_hawkers)
                
                # Find route from last location
                current_location = current_location if 'current_location' in locals() else lunch_hawker
                
                for j in range(problem.num_transport_types):
                    try:
                        dinner_hour = problem.get_transport_hour(u_var[day, current_location])
                        transport_key = (problem.locations[current_location]["name"], 
                                         problem.locations[dinner_hawker]["name"], 
                                         dinner_hour)
                        
                        # Randomly choose transport and add route
                        if np.random.random() < 0.5:
                            x_var[day, j, current_location, dinner_hawker] = 1
                            u_var[day, dinner_hawker] = max(u_var[day, current_location], problem.DINNER_START)
                            current_location = dinner_hawker
                            break
                    except KeyError:
                        continue
        
        # Return to hotel
        if 'current_location' in locals():
            for j in range(problem.num_transport_types):
                try:
                    hotel_hour = problem.get_transport_hour(u_var[day, current_location])
                    transport_key = (problem.locations[current_location]["name"], 
                                     problem.locations[hotel_idx]["name"], 
                                     hotel_hour)
                    
                    # Randomly choose transport and add route
                    if np.random.random() < 0.5:
                        x_var[day, j, current_location, hotel_idx] = 1
                        u_var[day, hotel_idx] = u_var[day, current_location] + \
                            problem.transport_matrix[transport_key][problem.transport_types[j]]["duration"]
                        break
                except KeyError:
                    continue
    
    return solution

def export_solution_to_json(problem, solution, filename=None):
    """
    Export a solution to a JSON file for later analysis or reproduction
    
    Args:
        problem: TravelItineraryProblem instance
        solution: Solution vector to export
        filename: Optional filename to save the solution
    
    Returns:
        dict: Solution data dictionary
    """
    # Evaluate the solution first
    evaluation = problem.evaluate_solution(solution)
    
    # Reshape solution into x_var and u_var
    x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                             problem.num_locations, problem.num_locations)
    u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
    
    # Prepare solution data for export
    solution_data = {
        "metadata": {
            "total_cost": evaluation["total_cost"],
            "total_travel_time": evaluation["total_travel_time"],
            "total_satisfaction": evaluation["total_satisfaction"],
            "is_feasible": evaluation["is_feasible"]
        },
        "daily_routes": [],
        "route_details": []
    }
    
    # Trace routes for each day
    for day in range(problem.NUM_DAYS):
        day_routes = []
        day_details = []
        
        # Trace route for this day
        current_loc = 0  # Start at hotel
        current_time = u_var[day, current_loc]
        
        while True:
            # Find next location
            next_loc_found = False
            for j in range(problem.num_transport_types):
                for l in range(problem.num_locations):
                    if k != l and x_var[day, j, current_loc, l] > 0:
                        # Route details
                        route_detail = {
                            "day": day,
                            "from": problem.locations[current_loc]["name"],
                            "to": problem.locations[l]["name"],
                            "transport_mode": problem.transport_types[j],
                            "departure_time": current_time,
                            "location_type": problem.locations[l]["type"]
                        }
                        
                        # Try to get route information
                        try:
                            transport_hour = problem.get_transport_hour(current_time)
                            transport_key = (problem.locations[current_loc]["name"], 
                                             problem.locations[l]["name"], 
                                             transport_hour)
                            
                            route_info = problem.transport_matrix[transport_key][problem.transport_types[j]]
                            route_detail.update({
                                "transport_duration": route_info["duration"],
                                "transport_cost": route_info["price"]
                            })
                        except KeyError:
                            pass
                        
                        # Add additional location details based on type
                        if problem.locations[l]["type"] == "attraction":
                            route_detail.update({
                                "entrance_fee": problem.locations[l].get("entrance_fee", 0),
                                "satisfaction": problem.locations[l].get("satisfaction", 0)
                            })
                        elif problem.locations[l]["type"] == "hawker":
                            route_detail.update({
                                "food_cost": 10,  # Standard meal cost
                                "rating": problem.locations[l].get("rating", 0)
                            })
                        
                        day_details.append(route_detail)
                        
                        # Update current location and time
                        current_time += route_info["duration"]
                        current_loc = l
                        next_loc_found = True
                        break
                
                if next_loc_found:
                    break
            
            # If no next location, end route tracing
            if not next_loc_found:
                break
        
        solution_data["daily_routes"].append(day_routes)
        solution_data["route_details"].append(day_details)
    
    # Save to file if filename provided
    if filename:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Write JSON file
            with open(filename, 'w') as f:
                json.dump(solution_data, f, indent=2)
            
            logger.info(f"Solution exported to {filename}")
        except Exception as e:
            logger.error(f"Error exporting solution: {e}")
    
    return solution_data

def load_solution_from_json(filename):
    """
    Load a previously exported solution from a JSON file
    
    Args:
        filename: Path to the JSON solution file
    
    Returns:
        dict: Loaded solution data
    """
    try:
        with open(filename, 'r') as f:
            solution_data = json.load(f)
        
        logger.info(f"Solution loaded from {filename}")
        return solution_data
    except Exception as e:
        logger.error(f"Error loading solution: {e}")
        return None

def visualize_solution(problem, solution, output_format='text'):
    """
    Visualize the solution in various formats
    
    Args:
        problem: TravelItineraryProblem instance
        solution: Solution vector
        output_format: 'text', 'json', or 'plot'
    
    Returns:
        Visualization of the solution in specified format
    """
    # Evaluate solution
    evaluation = problem.evaluate_solution(solution)
    
    if output_format == 'text':
        # Text-based visualization
        output = f"Solution Overview:\n"
        output += f"Total Cost: ${evaluation['total_cost']:.2f}\n"
        output += f"Total Travel Time: {evaluation['total_travel_time']:.1f} minutes\n"
        output += f"Total Satisfaction: {evaluation['total_satisfaction']:.1f}\n"
        output += f"Feasible: {evaluation['is_feasible']}\n\n"
        
        # Daily routes
        for day, route in enumerate(evaluation['daily_routes']):
            output += f"Day {day + 1} Route:\n"
            for step in route:
                output += f"  {step['time']:.1f} - {step['name']} ({step['type']})\n"
            output += "\n"
        
        return output
    
    elif output_format == 'json':
        # JSON visualization
        return export_solution_to_json(problem, solution)
    
    elif output_format == 'plot':
        # Plotting functionality (requires matplotlib)
        try:
            import matplotlib.pyplot as plt
            
            # Create figure for visualization
            plt.figure(figsize=(15, 5 * problem.NUM_DAYS))
            
            # Plot each day's route
            for day, route in enumerate(evaluation['daily_routes']):
                plt.subplot(problem.NUM_DAYS, 1, day + 1)
                
                # Extract location names and times
                locations = [step['name'] for step in route]
                times = [step['time'] for step in route]
                
                plt.plot(times, range(len(times)), 'o-')
                plt.yticks(range(len(times)), locations)
                plt.title(f"Day {day + 1} Route")
                plt.xlabel("Time (minutes)")
                plt.grid(True)
            
            plt.tight_layout()
            
            # Save or return the plot
            plt.savefig('solution_visualization.png')
            plt.close()
            
            return 'solution_visualization.png'
        
        except ImportError:
            logger.error("Matplotlib not installed. Cannot create plot visualization.")
            return None
    
    else:
        logger.warning(f"Unsupported visualization format: {output_format}")
        return None