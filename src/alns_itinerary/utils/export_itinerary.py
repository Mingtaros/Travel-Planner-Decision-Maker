import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def export_itinerary(problem, solution, filename=None):
    """
    Export the optimized solution as a readable itinerary
    
    Args:
        problem: The TravelItineraryProblem instance
        solution: The optimal solution vector
        filename: Path to the output file (optional)
    
    Returns:
        str: Path to the exported itinerary file
    """
    # Reshape solution into x_var and u_var
    x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                               problem.num_locations, problem.num_locations)
    u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
    
    # Evaluate the solution
    evaluation = problem.evaluate_solution(solution)
    
    # Generate a default filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)
        filename = f"results/itinerary_{timestamp}.txt"
    else:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Open file for writing
    with open(filename, 'w') as f:
        # Write header
        f.write("=================================================\n")
        f.write("           PERSONALIZED TRAVEL ITINERARY          \n")
        f.write("=================================================\n\n")
        
        # Write trip summary
        f.write("TRIP SUMMARY\n")
        f.write("-------------------------------------------------\n")
        f.write(f"Duration: {problem.NUM_DAYS} days\n")
        f.write(f"Total Budget: ${problem.budget:.2f} SGD\n")
        f.write(f"Actual Expenditure: ${evaluation['total_cost']:.2f} SGD\n")
        f.write(f"Total Travel Time: {evaluation['total_travel_time']:.1f} minutes\n")
        f.write(f"Total Satisfaction Rating: {evaluation['total_satisfaction']:.1f}\n")
        f.write(f"Starting Hotel: {problem.locations[0]['name']}\n\n")
        
        # List attractions visited
        attractions_visited = evaluation.get('visited_attractions', [])
        f.write(f"Attractions Visited ({len(attractions_visited)}):\n")
        for i, attr in enumerate(attractions_visited, 1):
            f.write(f"  {i}. {attr}\n")
        f.write("\n")
        
        # Detailed daily itinerary
        for day, daily_route in enumerate(evaluation.get('daily_routes', []), 1):
            f.write(f"DAY {day}\n")
            f.write("-------------------------------------------------\n")
            
            for i, step in enumerate(daily_route):
                # Format time
                arrival_time = step['time']
                hours = int(arrival_time // 60)
                minutes = int(arrival_time % 60)
                time_str = f"{hours:02d}:{minutes:02d}"
                
                # Write location details
                location_type = step['type']
                location_name = step['name']
                
                # Detailed step information
                f.write(f"{time_str} - Arrive at {location_name}")
                if location_type != "hotel":
                    f.write(f" ({location_type})")
                
                # Add transport details if available
                if 'transport_from_prev' in step and step['transport_from_prev']:
                    f.write(f" via {step['transport_from_prev'].capitalize()} transport")
                
                f.write("\n")
                
                # Add additional details for attractions and hawkers
                if location_type == "attraction":
                    loc_idx = step['location']
                    attr = problem.locations[loc_idx]
                    f.write(f"       Details:\n")
                    f.write(f"         - Duration: {attr.get('duration', 60):.0f} minutes\n")
                    f.write(f"         - Entrance Fee: ${attr.get('entrance_fee', 0):.2f}\n")
                    f.write(f"         - Satisfaction Rating: {attr.get('satisfaction', 0):.1f}/10\n")
                
                elif location_type == "hawker":
                    loc_idx = step['location']
                    hawker = problem.locations[loc_idx]
                    
                    # Determine meal type
                    meal_type = "Meal"
                    if arrival_time >= problem.LUNCH_START and arrival_time <= problem.LUNCH_END:
                        meal_type = "Lunch"
                    elif arrival_time >= problem.DINNER_START and arrival_time <= problem.DINNER_END:
                        meal_type = "Dinner"
                    
                    f.write(f"       Details:\n")
                    f.write(f"         - {meal_type} Meal\n")
                    f.write(f"         - Duration: {hawker.get('duration', 60):.0f} minutes\n")
                    f.write(f"         - Estimated Meal Cost: $10.00\n")
                    f.write(f"         - Food Rating: {hawker.get('rating', 0):.1f}/5\n")
                
                f.write("\n")
        
        # Budget breakdown
        f.write("BUDGET BREAKDOWN\n")
        f.write("-------------------------------------------------\n")
        
        # Hotel costs
        f.write(f"Hotel Accommodation ({problem.NUM_DAYS} nights): ${problem.NUM_DAYS * problem.HOTEL_COST:.2f}\n")
        
        # Compute transport costs by type
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
                                transport_data = problem.transport_matrix[
                                    (problem.locations[k]["name"], 
                                     problem.locations[l]["name"], 
                                     transport_hour)][problem.transport_types[j]]
                                
                                transport_costs[problem.transport_types[j]] += transport_data["price"]
                            except KeyError:
                                pass
        
        f.write(f"Public Transport: ${transport_costs['transit']:.2f}\n")
        f.write(f"Taxi/Ride-sharing: ${transport_costs['drive']:.2f}\n")
        
        # Attraction and meal costs
        attraction_cost = 0
        hawker_visits = 0
        
        for day in range(problem.NUM_DAYS):
            for k in range(problem.num_locations):
                for l in range(problem.num_locations):
                    if x_var[day, 0, k, l] == 1 or x_var[day, 1, k, l] == 1:
                        if problem.locations[l]["type"] == "attraction":
                            attraction_cost += problem.locations[l]["entrance_fee"]
                        elif problem.locations[l]["type"] == "hawker":
                            hawker_visits += 1
        
        f.write(f"Attraction Entrance Fees: ${attraction_cost:.2f}\n")
        f.write(f"Food (Hawker Centers): ${hawker_visits * 10:.2f}\n")
        
        # Total
        f.write("-------------------------------------------------\n")
        f.write(f"TOTAL: ${evaluation['total_cost']:.2f}\n")
    
    logger.info(f"Itinerary exported to {filename}")
    return filename