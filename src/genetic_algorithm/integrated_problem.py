import re
import os
import pickle
import hashlib
import numpy as np
import logging
from datetime import datetime
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.decomposition.asf import ASF

from generate_route_matrix import save_matrix_to_file
from utils.transport_utility import get_transport_matrix, get_all_locations
from utils.google_maps_client import GoogleMapsClient
from utils.get_trip_detail import calculate_public_transport_fare, calculate_car_fare
from utils.generate_init_solution import HeuristicInitialization
from utils.export_itinerary import export_itinerary

# Global cache for hotel routes
HOTEL_ROUTES_CACHE = {}
CACHE_DIRECTORY = "data/cache"

# Set up logging
os.makedirs("log", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/integrated_problem.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("integrated_problem")

np.random.seed(42)

def get_cache_key(hotel):
    """Generate a unique cache key for a hotel"""
    # Use hotel name and coordinates to create a unique key
    key_str = f"{hotel['name']}_{hotel['lat']}_{hotel['lng']}"
    return hashlib.md5(key_str.encode()).hexdigest()

def save_hotel_routes_to_cache(hotel, hotel_routes):
    """Save hotel routes to disk cache"""
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIRECTORY, exist_ok=True)
    
    # Generate cache key and file path
    cache_key = get_cache_key(hotel)
    cache_file = os.path.join(CACHE_DIRECTORY, f"hotel_routes_{cache_key}.pkl")
    
    # Save to file
    with open(cache_file, 'wb') as f:
        pickle.dump(hotel_routes, f)
    
    # Also store in memory cache
    HOTEL_ROUTES_CACHE[cache_key] = hotel_routes
    
    logger.info(f"Saved hotel routes to cache: {cache_file}")
    return cache_key

def load_hotel_routes_from_cache(hotel, locations=None):
    """
    Load hotel routes from cache if available and valid
    
    Args:
        hotel: Hotel location information
        locations: List of all locations to validate cache against (optional)
        
    Returns:
        dict: Cached hotel routes if available and valid, None otherwise
    """
    cache_key = get_cache_key(hotel)
    
    # Function to validate cached routes against current locations
    def validate_cached_routes(hotel_routes, locations):
        """Validate that cached routes are compatible with current locations"""
        if not hotel_routes:
            return False
        
        # Get non-hotel location names
        location_names = [loc["name"] for loc in locations if loc["type"] != "hotel"]
        
        # Check if all required location pairs are in the cache
        time_brackets = [8, 12, 16, 20]  # The time brackets used in the system
        
        for location_name in location_names:
            # Check hotel to location routes
            hotel_to_loc_exists = False
            # Check location to hotel routes
            loc_to_hotel_exists = False
            
            for hour in time_brackets:
                # Check hotel -> location route
                if (hotel["name"], location_name, hour) in hotel_routes:
                    hotel_to_loc_exists = True
                
                # Check location -> hotel route
                if (location_name, hotel["name"], hour) in hotel_routes:
                    loc_to_hotel_exists = True
            
            # If either direction is missing for this location, cache is invalid
            if not hotel_to_loc_exists or not loc_to_hotel_exists:
                logger.warning(f"Missing routes between hotel and location '{location_name}' in cached data")
                return False
        
        logger.info("Cached hotel routes validated successfully")
        return True
    
    # Check memory cache first
    if cache_key in HOTEL_ROUTES_CACHE:
        hotel_routes = HOTEL_ROUTES_CACHE[cache_key]
        
        # Validate if locations are provided
        if locations and not validate_cached_routes(hotel_routes, locations):
            logger.warning(f"Memory-cached routes for {hotel['name']} don't match current locations, cache invalid")
            return None
        
        logger.info(f"Loaded hotel routes from memory cache for {hotel['name']}")
        return hotel_routes
    
    # Check disk cache
    cache_file = os.path.join(CACHE_DIRECTORY, f"hotel_routes_{cache_key}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                hotel_routes = pickle.load(f)
            
            # Validate if locations are provided
            if locations and not validate_cached_routes(hotel_routes, locations):
                logger.warning(f"Disk-cached routes for {hotel['name']} don't match current locations, cache invalid")
                return None
            
            # Store in memory cache for faster access next time
            HOTEL_ROUTES_CACHE[cache_key] = hotel_routes
            
            logger.info(f"Loaded hotel routes from disk cache: {cache_file}")
            return hotel_routes
        except Exception as e:
            logger.error(f"Error loading hotel routes from cache: {e}")
    
    # Cache miss
    return None


def clear_hotel_routes_cache():
    """Clear all cached hotel routes"""
    global HOTEL_ROUTES_CACHE
    HOTEL_ROUTES_CACHE = {}
    
    if os.path.exists(CACHE_DIRECTORY):
        for file in os.listdir(CACHE_DIRECTORY):
            if file.startswith("hotel_routes_"):
                os.remove(os.path.join(CACHE_DIRECTORY, file))
    
    logger.info("Cleared hotel routes cache")

def count_indentation(line_of_code):
    # given line of code, get the indentation to replicate in added constraints
    num_indent = 0
    for chara in line_of_code:
        if chara != " ":
            break
        num_indent += 1
    
    return num_indent

def integrate_problem(base_problem_str, inequality_constraints, equality_constraints):
    
    INEQUALITY_CONSTRAINT_LINE = 187
    EQUALITY_CONSTRAINT_LINE = INEQUALITY_CONSTRAINT_LINE + 1
    INDENTATION_COUNT_LINE = 391
    ADD_CONSTRAINT_LINE = INDENTATION_COUNT_LINE + 1
    
    # update the number of constraints in class initialization
    base_problem_str[INEQUALITY_CONSTRAINT_LINE] = base_problem_str[INEQUALITY_CONSTRAINT_LINE].replace(",", " + " + str(len(inequality_constraints)) + ",")
    base_problem_str[EQUALITY_CONSTRAINT_LINE] = base_problem_str[EQUALITY_CONSTRAINT_LINE].replace(",", " + " + str(len(equality_constraints)) + ",")

    # add additional constraints
    # known location of <ADD ADDITIONAL CONSTRAINTS HERE> is in this line
    num_indent = count_indentation(base_problem_str[INDENTATION_COUNT_LINE]) # see indentation there, match in every added constraints
    for constraint in inequality_constraints: # inequality constraints
        # add indent for each line
        constraint = [" " * num_indent + constraint_line.strip() for constraint_line in constraint.split("\n")]
        constraint = "\n".join(constraint) # re-join to make new constraint
        # add the constraint to the code
        base_problem_str.insert(ADD_CONSTRAINT_LINE, constraint)

    for constraint in equality_constraints: # equality constraints
        # add indent for each line
        constraint = [" " * num_indent + constraint_line.strip() for constraint_line in constraint.split("\n")]
        constraint = "\n".join(constraint) # re-join to make new constraint
        # add the constraint to the code
        base_problem_str.insert(ADD_CONSTRAINT_LINE, constraint)
    
    return base_problem_str

def get_hotel_waypoint(hotel_name):
    """
    Get hotel waypoint from user input (name)
    
    Args:
        hotel_name: Name of the hotel
        
    Returns:
        dict: Hotel location information
    """
    if not hotel_name:
        logger.warning("No hotel information provided, using default hotel")
        # Return a default hotel in central Singapore if none provided
        return {
            "type": "hotel",
            "name": "DEFAULT HOTEL",
            "lat": 1.2904527,  # Marina Bay Sands coordinates as default
            "lng": 103.8577566,
        }
    
    try:
        # Initialize Google Maps client with API key from environment variables
        maps_client = GoogleMapsClient()
        
        # Create search query
        search_query = f"{hotel_name}, Singapore"
        
        logger.info(f"Geocoding hotel: {search_query}")
        
        # Get place details using Google Maps API
        place_details = maps_client.get_place_details(place_name=search_query)
        place_data = maps_client.parse_place_details(place_details)
        
        if not place_data or 'location' not in place_data:
            logger.warning(f"Could not geocode hotel: {search_query}, using default location")
            return {
                "type": "hotel",
                "name": hotel_name,
                "lat": 1.2904527,  # Marina Bay Sands coordinates as default
                "lng": 103.8577566,
            }
        
        # Create and return hotel location information
        hotel = {
            "type": "hotel",
            "name": place_data.get('name', hotel_name),
            "lat": place_data['location']['lat'],
            "lng": place_data['location']['lng'],
        }
        
        logger.info(f"Successfully geocoded hotel: {hotel['name']} at {hotel['lat']}, {hotel['lng']}")
        return hotel
        
    except Exception as e:
        logger.error(f"Error geocoding hotel: {e}")
        # Return default hotel on error
        return {
            "type": "hotel",
            "name": hotel_name,
            "lat": 1.2904527,  # Default coordinates
            "lng": 103.8577566,
        }

def compute_hotel_routes(hotel, locations):
    """
    Compute routes between the hotel and all other locations using route matrix batch method
    Uses cache if available for the same hotel
    
    Args:
        hotel: Hotel location information
        locations: List of all other locations
        
    Returns:
        dict: Route matrix entries for the hotel
    """
    
    # Try to load from cache first
    cached_routes = load_hotel_routes_from_cache(hotel, locations)
    if cached_routes is not None:
        return cached_routes
    
    try:
        # Initialize Google Maps client
        maps_client = GoogleMapsClient()
        
        # Time brackets used in the travel itinerary problem
        time_brackets = [8, 12, 16, 20]  # morning, midday, evening, night
        
        # Create a dictionary to store route information
        hotel_routes = {}
        
        # Create a hotel waypoint for the matrix calculation
        hotel_waypoint = [hotel["name"], hotel["lat"], hotel["lng"]]
        
        # Filter out other hotels from locations
        non_hotel_locations = [loc for loc in locations if loc["type"] != "hotel"]
        
        # Create waypoints list for the other locations
        destination_waypoints = [[loc["name"], loc["lat"], loc["lng"]] for loc in non_hotel_locations]
        
        # Log the operation
        logger.info(f"Computing route matrices for hotel '{hotel['name']}' with {len(destination_waypoints)} destinations")
        
        # For each time bracket, compute route matrix
        for hour in time_brackets:
            # Set departure time based on hour
            departure_time = datetime(2025, 5, 17, hour, 0, 0)
            
            # Compute transit route matrix (hotel to all locations)
            logger.info(f"Computing transit route matrix for hour {hour}...")
            transit_matrix = maps_client.compute_route_matrix(
                origins=[hotel_waypoint],
                destinations=destination_waypoints,
                mode="transit",
                departure_time=departure_time
            )
            
            # Compute driving route matrix (hotel to all locations)
            logger.info(f"Computing driving route matrix for hour {hour}...")
            driving_matrix = maps_client.compute_route_matrix(
                origins=[hotel_waypoint],
                destinations=destination_waypoints,
                mode="drive",
                departure_time=departure_time
            )
            
            # Compute transit route matrix (all locations to hotel)
            logger.info(f"Computing transit route matrix from locations to hotel for hour {hour}...")
            transit_matrix_return = maps_client.compute_route_matrix(
                origins=destination_waypoints,
                destinations=[hotel_waypoint],
                mode="transit",
                departure_time=departure_time
            )
            
            # Compute driving route matrix (all locations to hotel)
            logger.info(f"Computing driving route matrix from locations to hotel for hour {hour}...")
            driving_matrix_return = maps_client.compute_route_matrix(
                origins=destination_waypoints,
                destinations=[hotel_waypoint],
                mode="drive",
                departure_time=departure_time
            )
            
            # Process the matrices and update hotel_routes
            # Process hotel to locations routes (transit)
            for route in transit_matrix:
                if route.get('condition') == 'ROUTE_EXISTS':
                    dest_index = route.get('destinationIndex', 0)
                    if dest_index < len(non_hotel_locations):
                        dest_location = non_hotel_locations[dest_index]
                        
                        # Extract distance and duration
                        distance_meters = route.get('distanceMeters', 0)
                        distance_km = distance_meters / 1000
                        
                        duration_seconds = 0
                        if 'duration' in route:
                            duration_str = route['duration']
                            if duration_str.endswith('s'):
                                duration_seconds = int(duration_str[:-1])
                        
                        # Calculate fare
                        transit_fare = calculate_public_transport_fare(distance_km)
                        
                        # Store the route data
                        if (hotel["name"], dest_location["name"], hour) not in hotel_routes:
                            hotel_routes[(hotel["name"], dest_location["name"], hour)] = {
                                "transit": {
                                    "duration": round(duration_seconds / 60, 1),  # Convert to minutes
                                    "price": round(transit_fare, 2) if transit_fare else 0,
                                },
                                "drive": {
                                    "duration": 0,
                                    "price": 0,
                                }
                            }
                        else:
                            hotel_routes[(hotel["name"], dest_location["name"], hour)]["transit"] = {
                                "duration": round(duration_seconds / 60, 1),  # Convert to minutes
                                "price": round(transit_fare, 2) if transit_fare else 0,
                            }
            
            # Process hotel to locations routes (driving)
            for route in driving_matrix:
                if route.get('condition') == 'ROUTE_EXISTS':
                    dest_index = route.get('destinationIndex', 0)
                    if dest_index < len(non_hotel_locations):
                        dest_location = non_hotel_locations[dest_index]
                        
                        # Extract distance and duration
                        distance_meters = route.get('distanceMeters', 0)
                        distance_km = distance_meters / 1000
                        
                        duration_seconds = 0
                        if 'duration' in route:
                            duration_str = route['duration']
                            if duration_str.endswith('s'):
                                duration_seconds = int(duration_str[:-1])
                        
                        # Calculate fare
                        driving_fare = calculate_car_fare(distance_meters)
                        
                        # Store the route data
                        if (hotel["name"], dest_location["name"], hour) not in hotel_routes:
                            hotel_routes[(hotel["name"], dest_location["name"], hour)] = {
                                "transit": {
                                    "duration": 0,
                                    "price": 0,
                                },
                                "drive": {
                                    "duration": round(duration_seconds / 60, 1),  # Convert to minutes
                                    "price": round(driving_fare, 2) if driving_fare else 0,
                                }
                            }
                        else:
                            hotel_routes[(hotel["name"], dest_location["name"], hour)]["drive"] = {
                                "duration": round(duration_seconds / 60, 1),  # Convert to minutes
                                "price": round(driving_fare, 2) if driving_fare else 0,
                            }
            
            # Process locations to hotel routes (transit)
            for route in transit_matrix_return:
                if route.get('condition') == 'ROUTE_EXISTS':
                    origin_index = route.get('originIndex', 0)
                    if origin_index < len(non_hotel_locations):
                        origin_location = non_hotel_locations[origin_index]
                        
                        # Extract distance and duration
                        distance_meters = route.get('distanceMeters', 0)
                        distance_km = distance_meters / 1000
                        
                        duration_seconds = 0
                        if 'duration' in route:
                            duration_str = route['duration']
                            if duration_str.endswith('s'):
                                duration_seconds = int(duration_str[:-1])
                        
                        # Calculate fare
                        transit_fare = calculate_public_transport_fare(distance_km)
                        
                        # Store the route data
                        if (origin_location["name"], hotel["name"], hour) not in hotel_routes:
                            hotel_routes[(origin_location["name"], hotel["name"], hour)] = {
                                "transit": {
                                    "duration": round(duration_seconds / 60, 1),  # Convert to minutes
                                    "price": round(transit_fare, 2) if transit_fare else 0,
                                },
                                "drive": {
                                    "duration": 0,
                                    "price": 0,
                                }
                            }
                        else:
                            hotel_routes[(origin_location["name"], hotel["name"], hour)]["transit"] = {
                                "duration": round(duration_seconds / 60, 1),  # Convert to minutes
                                "price": round(transit_fare, 2) if transit_fare else 0,
                            }
            
            # Process locations to hotel routes (driving)
            for route in driving_matrix_return:
                if route.get('condition') == 'ROUTE_EXISTS':
                    origin_index = route.get('originIndex', 0)
                    if origin_index < len(non_hotel_locations):
                        origin_location = non_hotel_locations[origin_index]
                        
                        # Extract distance and duration
                        distance_meters = route.get('distanceMeters', 0)
                        distance_km = distance_meters / 1000
                        
                        duration_seconds = 0
                        if 'duration' in route:
                            duration_str = route['duration']
                            if duration_str.endswith('s'):
                                duration_seconds = int(duration_str[:-1])
                        
                        # Calculate fare
                        driving_fare = calculate_car_fare(distance_meters)
                        
                        # Store the route data
                        if (origin_location["name"], hotel["name"], hour) not in hotel_routes:
                            hotel_routes[(origin_location["name"], hotel["name"], hour)] = {
                                "transit": {
                                    "duration": 0,
                                    "price": 0,
                                },
                                "drive": {
                                    "duration": round(duration_seconds / 60, 1),  # Convert to minutes
                                    "price": round(driving_fare, 2) if driving_fare else 0,
                                }
                            }
                        else:
                            hotel_routes[(origin_location["name"], hotel["name"], hour)]["drive"] = {
                                "duration": round(duration_seconds / 60, 1),  # Convert to minutes
                                "price": round(driving_fare, 2) if driving_fare else 0,
                            }
        
        logger.info(f"Successfully computed {len(hotel_routes)} hotel route entries across {len(time_brackets)} time periods")
        
        # Save to cache before returning
        save_hotel_routes_to_cache(hotel, hotel_routes)
        
        return hotel_routes
        
    except Exception as e:
        import traceback
        logger.error(f"Error computing hotel routes: {e}")
        logger.error(traceback.format_exc())
        # Return empty dictionary on error
        return {}

def integrate_hotel_with_locations(hotel, locations, transport_matrix):
    """
    Integrate hotel with existing locations and transport matrix
    
    Args:
        hotel: Hotel location dictionary
        locations: List of location dictionaries
        transport_matrix: Existing transport matrix
        
    Returns:
        tuple: (updated_locations, updated_transport_matrix)
    """
    # Check if hotel already exists in locations
    hotel_exists = False
    for i, loc in enumerate(locations):
        if loc["type"] == "hotel" and loc["name"] == hotel["name"]:
            # Hotel already exists, update its information
            locations[i] = hotel
            hotel_exists = True
            logger.info(f"Updated existing hotel: {hotel['name']}")
            break
    
    # If hotel doesn't exist, add it to locations (at index 0)
    if not hotel_exists:
        locations.insert(0, hotel)
        logger.info(f"Added new hotel: {hotel['name']}")
    
    # Compute routes between hotel and all locations
    hotel_routes = compute_hotel_routes(hotel, locations)
    
    # Check if we have any hotel routes
    if not hotel_routes:
        logger.warning("Failed to compute hotel routes, using default values")
        # Create default routes (this is a fallback to prevent optimization failure)
        return None, None
    
    # Merge hotel routes with existing transport matrix
    updated_matrix = {**transport_matrix, **hotel_routes}
    logger.info(f"Added {len(hotel_routes)} hotel route entries to transport matrix")
    
    return locations, updated_matrix

def filter_locations(locations, max_attractions=None, max_hawkers=None, filter_criteria=None):
    """
    Filter locations by type and criteria, limiting the number of attractions and hawkers.
    
    Args:
        locations: List of all location dictionaries
        max_attractions: Maximum number of attractions to include (None = all)
        max_hawkers: Maximum number of hawkers to include (None = all)
        filter_criteria: Optional dict with criteria to prioritize locations
            (e.g., {'attractions': 'satisfaction', 'hawkers': 'rating'})
    
    Returns:
        list: Filtered list of locations
    """
    # Sort hotel(s) to the front of the list
    hotels = [loc for loc in locations if loc["type"] == "hotel"]
    attractions = [loc for loc in locations if loc["type"] == "attraction"]
    hawkers = [loc for loc in locations if loc["type"] == "hawker"]
    
    # Default sorting criteria
    attraction_sort_key = "satisfaction" if not filter_criteria else filter_criteria.get("attractions", "satisfaction")
    hawker_sort_key = "rating" if not filter_criteria else filter_criteria.get("hawkers", "rating")
    
    # Sort attractions and hawkers by criteria (if present in data)
    if all(attraction_sort_key in loc for loc in attractions):
        attractions.sort(key=lambda x: x.get(attraction_sort_key, 0), reverse=True)
    
    if all(hawker_sort_key in loc for loc in hawkers):
        hawkers.sort(key=lambda x: x.get(hawker_sort_key, 0), reverse=True)
    
    # Limit the number of attractions and hawkers if specified
    if max_attractions is not None and max_attractions > 0:
        attractions = attractions[:max_attractions]
    
    if max_hawkers is not None and max_hawkers > 0:
        hawkers = hawkers[:max_hawkers]
    
    # Combine filtered locations, ensuring hotels come first
    filtered_locations = hotels + attractions + hawkers
    
    return filtered_locations

def main(hotel_name=None, budget=1000, num_days=2, max_attractions=12, max_hawkers=8): # 16 12
    """
    Main function to run the integrated problem with a specified hotel
    
    Args:
        hotel_name: Name of the hotel (optional)
    
    Returns:
        tuple: (result, problem, updated_locations)
    """
    # Make sure log directory exists
    os.makedirs("log", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    with open("src/base_problem.py", 'r') as base_problem_file:
        base_problem_str = base_problem_file.readlines()

    # Define constraints
    inequality_constraints = []
        # """
        # day_one_attraction_limit = np.sum(x_var[0, :, :, :]) - 3 # should be <= 3
        # out["G"].append(day_one_attraction_limit)
        # """
    # ]
    equality_constraints = []
        # """out["H"].append(np.sum(x_var) - 5) # should be == 5""",
    # ]

    # Get user-specified hotel or use default
    logger.info("Getting hotel information...")
    if hotel_name is None:
        hotel_name = input("Enter hotel name (or press Enter for default): ")
    
    hotel = get_hotel_waypoint(hotel_name if hotel_name else None)

    # Get transport matrix and locations
    logger.info("Loading transport matrix and locations...")
    all_locations = get_all_locations()
    transport_matrix = get_transport_matrix()
    
    # Filter locations based on max_attractions and max_hawkers
    if max_attractions is not None or max_hawkers is not None:
        logger.info(f"Filtering locations (max attractions: {max_attractions}, max hawkers: {max_hawkers})...")
        all_locations = filter_locations(all_locations, max_attractions, max_hawkers)
        logger.info(f"Filtered to {len(all_locations)} locations")
    
    # Integrate hotel with locations and transport matrix
    logger.info("Integrating hotel with transport matrix...")
    updated_locations, updated_matrix = integrate_hotel_with_locations(
        hotel, all_locations, transport_matrix
    )

    # For all locations, get necessary data (only if not already present)
    for loc in updated_locations:
        if loc["type"] == "hawker":
            if not "rating" in loc:
                loc["rating"] = np.random.uniform(0, 5)
            if not "avg_food_price" in loc:
                loc["avg_food_price"] = np.random.uniform(5, 15)
            if not "duration" in loc:
                loc["duration"] = 60  # just standardize 60 mins
        elif loc["type"] == "attraction":
            if not "satisfaction" in loc:
                loc["satisfaction"] = np.random.uniform(0, 10)
            if not "entrance_fee" in loc:
                loc["entrance_fee"] = np.random.uniform(5, 100)
            if not "duration" in loc:
                loc["duration"] = np.random.randint(30, 90)
        elif loc["type"] == "hotel":
            # Set hotel duration to 0 (no time spent at hotel for activities)
            loc["duration"] = 0

    # Integrate the problem with constraints
    base_problem_str = integrate_problem(base_problem_str, inequality_constraints, equality_constraints)

    # have base problem set as None for defaulting in case of error
    # class TravelItineraryProblem():
    #     def __init__(self, **kwargs):
    #         pass
    
    exec_namespace = {}
    exec("".join(base_problem_str), globals(), exec_namespace)

    TravelItineraryProblem = exec_namespace["TravelItineraryProblem"]

    with open("src/generated_problem.py", 'w') as f:
        f.writelines(base_problem_str)
    
    # execute the code inside base_problem_str, importing the Problem class.
    logger.info("Executing integrated problem...")
    exec("".join(base_problem_str))

    # make the problemset and solve it.
    logger.info("Creating and solving the optimization problem...")
    problem = TravelItineraryProblem(
        num_days=num_days,
        budget=budget,
        locations=updated_locations,
        transport_matrix=updated_matrix,
    )
    
    heuristic_solution = HeuristicInitialization.create_heuristic_solution(problem)
    logger.info("Generated heuristic initial solution")
    
    np.save('results/heuristic_solution.npy', heuristic_solution)
    
    HeuristicInitialization.save_solution_to_file(problem, heuristic_solution, "log/heuristic_solution.log")
    
    validation_results = HeuristicInitialization.validate_heuristic_solution(problem, heuristic_solution)
    
    HeuristicInitialization.print_daily_routes(problem, validation_results)

    # You can also use the validation results to debug your solution
    if not validation_results["is_feasible"]:
        logger.info(f"Heuristic solution has {len(validation_results['inequality_violations'])} inequality and "
                    f"{len(validation_results['equality_violations'])} equality constraint violations")
    
    # Create initial population with explicit integer type for binary part
    initial_population = np.zeros((200, problem.n_var), dtype=np.int64)  # Use float64 for the entire solution

    # Set the first solution to our heuristic
    initial_population[0] = heuristic_solution

    # For the remaining solutions, create proper random values
    for i in range(1, 200):
        # Binary part: Ensure it's integer 0 or 1
        initial_population[i, :problem.x_shape] = np.random.randint(0, 2, size=problem.x_shape)
        
        # Continuous part: Random values within bounds
        xl_cont, xu_cont = problem.xl[problem.x_shape:], problem.xu[problem.x_shape:]
        initial_population[i, problem.x_shape:] = np.random.uniform(xl_cont, xu_cont)

    # Set up the NSGA2 algorithm with our initialization
    algorithm = NSGA2(
        pop_size=200,
        sampling=initial_population.astype(int),
        crossover=TwoPointCrossover(prob=0.9),
        mutation=BitflipMutation(prob=0.05),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 200)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        save_history=True,
        verbose=True
    )

    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
    
    # Save the best solution to a file
    np.save("results/best_solution.npy", res.X)
    np.save("results/best_objectives.npy", res.F)
    logger.info("Best solution and objectives saved to results directory")
    
    weights = np.array([0.3, 0.3, 0.4])
    
    decomp = ASF()

    approx_ideal = res.F.min(axis=0)
    approx_nadir = res.F.max(axis=0)

    nF = (res.F - approx_ideal) / (approx_nadir - approx_ideal)

    i = decomp.do(nF, 1/weights).argmin()

    pymoo_solution = res.X[i]

    # Export the optimized itinerary
    itinerary_file = export_itinerary(problem, pymoo_solution, "results/singapore_itinerary.txt")
    print(f"Itinerary saved to {itinerary_file}")
    
    return pymoo_solution, res, problem, updated_locations

if __name__ == "__main__":
    import sys
    
    # Get hotel name from command line arguments if provided
    hotel_name = None
    if len(sys.argv) > 1:
        hotel_name = sys.argv[1]
    
    main(hotel_name)