"""
Travel Itinerary Optimizer
==========================

This module provides the main entry point for the VRP-based travel itinerary optimizer.
It orchestrates the optimization process for creating multi-day travel itineraries
that balance satisfaction, cost, and time constraints.

The optimizer uses an Adaptive Large Neighborhood Search (ALNS) approach, adapted 
from Vehicle Routing Problem techniques to handle the complexities of itinerary planning.

Usage:
    # Basic usage with default parameters
    python main.py
    
    # Or import and run programmatically
    from main import main
    results = main(
        seed=42,
        config_path="./config.json",
        llm_path="./llm.json",
        max_attractions=15,
        max_hawkers=10
    )
"""

import os
import logging
import numpy as np
from datetime import datetime

# Import VRP components
from alns.vrp_alns import VRPALNS
from alns.vrp_solution import VRPSolution
from problem.itinerary_problem import TravelItineraryProblem
from data.transport_utils import get_transport_matrix, get_all_locations
from utils.export_json_itinerary import export_json_itinerary
from utils.google_maps_client import GoogleMapsClient
from utils.config import load_config
from data.location_utils import (
    get_hotel_waypoint, 
    integrate_hotel_with_locations, 
    filter_locations
)
from data.llm_utils import (
    load_recommendations,
    filter_by_recommendations,
    augment_location_data
)

def setup_logging():
    """
    Configure application logging.
    
    Sets up both file and console logging with timestamps and appropriate
    log levels. Log files are stored in the 'log' directory with filenames
    that include the current timestamp.
    """
    # Create logs directory if it doesn't exist
    os.makedirs("log", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"log/vrp_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

def enrich_location_data(locations):
    """Add missing data fields to locations with reasonable defaults."""
    for loc in locations:
        if loc["type"] == "hawker":
            if "rating" not in loc:
                loc["rating"] = np.random.uniform(3, 10)
            if "avg_food_price" not in loc:
                loc["avg_food_price"] = np.random.uniform(5, 15)
            if "duration" not in loc:
                loc["duration"] = 60  # standardize 60 mins for meals
        elif loc["type"] == "attraction":
            if "satisfaction" not in loc:
                loc["satisfaction"] = np.random.uniform(3, 10)
            if "entrance_fee" not in loc:
                loc["entrance_fee"] = np.random.uniform(5, 100)
            if "duration" not in loc:
                loc["duration"] = np.random.randint(45, 120)
        elif loc["type"] == "hotel":
            # Set hotel duration to 0 (no time spent at hotel for activities)
            loc["duration"] = 0
    
    return locations

def main(
    seed=42,
    config_path="./src/alns_itinerary/config.json",
    llm_path="./data/alns_inputs/",
    max_attractions=None, 
    max_hawkers=None,
    hotel_name='Marina Bay Sands'
):
    """
    Run the travel itinerary optimization process.
    
    This function orchestrates the entire optimization workflow:
    1. Loading and preparing location and transportation data
    2. Setting up the optimization problem with constraints
    3. Running the ALNS algorithm to find optimal itineraries
    4. Exporting results as JSON itineraries
    
    Args:
        seed (int, optional): Random seed for reproducibility (default: 42)
        config_path (str): Path to the main configuration file
        llm_path (str): Path to the LLM-generated parameters file containing:
            - HOTEL_NAME: Name of the hotel for the trip
            - BUDGET: Total budget in SGD
            - NUM_DAYS: Number of days for the trip
        max_attractions (int, optional): Maximum number of attractions to include
        max_hawkers (int, optional): Maximum number of hawker centers to include
        
    Returns:
        dict or None: Optimization results including:
            - best_solution: The VRPSolution object
            - best_evaluation: Detailed metrics of the best solution
            - stats: Optimization statistics
            Returns None if optimization fails
            
    Note:
        The optimization uses various parameters from the config file to control
        the ALNS algorithm behavior, including weights for different objectives,
        destroy/repair operators, and termination conditions.
    """
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config(config_path)
    
    # Load recommendations if available
    recommendations = None
    parameter_data = None
    if llm_path and os.path.exists(llm_path):
        recommendations, parameter_data = load_recommendations(llm_path)
        logger.info(f"Loaded {len(recommendations)} recommendations")
    
    budget = parameter_data["Budget"]
    num_days = parameter_data["Number of days"]
    max_iterations = config["MAX_ITERATIONS"]
    segment_size = config["SEGMENT_SIZE"]
    time_limit = config["TIME_LIMIT"]
    early_termination_iterations = config["EARLY_TERMINATION_ITERATIONS"]
    weights_destroy = config["WEIGHTS_DESTROY"]
    weights_repair = config["WEIGHTS_REPAIR"]
    objective_weights = parameter_data.get('params', config["OBJECTIVE_WEIGHTS"])
    infeasible_penalty = config["INFEASIBLE_PENALTY"]
    attraction_per_day = config["MAX_ATTRACTION_PER_DAY"]
    meal_buffer_time = config["MEAL_BUFFER_TIME"]
    rich_threshold = config["RICH_THRESHOLD"]
    avg_hawker_cost = config["AVG_HAWKER_COST"]
    rating_max = config["RATING_MAX"]
    approx_hotel_travel_cost = config["APPROX_HOTEL_TRAVEL_COST"]
    weights_scores = config["WEIGHTS_SCORES"]
    destroy_remove_percentage = config["DESTROY_REMOVE_PERCENTAGE"]
    destroy_distant_loc_weights = config["DESTROY_DISTANT_LOC_WEIGHTS"]
    destroy_expensive_threshold = config["DESTROY_EXPENSIVE_THRESHOLD"]
    repair_transit_weights = config["REPAIR_TRANSIT_WEIGHTS"]
    repair_satisfaction_weights = config["REPAIR_SATISFACTION_WEIGHTS"]
    destroy_day_hawker_preserve = config["DESTROY_DAY_HAWKER_PRESERVE"]
    
    if seed is not None:
        np.random.seed(seed)
    
    try:
        # Get hotel waypoint
        logger.info(f"Determining hotel location: {hotel_name or 'Using default'}")
        hotel = get_hotel_waypoint(hotel_name)
        
        # Get transport matrix and locations
        logger.info("Loading transport matrix and locations...")
        all_locations = get_all_locations()
        transport_matrix = get_transport_matrix()
        
        logger.info(f"Loaded {len(all_locations)} locations and transport matrix")
        
        # Apply recommendation-based filtering first
        if recommendations:
            all_locations = filter_by_recommendations(all_locations, recommendations)
            logger.info(f"Filtered locations based on recommendations")
        
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
        
        # Validate integration results
        if updated_locations is None or updated_matrix is None:
            logger.error("Failed to integrate hotel with locations and transport matrix")
            return None
        
        if recommendations:
            updated_locations = augment_location_data(updated_locations, recommendations)
        else:
            updated_locations = enrich_location_data(updated_locations)
        
        # Create problem instance
        logger.info(f"Creating problem instance: {num_days} days, ${budget} budget")
        problem = TravelItineraryProblem(
            budget=budget, 
            locations=updated_locations, 
            transport_matrix=updated_matrix,
            num_days=num_days
        )
        
        # Check problem feasibility
        if not problem.is_feasible:
            logger.error("Problem is not feasible with the given constraints")
            return None
        
        # Configure VRP-ALNS parameters
        alns_config = {
            "max_iterations": max_iterations, # 5000,
            "segment_size": segment_size, #100,
            "time_limit": time_limit,  # 1 hour time limit
            "seed": seed,  # For reproducibility
            "early_termination_iterations": early_termination_iterations,  # Early termination if no improvement
            "weights_destroy": weights_destroy,  # Weights for destroy operators
            "weights_repair": weights_repair,  # Weights for repair operators
            "objective_weights": objective_weights,  # Weight for objective function
            "infeasible_penalty": infeasible_penalty,  # Penalty for infeasible solutions
            "attraction_per_day": attraction_per_day, # Maximum attractions per day
            "rich_threshold": rich_threshold,  # Threshold for rich ratio
            "meal_buffer_time": meal_buffer_time,  # Buffer time for meals
            "avg_hawker_cost": avg_hawker_cost,  # Average hawker cost
            "rating_max": rating_max, # Maximum rating
            "approx_hotel_travel_cost": approx_hotel_travel_cost, # Approximate hotel travel cost
            "weights_scores": weights_scores,  # Weights for scoring function
            "destroy_remove_percentage": destroy_remove_percentage,  # Percentage of destroy to remove
            "destroy_distant_loc_weights": destroy_distant_loc_weights, # Weights for distant locations
            "destroy_expensive_threshold": destroy_expensive_threshold,  # Threshold for expensive destroyer
            "destroy_day_hawker_preserve": destroy_day_hawker_preserve, # Preserve hawker centers per day
            "repair_transit_weights": repair_transit_weights,  # Weights for transit repair
            "repair_satisfaction_weights": repair_satisfaction_weights  # Weights for satisfaction repair
        }
        
        # Initialize VRP-ALNS
        logger.info("Starting VRP-based ALNS optimization...")
        alns = VRPALNS(problem, **alns_config)
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Export the initial solution as JSON
        initial_solution = alns.current_solution
        initial_json_path = f"results/initial_itinerary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        initial_json_file = export_json_itinerary(problem, initial_solution, initial_json_path)
        logger.info(f"Initial solution exported to: {initial_json_file}")
        
        # Run the optimization
        results = alns.run(verbose=True)
        
        # Get best solution
        best_vrp_solution = results['best_solution']
        
        # Export the best solution as JSON
        json_path = f"results/best_itinerary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        json_file = export_json_itinerary(problem, best_vrp_solution, json_path)
        
        # Log optimization results
        logger.info(f"Optimization completed. Solutions exported to:")
        logger.info(f"- Initial solution: {initial_json_file}")
        logger.info(f"- Best solution: {json_file}")
        logger.info(f"Best solution objective: {results['stats']['best_objective']:.4f}")
        logger.info(f"Total cost: ${results['best_evaluation']['total_cost']:.2f}")
        logger.info(f"Total travel time: {results['best_evaluation']['total_travel_time']:.2f} minutes")
        logger.info(f"Total satisfaction: {results['best_evaluation']['total_satisfaction']:.2f}")
        logger.info(f"Solution feasible: {results['stats']['best_is_feasible']}")
        
        return results
    
    except Exception as e:
        logger.error(f"An error occurred during optimization: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    # Example usage with default parameters
    main(
        seed=42,
        config_path="./src/alns_itinerary/config.json",
        llm_path="./data/alns_inputs/",
    )