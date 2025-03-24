"""
Main entry point for the VRP-based travel itinerary optimizer.
This file replaces the original main.py with a position-based VRP approach
that handles time constraints more effectively.
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
from data.location_utils import (
    get_hotel_waypoint, 
    integrate_hotel_with_locations, 
    filter_locations
)

def setup_logging():
    """
    Set up logging configuration
    """
    # Create logs directory if it doesn't exist
    os.makedirs("log", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"log/vrp_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

def main(
    hotel_name=None, 
    budget=500, 
    num_days=3, 
    max_attractions=None, 
    max_hawkers=None,
    seed=42
):
    """
    Main function to run VRP-based optimization for travel itinerary
    
    Args:
        hotel_name (str, optional): Name of the hotel to start the trip
        budget (float): Total budget for the trip
        num_days (int): Number of days in the trip
        max_attractions (int, optional): Maximum number of attractions to consider
        max_hawkers (int, optional): Maximum number of hawkers to consider
        
    Returns:
        dict: Optimization results including best solution
    """
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
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
        
        # For all locations, add missing data with randomization
        for loc in updated_locations:
            if loc["type"] == "hawker":
                if "rating" not in loc:
                    loc["rating"] = np.random.uniform(3, 5)  # More realistic ratings
                if "avg_food_price" not in loc:
                    loc["avg_food_price"] = np.random.uniform(5, 15)
                if "duration" not in loc:
                    loc["duration"] = 60  # standardize 60 mins for meals
            elif loc["type"] == "attraction":
                if "satisfaction" not in loc:
                    loc["satisfaction"] = np.random.uniform(5, 10)  # More realistic ratings
                if "entrance_fee" not in loc:
                    loc["entrance_fee"] = np.random.uniform(5, 100)
                if "duration" not in loc:
                    loc["duration"] = np.random.randint(45, 120)  # More realistic durations
            elif loc["type"] == "hotel":
                # Set hotel duration to 0 (no time spent at hotel for activities)
                loc["duration"] = 0
        
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
            "max_iterations": 5000, # 5000,
            "segment_size": 100, #100,
            "time_limit": 3600,  # 1 hour time limit
            "seed": seed,  # For reproducibility
            "early_termination_iterations": 2000,  # Early termination if no improvement
            "weights_destroy": [1.0, 1.0, 1.0, 1.0, 1.0],  # Weights for destroy operators
            "weights_repair": [1.0, 1.0, 1.0],  # Weights for repair operators
            # "weights_destroy": [1.0],  # Weights for destroy operators
            # "weights_repair": [1.0],  # Weights for repair operators
            "objective_weights": [0.3, 0.3, 0.4],  # Weight for objective function
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
        hotel_name="Marina Bay Sands",  # Optional: specific hotel name
        budget=500,  # Total budget in SGD
        num_days=3,  # Number of days
        max_attractions=16,  # Optional: limit number of attractions  
        max_hawkers=12  # Optional: limit number of hawkers
    )