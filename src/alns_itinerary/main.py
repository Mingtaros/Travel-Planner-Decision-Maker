import os
import logging
import numpy as np
from datetime import datetime

# Import ALNS components
from alns.alns_core import ALNS
from problem.itinerary_problem import TravelItineraryProblem
from data.transport_utils import get_transport_matrix, get_all_locations
from utils.export_itinerary import export_itinerary
from utils.google_maps_client import GoogleMapsClient
from data.location_utils import (
    get_hotel_waypoint, 
    integrate_hotel_with_locations, 
    filter_locations
)
from utils import SolutionVisualizer

def setup_logging():
    """
    Set up logging configuration
    """
    # Create logs directory if it doesn't exist
    os.makedirs("log", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"log/alns_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

def main(
    hotel_name=None, 
    budget=500, 
    num_days=3, 
    max_attractions=None, 
    max_hawkers=None
):
    """
    Main function to run ALNS optimization for travel itinerary
    
    Args:
        hotel_name (str, optional): Name of the hotel to start the trip
        budget (float): Total budget for the trip
        num_days (int): Number of days in the trip
        max_attractions (int, optional): Maximum number of attractions to consider
        max_hawkers (int, optional): Maximum number of hawkers to consider
    """
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Get hotel waypoint
        logger.info(f"Determining hotel location: {hotel_name or 'Using default'}")
        hotel = get_hotel_waypoint(hotel_name)
        
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
        
        # Validate integration results
        if updated_locations is None or updated_matrix is None:
            logger.error("Failed to integrate hotel with locations and transport matrix")
            return
        
        # For all locations, add missing data with randomization
        for loc in updated_locations:
            if loc["type"] == "hawker":
                if "rating" not in loc:
                    loc["rating"] = np.random.uniform(0, 5)
                if "avg_food_price" not in loc:
                    loc["avg_food_price"] = np.random.uniform(5, 15)
                if "duration" not in loc:
                    loc["duration"] = 60  # standardize 60 mins
            elif loc["type"] == "attraction":
                if "satisfaction" not in loc:
                    loc["satisfaction"] = np.random.uniform(0, 10)
                if "entrance_fee" not in loc:
                    loc["entrance_fee"] = np.random.uniform(5, 100)
                if "duration" not in loc:
                    loc["duration"] = np.random.randint(30, 90)
            elif loc["type"] == "hotel":
                # Set hotel duration to 0 (no time spent at hotel for activities)
                loc["duration"] = 0
        
        # Create problem instance
        problem = TravelItineraryProblem(
            budget=budget, 
            locations=updated_locations, 
            transport_matrix=updated_matrix,
            num_days=num_days
        )
        
        # Check problem feasibility
        if not problem.is_feasible:
            logger.error("Problem is not feasible with the given constraints")
            return
        
        # Configure ALNS parameters
        alns_config = {
            "max_iterations": 5000,
            "segment_size": 100,
            "time_limit": 3600,  # 1 hour time limit
            "seed": 42,  # For reproducibility
        }
        
        # Initialize and run ALNS
        logger.info("Starting ALNS optimization...")
        alns = ALNS(problem, **alns_config)
        results = alns.run(verbose=True)
        
        # Export the best solution
        export_path = f"results/best_itinerary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        exported_file = export_itinerary(problem, results['best_solution'], export_path)
        
        # Log optimization results
        logger.info(f"Optimization completed. Best solution exported to {exported_file}")
        logger.info(f"Best solution objective: {results['stats']['best_objective']:.4f}")
        logger.info(f"Total cost: ${results['best_evaluation']['total_cost']:.2f}")
        logger.info(f"Total travel time: {results['best_evaluation']['total_travel_time']:.2f} minutes")
        logger.info(f"Total satisfaction: {results['best_evaluation']['total_satisfaction']:.2f}")
        logger.info(f"Solution feasible: {results['stats']['best_is_feasible']}")
        
        
        # # Generate comprehensive visualizations
        # visualizations = SolutionVisualizer.generate_comprehensive_visualization(
        #     problem, results['best_solution']
        # )
        
        return results
    
    except Exception as e:
        logger.error(f"An error occurred during optimization: {e}", exc_info=True)

if __name__ == "__main__":
    # Example usage with default parameters
    main(
        hotel_name="Marina Bay Sands",  # Optional: specific hotel name
        budget=500,  # Total budget in SGD
        num_days=3,  # Number of days
        max_attractions=None,  # Optional: limit number of attractions
        max_hawkers=None  # Optional: limit number of hawkers
    )