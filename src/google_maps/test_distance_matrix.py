#!/usr/bin/env python3
"""
Test script for the Google Maps Routes API Distance Matrix

This script demonstrates how to use the compute_route_matrix function
to generate a route matrix between multiple waypoints in Singapore.
"""

import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from google_maps_client import GoogleMapsClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define sample waypoints for Singapore attractions and food centers
SAMPLE_WAYPOINTS = [
    ["Gardens by the Bay", 1.2816, 103.8636],
    ["Marina Bay Sands", 1.2834, 103.8607],
    ["Sentosa Island", 1.2494, 103.8303],
    # ["Singapore Zoo", 1.4043, 103.7930],
    # ["Maxwell Food Centre", 1.2803, 103.8451],
    # ["Lau Pa Sat", 1.2807, 103.8505],
    # ["Newton Food Centre", 1.3138, 103.8381]
]

def process_route_matrix(matrix_response, waypoints):
    """
    Process the raw route matrix response into a more usable format
    
    Args:
        matrix_response (list): Raw response from the API
        waypoints (list): List of waypoints [name, lat, lng]
        
    Returns:
        dict: Processed matrix with useful route information
    """
    if not matrix_response:
        logger.error("No matrix response to process")
        return None
        
    # Extract waypoint names
    waypoint_names = [wp[0] for wp in waypoints]
    
    # Create results dictionary
    results = {
        "metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_locations": len(waypoints),
            "num_routes": 0
        },
        "locations": {
            f"loc_{i}": {
                "id": f"loc_{i}",
                "name": waypoint[0],
                "lat": waypoint[1],
                "lng": waypoint[2]
            }
            for i, waypoint in enumerate(waypoints)
        },
        "routes": {}
    }
    
    # Process each route in the matrix
    for route in matrix_response:
        # Skip invalid routes
        if route.get('condition') != 'ROUTE_EXISTS':
            continue
            
        origin_idx = route.get('originIndex')
        dest_idx = route.get('destinationIndex')
        
        # Skip self-routes (optional, depending on your needs)
        if origin_idx == dest_idx:
            continue
            
        # Get location names
        origin_name = waypoint_names[origin_idx]
        dest_name = waypoint_names[dest_idx]
        
        # Create route ID
        route_id = f"loc_{origin_idx}_to_loc_{dest_idx}"
        
        # Extract distance and duration
        distance_meters = route.get('distanceMeters', 0)
        duration_seconds = 0
        
        # Parse duration string (format: "123s")
        if 'duration' in route:
            duration_str = route['duration']
            if duration_str.endswith('s'):
                duration_seconds = int(duration_str[:-1])
        
        # Extract fare information (if available)
        fare = 0
        currency = "SGD"  # Default to Singapore Dollar
        
        if 'travelAdvisory' in route and 'transitFare' in route['travelAdvisory']:
            fare_info = route['travelAdvisory']['transitFare']
            
            # Get currency code
            if 'currencyCode' in fare_info:
                currency = fare_info['currencyCode']
                
            # Calculate fare from units and nanos
            if 'units' in fare_info:
                fare = int(fare_info['units'])
                if 'nanos' in fare_info:
                    fare += fare_info['nanos'] / 1_000_000_000
        
        # Store route information
        results["routes"][route_id] = {
            "origin_id": f"loc_{origin_idx}",
            "destination_id": f"loc_{dest_idx}",
            "origin_name": origin_name,
            "destination_name": dest_name,
            "distance_km": distance_meters / 1000,
            "duration_minutes": duration_seconds / 60,
            "price_sgd": fare,
            "currency": currency
        }
        
        # Increment route count
        results["metadata"]["num_routes"] += 1
    
    return results

def test_route_matrix():
    """Test the compute_route_matrix function with sample waypoints"""
    try:
        # Initialize Google Maps client
        maps_client = GoogleMapsClient()
        logger.info("Google Maps client initialized")
        
        # Log waypoints
        logger.info(f"Testing with {len(SAMPLE_WAYPOINTS)} waypoints:")
        for i, (name, lat, lng) in enumerate(SAMPLE_WAYPOINTS):
            logger.info(f"  {i+1}. {name}: ({lat}, {lng})")
        
        # Create output directory
        os.makedirs("data/googleMaps/matrix", exist_ok=True)
        
        # Test transit mode
        logger.info("Computing transit route matrix...")
        transit_response = maps_client.compute_route_matrix(
            waypoints=SAMPLE_WAYPOINTS,
            mode="transit",
            departure_time=datetime(2025, 3, 20, 10, 0, 0)
        )
        
        if transit_response:
            # Save raw response
            with open("data/googleMaps/matrix/raw_transit_matrix.json", "w") as f:
                json.dump(transit_response, f, indent=2)
            logger.info("Raw transit matrix saved to data/googleMaps/matrix/raw_transit_matrix.json")
            
            # Process and save results
            transit_results = process_route_matrix(transit_response, SAMPLE_WAYPOINTS)
            if transit_results:
                with open("data/googleMaps/matrix/processed_transit_matrix.json", "w") as f:
                    json.dump(transit_results, f, indent=2)
                logger.info("Processed transit matrix saved to data/googleMaps/matrix/processed_transit_matrix.json")
                logger.info(f"Generated {transit_results['metadata']['num_routes']} transit routes")
                
                # Log a sample route
                if transit_results["routes"]:
                    sample_route = next(iter(transit_results["routes"].values()))
                    logger.info("Sample transit route:")
                    logger.info(f"  From: {sample_route['origin_name']}")
                    logger.info(f"  To: {sample_route['destination_name']}")
                    logger.info(f"  Distance: {sample_route['distance_km']:.2f} km")
                    logger.info(f"  Duration: {sample_route['duration_minutes']:.1f} minutes")
                    logger.info(f"  Fare: {sample_route['price_sgd']:.2f} {sample_route['currency']}")
        else:
            logger.error("Failed to compute transit route matrix")
        
        # Test driving mode
        logger.info("Computing driving route matrix...")
        driving_response = maps_client.compute_route_matrix(
            waypoints=SAMPLE_WAYPOINTS,
            mode="drive"
        )
        
        if driving_response:
            # Save raw response
            with open("data/googleMaps/matrix/raw_driving_matrix.json", "w") as f:
                json.dump(driving_response, f, indent=2)
            logger.info("Raw driving matrix saved to data/googleMaps/matrix/raw_driving_matrix.json")
            
            # Process and save results
            driving_results = process_route_matrix(driving_response, SAMPLE_WAYPOINTS)
            if driving_results:
                with open("data/googleMaps/matrix/processed_driving_matrix.json", "w") as f:
                    json.dump(driving_results, f, indent=2)
                logger.info("Processed driving matrix saved to data/googleMaps/matrix/processed_driving_matrix.json")
                logger.info(f"Generated {driving_results['metadata']['num_routes']} driving routes")
                
                # Log a sample route
                if driving_results["routes"]:
                    sample_route = next(iter(driving_results["routes"].values()))
                    logger.info("Sample driving route:")
                    logger.info(f"  From: {sample_route['origin_name']}")
                    logger.info(f"  To: {sample_route['destination_name']}")
                    logger.info(f"  Distance: {sample_route['distance_km']:.2f} km")
                    logger.info(f"  Duration: {sample_route['duration_minutes']:.1f} minutes")
                    # No fare for driving routes in the API
        else:
            logger.error("Failed to compute driving route matrix")
            
    except Exception as e:
        logger.error(f"Error in test_route_matrix: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    logger.info("Starting test of Google Maps Routes API Distance Matrix")
    test_route_matrix()
    logger.info("Test completed")