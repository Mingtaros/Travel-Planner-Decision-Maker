import os
import json
from datetime import datetime
from google_maps_client import GoogleMapsClient

def get_transit_directions(origin, destination, departure_time=None, output_dir="../data/googleMaps/"):
    """
    Get transit directions between origin and destination and save to JSON
    
    Args:
        origin (str): Starting address
        destination (str): Destination address
        output_dir (str): Directory to save output file
        
    Returns:
        dict: Parsed route data
    """
    try:
        # Create Maps client
        maps_client = GoogleMapsClient()
        
        print(f"Fetching public transport directions from '{origin}' to '{destination}'...")
        
        # Get directions
        routes = maps_client.get_route_directions(origin, destination, departure_time)
        
        if not routes:
            print("Error: Could not get directions.")
            return None
            
        # Parse routes
        route_data = maps_client.parse_routes_to_json(routes, origin, destination)
        
        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        route_file = os.path.join(output_dir, "transit_directions.json")
        
        with open(route_file, "w") as f:
            json.dump(route_data, f, indent=2)
        print(f"Directions saved to {route_file}")
        
        # Print summary
        print(f"\nFound {route_data['num_routes']} route options.")
        
        return route_data
        
    except Exception as e:
        print(f"Error getting transit directions: {e}")
        return None

if __name__ == "__main__":
    # Example usage when run directly
    origin = 'Singpost Centre, 10 Eunos Rd 8, Singapore 408600'
    destination = 'Singapore Management University, 81 Victoria St, Singapore 188065'
    departure_time = None
    
    get_transit_directions(origin, destination, departure_time)