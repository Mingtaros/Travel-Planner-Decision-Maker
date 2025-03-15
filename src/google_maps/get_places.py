import os
import json
import time
from google_maps.google_maps_client import GoogleMapsClient

def get_places_details(place_list, output_dir="../data/googleMaps/places/"):
    """
    Get details for a list of places and save to individual JSON files
    
    Args:
        place_list (list): List of place names or IDs to fetch details for
        output_dir (str): Directory to save output files
        
    Returns:
        dict: Dictionary mapping place names to their parsed details
    """
    try:
        # Create Maps client
        maps_client = GoogleMapsClient()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Store all place details
        all_places = {}
        
        # Process each place
        for i, place in enumerate(place_list):
            print(f"Fetching details for place {i+1}/{len(place_list)}: '{place}'...")
            
            # Determine if this is a place_id or place_name
            if place.startswith("ChI") and len(place) > 20:  # Likely a place_id
                place_details = maps_client.get_place_details(place_id=place)
            else:
                place_details = maps_client.get_place_details(place_name=place)
            
            if not place_details:
                print(f"No details found for '{place}'")
                continue
                
            # Parse place details
            parsed_place = maps_client.parse_place_details(place_details)
            
            if not parsed_place:
                print(f"Could not parse details for '{place}'")
                continue
                
            # Store in our dictionary
            all_places[place] = parsed_place
            
            # Save to individual file
            safe_filename = place.replace(" ", "_").replace(",", "").replace("/", "_")[:50]
            place_file = os.path.join(output_dir, f"{safe_filename}.json")
            
            with open(place_file, "w") as f:
                json.dump(parsed_place, f, indent=2)
            print(f"Details for '{place}' saved to {place_file}")
            
            # Avoid rate limiting
            if i < len(place_list) - 1:
                time.sleep(0.5)  # Sleep between requests to avoid hitting rate limits
        
        # Save all places to a combined file
        combined_file = os.path.join(output_dir, "all_places.json")
        with open(combined_file, "w") as f:
            json.dump(all_places, f, indent=2)
        print(f"All place details saved to {combined_file}")
        
        return all_places
        
    except Exception as e:
        print(f"Error getting place details: {e}")
        return None

if __name__ == "__main__":
    # Example usage when run directly
    places = [
        "Singapore Management University, Singapore",
        "National University of Singapore",
        "Gardens by the Bay, Singapore",
        "Marina Bay Sands, Singapore",
        "Sentosa Island, Singapore"
    ]
    
    get_places_details(places)