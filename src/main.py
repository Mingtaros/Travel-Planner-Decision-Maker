# main.py
import sys
import json
from get_directions import get_transit_directions
from get_places import get_places_details

def main():
    """
    Main function to process command line arguments
    
    Usage:
    python main.py directions <origin> <destination>
    python main.py places <place_list_file>

    python main.py directions 'Singpost Centre, Singapore' 'SMU, Singapore'
    python main.py places ../places.txt
    """
    command = sys.argv[1].lower()
    
    if command == "directions" and len(sys.argv) >= 4:
        # Get transit directions
        origin = sys.argv[2]
        destination = sys.argv[3]
        departure_time = None
        
        get_transit_directions(origin, destination, departure_time)
    
    elif command == "places" and len(sys.argv) >= 3:
        # Get place details from a file listing places
        place_file = sys.argv[2]
        
        try:
            with open(place_file, 'r') as f:
                places = [line.strip() for line in f if line.strip()]
                
            if not places:
                print(f"No places found in {place_file}")
                return
                
            print(f"Found {len(places)} places to process")
            get_places_details(places)
            
        except FileNotFoundError:
            print(f"File not found: {place_file}")
        except Exception as e:
            print(f"Error reading place list: {e}")

if __name__ == "__main__":
    main()