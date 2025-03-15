import os
import json
from datetime import datetime
import googlemaps
from dotenv import load_dotenv
import requests

class GoogleMapsClient:
    def __init__(self, api_key=None):
        """
        Initialize the Google Maps client
        
        Args:
            api_key (str, optional): Google Maps API key. If None, will try to load from environment.
        """
        # Load from environment if not provided
        if api_key is None:
            load_dotenv()
            self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
            
        if not self.api_key:
            raise ValueError("Google Maps API key is required")
            
        # Initialize the Google Maps client
        self.gmaps = googlemaps.Client(key=self.api_key)
    
    def compute_route_matrix(self, waypoints, mode="transit", departure_time=None):
        """
        Compute a route matrix using the Google Maps Routes API
        
        Args:
            waypoints (list): List of [lat, lng] coordinates or [name, lat, lng] tuples
            mode (str): Travel mode, "transit" or "driving"
            departure_time (datetime): Departure time
            
        Returns:
            dict: Raw API response
        """
        
        # If no departure time specified, use current time
        if departure_time is None:
            departure_time = datetime.now()
            
        # Convert mode to the format expected by the API
        travel_mode = mode.upper()
        
        # Format waypoints for the API
        formatted_waypoints = []
        for point in waypoints:
            # Handle both [lat, lng] and [name, lat, lng] formats
            if len(point) == 2:
                lat, lng = point
            elif len(point) == 3:
                _, lat, lng = point
            else:
                print(f"Invalid waypoint format: {point}")
                continue
                
            formatted_waypoints.append({
                "waypoint": {
                    "location": {
                        "latLng": {
                            "latitude": lat,
                            "longitude": lng
                        }
                    }
                }
            })
        
        # Prepare request payload
        payload = {
            "origins": formatted_waypoints,
            "destinations": formatted_waypoints,
            "travelMode": travel_mode,
        }
        
        # Add departure time for transit mode
        if travel_mode == "TRANSIT":
            payload["departureTime"] = departure_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Define the API endpoint
        url = 'https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix'
        
        # Define required headers
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': self.api_key,
            # 'X-Goog-FieldMask': 'originIndex,destinationIndex,duration,distanceMeters,status,condition,travelAdvisory'
            'X-Goog-FieldMask': '*'  # Get all fields
        }
        
        try:
            # Make the API request
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            
            # Check if the request was successful
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error computing route matrix: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error computing route matrix: {e}")
            return None
    
    def compute_distance_matrix(self, origins, destinations, mode="transit", departure_time=None):
        """
        Get a distance matrix between multiple origins and destinations
        """
        # If no departure time specified, use current time
        if departure_time is None:
            departure_time = datetime.now()
        
        try:
            # Make the distance matrix request
            matrix = self.gmaps.distance_matrix(
                origins=origins,
                destinations=destinations,
                mode=mode,
                departure_time=departure_time
            )
            
            return matrix
            
        except Exception as e:
            print(f"Error getting distance matrix: {e}")
            return None
    
    def get_route_directions(self, origin, destination, mode="transit", departure_time=None):
        """
        Get public transport and driving directions between two addresses
        
        Args:
            origin (str): Starting address
            destination (str): Destination address
            mode (str): Travel mode, either "transit" or "driving"
            departure_time (datetime, optional): Departure time. Defaults to now.
        
        Returns:
            list: Directions results
        """
        # If no departure time specified, use current time
        if departure_time is None:
            departure_time = datetime.now()
        
        try:
            # Make the directions request
            directions = self.gmaps.directions(
                origin=origin,
                destination=destination,
                mode=mode,
                departure_time=departure_time,
                alternatives=False
            )
            
            return directions
            
        except Exception as e:
            print(f"Error getting directions: {e}")
            return None
    
    def get_place_details(self, place_id=None, place_name=None):
        """
        Get details about a place using either place_id or by searching for place_name
        
        Args:
            place_id (str, optional): Google Maps place ID
            place_name (str, optional): Name of the place to search for
            
        Returns:
            dict: Place details including opening hours, etc.
        """
        if not place_id and not place_name:
            raise ValueError("Either place_id or place_name must be provided")
            
        try:
            # If only place_name is provided, search for the place first
            if not place_id and place_name:
                places_result = self.gmaps.places(
                    query=place_name,
                    language="en"
                )
                
                if not places_result.get("results"):
                    print(f"No places found for query: {place_name}")
                    return None
                    
                # Use the first result's place_id
                place_id = places_result["results"][0]["place_id"]
            
            # Get place details
            place_details = self.gmaps.place(
                place_id=place_id,
                fields=[
                    "name", "formatted_address", "formatted_phone_number",
                    "opening_hours", "current_opening_hours", "website", "price_level", "rating", "geometry"
                ]
            )
            
            return place_details
            
        except Exception as e:
            print(f"Error getting place details: {e}")
            return None
    
    def parse_routes_to_json(self, routes, origin_address, destination_address):
        """
        Parse Google Maps routes into a structured JSON format
        
        Args:
            routes (list): List of route dictionaries from Google Maps API
            origin_address (str): Original input origin address
            destination_address (str): Original input destination address
            
        Returns:
            dict: Structured JSON with route information
        """
        parsed_routes = []
        
        for i, route in enumerate(routes):
            leg = route["legs"][0]  # Assuming single leg journey
            
            # Extract base route information
            parsed_route = {
                "route_number": i + 1,
                "origin": leg["start_address"],
                "destination": leg["end_address"],
                "distance": {
                    "text": leg["distance"]["text"],
                    "value": leg["distance"]["value"]  # Distance in meters
                },
                "duration": {
                    "text": leg["duration"]["text"],
                    "value": leg["duration"]["value"]  # Duration in seconds
                },
                "departure_time": leg.get("departure_time", {}).get("text", "N/A"),
                "arrival_time": leg.get("arrival_time", {}).get("text", "N/A"),
                "steps": []
            }
            
            # Parse each step of the journey
            for step in leg["steps"]:
                parsed_step = {
                    "travel_mode": step["travel_mode"],
                    "instructions": step.get("html_instructions", "").replace("<b>", "").replace("</b>", "").replace("<div>", ", ").replace("</div>", ""),
                    "distance": step["distance"]["text"],
                    "duration": step["duration"]["text"]
                }
                
                # Add transit-specific details if this is a transit step
                if step.get("travel_mode") == "TRANSIT":
                    transit = step["transit_details"]
                    parsed_step.update({
                        "departure_stop": transit["departure_stop"]["name"],
                        "departure_time": transit.get("departure_time", {}).get("text", "N/A"),
                        "arrival_stop": transit["arrival_stop"]["name"],
                        "arrival_time": transit.get("arrival_time", {}).get("text", "N/A"),
                        "line": {
                            "name": transit["line"].get("name", ""),
                            "short_name": transit["line"].get("short_name", ""),
                            "vehicle_type": transit["line"]["vehicle"]["type"]
                        },
                        "num_stops": transit.get("num_stops", 0)
                    })
                
                parsed_route["steps"].append(parsed_step)
            
            parsed_routes.append(parsed_route)
        
        # Create the final output structure
        output = {
            "query": {
                "origin": origin_address,
                "destination": destination_address,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "num_routes": len(parsed_routes),
            "routes": parsed_routes
        }
        
        return output
    
    def parse_place_details(self, place_details):
        """
        Parse Google Maps place details into a structured format
        
        Args:
            place_details (dict): Place details from Google Maps API
            
        Returns:
            dict: Structured place information
        """
        if not place_details or "result" not in place_details:
            return None
            
        result = place_details["result"]
        
        # Build a structured output
        parsed_place = {
            "name": result.get("name", ""),
            "address": result.get("formatted_address", ""),
            "phone": result.get("formatted_phone_number", ""),
            "website": result.get("website", ""),
            "price_level": result.get("price_level", 0),
            "current_opening_hours": result.get("current_opening_hours", ""),
            "rating": result.get("rating", 0),
            "location": {
                "lat": result.get("geometry", {}).get("location", {}).get("lat", 0),
                "lng": result.get("geometry", {}).get("location", {}).get("lng", 0)
            }
        }
        
        # Parse opening hours if available
        if "opening_hours" in result:
            hours = result["opening_hours"]
            parsed_place["opening_hours"] = {
                "open_now": hours.get("open_now", False),
                "weekday_text": hours.get("weekday_text", [])
            }
        
        return parsed_place
    
    def get_route_and_place_details(self, origin, destination, mode, departure_time=None, save_to_file=True, output_dir="./"):
        """
        Get both route directions and destination place details in one call
        
        Args:
            origin (str): Starting address
            destination (str): Destination address
            save_to_file (bool): Whether to save results to JSON files
            output_dir (str): Directory to save output files
            
        Returns:
            tuple: (route_data, place_data)
        """
        print(f"Fetching public transport directions from '{origin}' to '{destination}'...")
        
        # Get directions
        routes = self.get_route_directions(origin, destination, mode, departure_time)
        
        if not routes:
            print("Error: Could not get directions.")
            return None, None
            
        # Parse routes
        route_data = self.parse_routes_to_json(routes, origin, destination)
        
        # Get place details for destination
        print(f"Fetching details for destination: '{destination}'...")
        place_details = self.get_place_details(place_name=destination)
        place_data = self.parse_place_details(place_details)
        
        # Save to files if requested
        if save_to_file:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save route data
            route_file = os.path.join(output_dir, "transit_directions.json")
            with open(route_file, "w") as f:
                json.dump(route_data, f, indent=2)
            print(f"Directions saved to {route_file}")
            
            # Save place data if available
            if place_data:
                place_file = os.path.join(output_dir, "place_details.json")
                with open(place_file, "w") as f:
                    json.dump(place_data, f, indent=2)
                print(f"Place details saved to {place_file}")
        
        return route_data, place_data