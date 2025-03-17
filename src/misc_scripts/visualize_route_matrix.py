#!/usr/bin/env python3
"""
Visualize the route matrix on a map using Folium

This script reads the generated route matrix and creates an interactive map showing:
1. All attractions and hawker centers as markers
2. Routes between selected locations
3. Statistics about distances, durations, and costs
"""

import os
import json
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import webbrowser
from google_maps.google_maps_client import GoogleMapsClient

def load_route_matrix(file_path="data/route_matrix.json"):
    """Load the route matrix from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded route matrix with {len(data['locations'])} locations and {len(data['routes'])} routes")
        return data
    except Exception as e:
        print(f"Error loading route matrix: {e}")
        return None

def get_location_coordinates(maps_client, location_details):
    """Get coordinates for each location using Google Maps API"""
    coordinates = {}
    
    for loc_id, details in location_details.items():
        place_name = f"{details['name']}, Singapore"
        
        # Use address if available (for hawker centers)
        if details['type'] == 'hawker' and details.get('address'):
            place_name = details['address']
        
        # Try to get place details from Google Maps
        try:
            place_details = maps_client.get_place_details(place_name=place_name)
            place_data = maps_client.parse_place_details(place_details)
            
            if place_data and place_data.get('location'):
                coordinates[loc_id] = (
                    place_data['location']['lat'], 
                    place_data['location']['lng']
                )
                print(f"Found coordinates for {details['name']}")
            else:
                print(f"Warning: Could not find coordinates for {details['name']}")
        except Exception as e:
            print(f"Error getting coordinates for {details['name']}: {e}")
    
    return coordinates

def create_map(matrix_data, coordinates):
    """Create an interactive map with locations and routes"""
    # Center the map on Singapore
    sg_center = [1.3521, 103.8198]
    m = folium.Map(location=sg_center, zoom_start=12)
    
    # Create marker clusters for attractions and hawker centers
    attraction_cluster = MarkerCluster(name="Attractions")
    hawker_cluster = MarkerCluster(name="Hawker Centers")
    
    # Add markers for each location
    for loc_id, details in matrix_data['locations'].items():
        if loc_id not in coordinates:
            continue
            
        location_name = details['name']
        lat, lng = coordinates[loc_id]
        
        # Create popup content
        popup_html = f"""
        <div style="width: 200px">
            <h4>{location_name}</h4>
            <p><b>Type:</b> {details['type'].capitalize()}</p>
            <p><b>Cost:</b> ${details['expenditure']}</p>
            <p><b>Time Spent:</b> {details['timespent']} hours</p>
        """
        
        # Add hawker-specific information
        if details['type'] == 'hawker':
            popup_html += f"""
            <p><b>Rating:</b> {details.get('rating', 'N/A')}/5</p>
            <p><b>Food Type:</b> {details.get('food_type', 'Various')}</p>
            <p><b>Best For:</b> {details.get('best_for', 'N/A')}</p>
            """
        
        popup_html += "</div>"
        
        # Create marker with appropriate icon and color
        if details['type'] == 'attraction':
            icon = folium.Icon(icon="camera", prefix="fa", color="blue")
            marker = folium.Marker(
                location=[lat, lng],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=location_name,
                icon=icon
            )
            attraction_cluster.add_child(marker)
        else:
            icon = folium.Icon(icon="utensils", prefix="fa", color="red")
            marker = folium.Marker(
                location=[lat, lng],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=location_name,
                icon=icon
            )
            hawker_cluster.add_child(marker)
    
    # Add clusters to map
    m.add_child(attraction_cluster)
    m.add_child(hawker_cluster)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def visualize_specific_route(m, matrix_data, coordinates, origin_id, destination_id):
    """Add a specific route to the map"""
    # Check if the route exists
    route_id = f"{origin_id}_to_{destination_id}"
    if route_id not in matrix_data['routes']:
        print(f"Route {route_id} not found in matrix")
        return m
    
    # Get route details
    route = matrix_data['routes'][route_id]
    origin_details = matrix_data['locations'][origin_id]
    destination_details = matrix_data['locations'][destination_id]
    
    # Check if coordinates exist for both locations
    if origin_id not in coordinates or destination_id not in coordinates:
        print(f"Coordinates missing for {origin_id} or {destination_id}")
        return m
    
    # Create a line between origin and destination
    folium.PolyLine(
        locations=[coordinates[origin_id], coordinates[destination_id]],
        color="green",
        weight=3,
        opacity=0.8,
        tooltip=f"{route['distance_km']:.1f} km | {route['duration_minutes']:.0f} min | ${route['price_sgd']:.2f}"
    ).add_to(m)
    
    # Add route information popup
    route_html = f"""
    <div style="width: 250px">
        <h4>Route: {origin_details['name']} to {destination_details['name']}</h4>
        <p><b>Distance:</b> {route['distance_km']:.1f} km</p>
        <p><b>Duration:</b> {route['duration_minutes']:.0f} minutes</p>
        <p><b>Cost:</b> ${route['price_sgd']:.2f}</p>
        <h5>Travel Modes:</h5>
        <ul>
    """
    
    for step in route['route_summary']:
        mode = step['mode']
        if mode == "TRANSIT":
            vehicle = step['vehicle_type'].lower()
            line = step['line']
            route_html += f"<li>{vehicle.capitalize()}: {line} ({step['duration']})</li>"
        else:
            route_html += f"<li>{mode.capitalize()}: {step['duration']}</li>"
    
    route_html += """
        </ul>
    </div>
    """
    
    # Add a marker for the route information
    mid_lat = (coordinates[origin_id][0] + coordinates[destination_id][0]) / 2
    mid_lng = (coordinates[origin_id][1] + coordinates[destination_id][1]) / 2
    
    folium.Marker(
        location=[mid_lat, mid_lng],
        popup=folium.Popup(route_html, max_width=300),
        tooltip=f"Route info: {origin_details['name']} → {destination_details['name']}",
        icon=folium.Icon(icon="info-sign", color="green")
    ).add_to(m)
    
    return m

def create_route_statistics(matrix_data):
    """Create DataFrame with route statistics for analysis"""
    routes_list = []
    
    for route_id, route in matrix_data['routes'].items():
        origin_id = route['origin_id']
        destination_id = route['destination_id']
        
        origin_details = matrix_data['locations'][origin_id]
        destination_details = matrix_data['locations'][destination_id]
        
        routes_list.append({
            'route_id': route_id,
            'origin': origin_details['name'],
            'origin_type': origin_details['type'],
            'destination': destination_details['name'],
            'destination_type': destination_details['type'],
            'distance_km': route['distance_km'],
            'duration_minutes': route['duration_minutes'],
            'price_sgd': route['price_sgd'],
            'travel_modes': '-'.join([step['mode'] for step in route['route_summary']])
        })
    
    # Create DataFrame
    df = pd.DataFrame(routes_list)
    
    # Calculate statistics
    stats = {
        'average_distance': df['distance_km'].mean(),
        'average_duration': df['duration_minutes'].mean(),
        'average_price': df['price_sgd'].mean(),
        'max_distance': df['distance_km'].max(),
        'max_duration': df['duration_minutes'].max(),
        'max_price': df['price_sgd'].max(),
        'attraction_to_hawker_avg_distance': df[
            (df['origin_type'] == 'attraction') & 
            (df['destination_type'] == 'hawker')
        ]['distance_km'].mean(),
        'attraction_to_hawker_avg_price': df[
            (df['origin_type'] == 'attraction') & 
            (df['destination_type'] == 'hawker')
        ]['price_sgd'].mean()
    }
    
    print("\nRoute Matrix Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    return df, stats

def main():
    """Main function to visualize the route matrix"""
    # Load route matrix
    matrix_data = load_route_matrix()
    if not matrix_data:
        return
    
    # Initialize Google Maps client to get coordinates
    maps_client = GoogleMapsClient()
    
    # Get coordinates for each location
    coordinates = get_location_coordinates(maps_client, matrix_data['locations'])
    
    # Create the map
    m = create_map(matrix_data, coordinates)
    
    # Example: Visualize some specific routes
    # Get first attraction and first hawker center
    attraction_ids = [loc_id for loc_id, details in matrix_data['locations'].items() 
                     if details['type'] == 'attraction'][:3]
    hawker_ids = [loc_id for loc_id, details in matrix_data['locations'].items() 
                 if details['type'] == 'hawker'][:3]
    
    # Visualize routes between them
    for a_id in attraction_ids:
        for h_id in hawker_ids:
            m = visualize_specific_route(m, matrix_data, coordinates, a_id, h_id)
    
    # Calculate statistics
    df, stats = create_route_statistics(matrix_data)
    
    # Save statistics to CSV
    df.to_csv("data/route_statistics.csv", index=False)
    print("Route statistics saved to data/route_statistics.csv")
    
    # Save the map
    output_file = "data/route_map.html"
    m.save(output_file)
    print(f"Map saved to {output_file}")
    
    # Open the map in the browser
    webbrowser.open('file://' + os.path.realpath(output_file))

if __name__ == "__main__":
    main()