#!/usr/bin/env python3
"""
Store waypoint data (location coordinates) in the database

This script:
1. Fetches attractions and hawker centers from the database
2. Geocodes them using the Google Maps API
3. Stores the coordinates in a waypoints table
4. Provides functions to retrieve waypoints instead of re-geocoding

Run this script once to cache location coordinates, then use the
fetch_waypoints function in your route matrix generator.
"""

import os
import logging
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv
from utils.google_maps_client import GoogleMapsClient

# Set up logging
os.makedirs("log", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/waypoints_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("waypoints")

# Load environment variables
load_dotenv()

db_user = os.getenv("MYSQL_USER")
db_port = os.getenv("MYSQL_PORT")
db_pwd = os.getenv("MYSQL_PASSWORD")
db_database = os.getenv("MYSQL_DATABASE")
db_host = os.getenv("MYSQL_HOST")
# Database connection parameters
DB_CONFIG = {
    'host': db_host,
    'port': db_port,
    'user': db_user,
    'password': db_pwd,
    'database': db_database
}

def connect_to_database():
    """Connect to the MySQL database"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        logger.info("Connected to MySQL database")
        return connection
    except mysql.connector.Error as e:
        logger.error(f"Error connecting to MySQL database: {e}")
        return None

def fetch_locations(connection):
    """Fetch all attractions and hawker centers from the database"""
    locations = []
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Fetch attractions
        cursor.execute("SELECT aid, aname FROM attractions")
        attractions = cursor.fetchall()
        for attraction in attractions:
            locations.append({
                'id': f"A{attraction['aid']}",
                'name': attraction['aname'],
                'type': 'attraction'
            })
        
        # Fetch hawker centers
        cursor.execute("SELECT fid, name, address FROM foodcentre")
        hawkers = cursor.fetchall()
        for hawker in hawkers:
            locations.append({
                'id': f"H{hawker['fid']}",
                'name': hawker['name'],
                'type': 'hawker',
                'address': hawker.get('address', '')
            })
        
        cursor.close()
        logger.info(f"Fetched {len(locations)} locations from database ({len(attractions)} attractions, {len(hawkers)} hawker centers)")
        return locations
    
    except mysql.connector.Error as e:
        logger.error(f"Error fetching locations: {e}")
        return []

def create_waypoints_table(connection):
    """Create the waypoints table in the database if it doesn't exist"""
    try:
        cursor = connection.cursor()
        
        # Create table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS waypoints (
            id INT AUTO_INCREMENT PRIMARY KEY,
            location_id VARCHAR(50) NOT NULL,
            name VARCHAR(255) NOT NULL,
            type VARCHAR(50) NOT NULL,
            latitude DOUBLE NOT NULL,
            longitude DOUBLE NOT NULL,
            address VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY unique_location (location_id)
        )
        """
        cursor.execute(create_table_query)
        connection.commit()
        cursor.close()
        
        logger.info("waypoints table created or already exists")
        return True
    except mysql.connector.Error as e:
        logger.error(f"Error creating waypoints table: {e}")
        return False

def geocode_locations(maps_client, locations):
    """Geocode locations to get their coordinates"""
    geocoded_locations = []
    
    for location in locations:
        try:
            # Prepare search query
            if location['type'] == 'hawker' and location.get('address'):
                query = location['address']
            else:
                query = f"{location['name']}, Singapore"
            
            # Get geocode
            place_details = maps_client.get_place_details(place_name=query)
            place_data = maps_client.parse_place_details(place_details)
            
            if place_data and 'location' in place_data:
                geocoded_location = location.copy()
                geocoded_location['lat'] = place_data['location']['lat']
                geocoded_location['lng'] = place_data['location']['lng']
                geocoded_location['address'] = place_data.get('address', '')
                geocoded_locations.append(geocoded_location)
                
                logger.info(f"Geocoded {location['name']}")
            else:
                logger.warning(f"Could not geocode {location['name']}")
        
        except Exception as e:
            logger.error(f"Error geocoding {location['name']}: {e}")
    
    logger.info(f"Successfully geocoded {len(geocoded_locations)} out of {len(locations)} locations")
    return geocoded_locations

def store_waypoints(connection, geocoded_locations):
    """Store geocoded locations in the waypoints table"""
    try:
        # Ensure the waypoints table exists
        if not create_waypoints_table(connection):
            return False
        
        # Prepare data for insertion
        waypoints_data = []
        for location in geocoded_locations:
            waypoint = (
                location['id'],
                location['name'],
                location['type'],
                location['lat'],
                location['lng'],
                location.get('address', '')
            )
            waypoints_data.append(waypoint)
        
        # Insert data
        cursor = connection.cursor()
        
        # Use REPLACE INTO to update existing waypoints or insert new ones
        insert_query = """
        REPLACE INTO waypoints (
            location_id, name, type, latitude, longitude, address
        ) VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        cursor.executemany(insert_query, waypoints_data)
        connection.commit()
        
        logger.info(f"Stored {len(waypoints_data)} waypoints in database")
        return True
    except mysql.connector.Error as e:
        logger.error(f"Error storing waypoints in database: {e}")
        return False

def fetch_waypoints(connection, limit=None):
    """Fetch waypoints from the database"""
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Query waypoints
        query = "SELECT * FROM waypoints"
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        waypoints_data = cursor.fetchall()
        cursor.close()
        
        if not waypoints_data:
            logger.warning("No waypoints found in database")
            return []
        
        # Convert to the format expected by the route matrix generator
        waypoints = []
        locations = []
        
        for wp in waypoints_data:
            # Format for locations list
            location = {
                'id': wp['location_id'],
                'name': wp['name'],
                'type': wp['type'],
                'lat': wp['latitude'],
                'lng': wp['longitude'],
                'address': wp['address']
            }
            locations.append(location)
            
            # Format for waypoints list (expected by Google Maps API)
            waypoint = [
                wp['name'],
                wp['latitude'],
                wp['longitude']
            ]
            waypoints.append(waypoint)
        
        logger.info(f"Fetched {len(waypoints)} waypoints from database")
        return locations, waypoints
    
    except mysql.connector.Error as e:
        logger.error(f"Error fetching waypoints from database: {e}")
        return [], []

def populate_waypoints_database():
    """
    Main function to populate the waypoints database
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Starting waypoints database population...")
    
    # Connect to database
    connection = connect_to_database()
    if not connection:
        return False
    
    try:
        # Initialize Google Maps client
        maps_client = GoogleMapsClient()
        logger.info("Google Maps client initialized")
        
        # Fetch locations from database
        locations = fetch_locations(connection)
        if not locations:
            logger.error("No locations found in database")
            return False
        
        # Geocode locations to get coordinates
        geocoded_locations = geocode_locations(maps_client, locations)
        if not geocoded_locations:
            logger.error("No locations could be geocoded")
            return False
        
        # Store waypoints in database
        success = store_waypoints(connection, geocoded_locations)
        
        return success
        
    except Exception as e:
        logger.error(f"Error populating waypoints database: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        # Close database connection
        if connection and connection.is_connected():
            connection.close()
            logger.info("Database connection closed")

def main():
    """Main function to demonstrate the script"""
    if populate_waypoints_database():
        logger.info("Successfully populated waypoints database")
        
        # Test fetching the waypoints
        connection = connect_to_database()
        if connection:
            try:
                locations, waypoints = fetch_waypoints(connection)
                
                if locations and waypoints:
                    logger.info(f"Successfully fetched {len(locations)} waypoints from database")
                    
                    # Print a sample waypoint
                    sample = locations[0]
                    logger.info(f"Sample waypoint: {sample['name']} ({sample['id']})")
                    logger.info(f"  Coordinates: {sample['lat']}, {sample['lng']}")
                    logger.info(f"  Type: {sample['type']}")
                    if sample.get('address'):
                        logger.info(f"  Address: {sample['address']}")
                else:
                    logger.error("Failed to fetch waypoints from database")
            finally:
                connection.close()
    else:
        logger.error("Failed to populate waypoints database")

if __name__ == "__main__":
    main()