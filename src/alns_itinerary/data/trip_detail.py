# get prices, distance, and time to go from point A to point B
import re
import json
import datetime
# from google_maps.get_directions import get_transit_directions


def calculate_car_fare(distance_m, flag_down=4.8):
    # Start with flag-down fare (covers first 1km)
    fare = flag_down
    
    # Calculate distance charge
    if distance_m > 1000:
        remaining_m = distance_m - 1000  # First 1 km is covered in flag-down

        if remaining_m <= 9000:
            # Charge $0.26 per 400m up to 10km total
            fare += (remaining_m // 400) * 0.26
            # Add partial unit if there's a remainder
            if remaining_m % 400 > 0:
                fare += 0.26
        else:
            # First 9km after flag-down
            fare += (9000 // 400) * 0.26
            
            # Calculate remaining distance after 10km total
            ultra_remaining = remaining_m - 9000
            
            # Beyond 10km, charge $0.26 per 350m
            fare += (ultra_remaining // 350) * 0.26
            # Add partial unit if there's a remainder
            if ultra_remaining % 350 > 0:
                fare += 0.26
    
    return round(fare, 2)

def calculate_public_transport_fare(distance_km):
    fare_table_brackets = [
        (3.2, 1.19), (4.2, 1.29), (5.2, 1.40), (6.2, 1.50), (7.2, 1.59),
        (8.2, 1.66), (9.2, 1.73), (10.2, 1.77), (11.2, 1.81), (12.2, 1.85),
        (13.2, 1.89), (14.2, 1.93), (15.2, 1.98), (16.2, 2.02), (17.2, 2.06),
        (18.2, 2.10), (19.2, 2.14), (20.2, 2.17), (21.2, 2.20), (22.2, 2.23),
        (23.2, 2.26), (24.2, 2.28), (25.2, 2.30), (26.2, 2.32), (27.2, 2.33),
        (28.2, 2.34), (29.2, 2.35), (30.2, 2.36), (31.2, 2.37), (32.2, 2.38),
        (33.2, 2.39), (34.2, 2.40), (35.2, 2.41), (36.2, 2.42), (37.2, 2.43),
        (38.2, 2.44), (39.2, 2.45), (40.2, 2.46), (float('inf'), 2.47)
    ]
    
    for limit, fare in fare_table_brackets:
        if distance_km <= limit:
            return fare
    
    return None  # Should never reach here


def get_trip_details(route_data, sort_priority="price", departure_time=datetime.datetime.now()):
    trip_details = []
    for route in route_data["routes"]:
        # accumulate the fares for the total price of this trip
        if route["steps"][0]["travel_mode"] == "DRIVING":
            # we assume use of grab / taxi
            accumulated_price = calculate_car_fare(route["distance"]["value"], flag_down=4.8)
            # departure time defaults to N/A, so put departure time as current time
            arriving_time = departure_time + datetime.timedelta(seconds=route["duration"]["value"])
            route["departure_time"] = departure_time.strftime("%-I:%M %p")
            route["arrival_time"] = arriving_time.strftime("%-I:%M %p")

        else:
            accumulated_price = 0
            for step in route["steps"]:
                # doesn't need to pay when walking...
                if step["travel_mode"] == "TRANSIT":
                    distance_km = float(step["distance"].split()[0])
                    accumulated_price += calculate_public_transport_fare(distance_km)
        
        trip_details.append({
            "price_sgd": accumulated_price,
            "distance_km": route["distance"]["value"] / 1000,
            "duration_minutes": round(route["duration"]["value"] / 60),
            "departure_time": route["departure_time"].replace("\u202f", " "),
            "arrival_time": route["arrival_time"].replace("\u202f", " "),
            "steps": route["steps"],
        })

    return sorted(trip_details, key=lambda x: (x["price_sgd"], x["arrival_time"]) if sort_priority == "price" else x["arrival_time"])
