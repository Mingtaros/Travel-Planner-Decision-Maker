"""
Export Itinerary as JSON
========================

This module exports optimized travel itineraries as structured JSON files for use in applications.

Features:
- Converts VRPSolution objects into detailed day-by-day itineraries
- Calculates realistic transit times between locations
- Includes rest periods and meal timing
- Provides budget breakdown by category (attractions, meals, transportation)
- Supports constraint violation reporting for debugging

Usage:
    result_file = export_json_itinerary(problem, solution, filename="my_itinerary.json")
"""

import os
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def format_time(minutes):
    """Convert minutes to HH:MM format."""
    hour = int(minutes // 60)
    minute = int(minutes % 60)
    return f"{hour:02d}:{minute:02d}"

def export_json_itinerary(problem, solution, stats=None, filename=None):
    """
    Export an optimized travel itinerary as a detailed JSON file.
    
    The JSON structure includes:
    - Trip summary (duration, budget, expenditure, satisfaction score)
    - Day-by-day itinerary with locations, arrival/departure times
    - Transit information between locations (mode, duration, cost)
    - Rest periods and meal timing
    - Budget breakdown by category
    - Constraint violations (if any)
    
    Args:
        problem: TravelItineraryProblem instance containing location data and constraints
        solution: VRPSolution instance with the optimized route
        stats: Evaluation statistics from the solution
        filename: Path to the output JSON file (optional, auto-generated if None)
        
    Returns:
        str: Path to the exported JSON file
    """
    # Get evaluation data from the VRPSolution
    evaluation = solution.evaluate()
    
    # Get daily routes from evaluation
    daily_routes = evaluation.get("daily_routes", [])
    
    # Create a base date for time calculations
    base_date = datetime.today().date()
    
    # Create itinerary JSON structure
    itinerary = {
        "trip_summary": {
            "duration": problem.NUM_DAYS,
            "total_budget": problem.budget,
            "actual_expenditure": evaluation["total_cost"],
            "total_travel_time": evaluation["total_travel_time"],
            "total_satisfaction": evaluation["total_satisfaction"],
            "objective_value": stats.get('best_objective', 0),
            "is_feasible": evaluation["is_feasible"],
            "starting_hotel": problem.locations[0]["name"]
        },
        "days": []
    }
    
    # Process each day
    for day_idx, route in enumerate(daily_routes):
        day_data = {
            "day": day_idx + 1,
            "date": (base_date + timedelta(days=day_idx)).strftime("%Y-%m-%d"),
            "locations": []
        }
        
        # Add hotel as starting point if not already included
        if not route or route[0]["type"] != "hotel":
            day_start_time = problem.START_TIME  # 9 AM in minutes
            start_time_str = f"{int(day_start_time // 60):02d}:{int(day_start_time % 60):02d}"
            hotel_location = problem.locations[0]
            
            day_data["locations"].append({
                "name": problem.locations[0]["name"],
                "type": "hotel",
                "position": "start",
                "arrival_time": start_time_str,
                "departure_time": start_time_str,
                "lat": hotel_location.get("lat"),
                "lng": hotel_location.get("lng"),
                "transit_from_prev": None,
                "transit_duration": 0,
                "transit_cost": 0,
                "satisfaction": 0,
                "cost": 0,
                "rest_duration": 0
            })
        
        prev_location = None
        prev_departure_time = problem.START_TIME
        
        # Process each location in the route
        for i, step in enumerate(route):
            location_idx = step["location"]
            location_name = step["name"]
            location_type = step["type"]
            arrival_time = step["time"]
            transport_mode = step.get("transport_from_prev")
            
            # Calculate transit duration and cost from previous location
            transit_duration = 0
            transit_cost = 0
            rest_duration = 0
            actual_arrival_time = arrival_time
            
            if prev_location is not None and transport_mode:
                transit_duration = arrival_time - prev_departure_time
                
                # Calculate transport cost based on mode and duration
                transport_hour = problem.get_transport_hour(prev_departure_time)

                transport_key = (problem.locations[prev_location]["name"], 
                               problem.locations[location_idx]["name"], 
                               transport_hour)
                transport_data = problem.transport_matrix[transport_key][transport_mode]
                
                transit_cost = transport_data["price"]
                actual_transit_time = round(transport_data["duration"])
            
                # Calculate rest periods for meal timing
                rest_duration = int(transit_duration - actual_transit_time)
                transit_duration = transit_duration - rest_duration
                actual_arrival_time = arrival_time - rest_duration
            
            # Format times as strings
            arrival_time_str = format_time(arrival_time)
            
            # For display purposes, create the actual arrival time string 
            # (before any rest periods)
            actual_arrival_time_str = format_time(actual_arrival_time)
            
            # Calculate departure time (arrival + duration at location)
            location_duration = problem.locations[location_idx].get("duration", 60)
            departure_time = arrival_time + location_duration
            departure_time_str = format_time(departure_time)
            
            # Determine meal type for hawkers
            meal_type = None
            if location_type == "hawker":
                if arrival_time >= problem.LUNCH_START and arrival_time <= problem.LUNCH_END:
                    meal_type = "lunch"
                elif arrival_time >= problem.DINNER_START and arrival_time <= problem.DINNER_END:
                    meal_type = "dinner"
            
            # Calculate cost
            location_cost = 0
            if location_type == "attraction":
                location_cost = problem.locations[location_idx].get("entrance_fee", 0)
            elif location_type == "hawker":
                location_cost = problem.locations[location_idx].get("avg_food_price", 0)
            
            # Calculate satisfaction/rating
            satisfaction = 0
            if location_type == "attraction":
                satisfaction = problem.locations[location_idx].get("satisfaction", 0)
            elif location_type == "hawker":
                satisfaction = problem.locations[location_idx].get("rating", 0)
            
            location_details = problem.locations[location_idx]
            # Create location entry
            location_entry = {
                "name": location_name,
                "type": location_type,
                "arrival_time": arrival_time_str,
                "departure_time": departure_time_str,
                "lat": location_details.get("lat"),
                "lng": location_details.get("lng"),
                "transit_from_prev": transport_mode,
                "transit_duration": round(transit_duration),
                "transit_cost": transit_cost,
                "duration": location_duration,
                "satisfaction": round(satisfaction, 1),
                "cost": round(location_cost, 2),
                "rest_duration": round(rest_duration),
                "actual_arrival_time": actual_arrival_time_str if rest_duration > 0 else None
            }
            
            # Add meal type for hawkers
            if meal_type:
                location_entry["meal_type"] = meal_type
            
            # Add additional details based on location type
            if location_type == "attraction":
                location_entry["description"] = f"Attraction with satisfaction rating {satisfaction:.1f}/{problem.RATING_MAX}"
                location_entry["entrance_fee"] = round(problem.locations[location_idx].get("entrance_fee", 0), 2)
            elif location_type == "hawker":
                location_entry["description"] = f"Food center with rating {satisfaction:.1f}/{problem.RATING_MAX}"
                location_entry["meal_cost"] = round(problem.locations[location_idx].get("avg_food_price", 0), 2)
                
                # Add rest information to description if applicable
                if rest_duration > 0:
                    rest_minutes = int(rest_duration)
                    location_entry["description"] += f". Arrived at {actual_arrival_time_str} and waited {rest_minutes} minutes for opening time."
            
            # Add to day's locations
            day_data["locations"].append(location_entry)
            
            # Update previous location for next iteration
            prev_location = location_idx
            prev_departure_time = departure_time
        
        # Add hotel return at the end if not already included
        if not route or route[-1]["type"] != "hotel":
            # Calculate return to hotel
            hotel_idx = 0
            transport_hour = problem.get_transport_hour(prev_departure_time)
            transit_duration = 30  # Default 30 minutes if transport data not available
            transport_mode = "transit"  # Default transport mode
            transit_cost = 0
            
            # Try to get actual transit time and cost
            if prev_location is not None:
                transport_key = (problem.locations[prev_location]["name"], 
                                problem.locations[hotel_idx]["name"], 
                                transport_hour)
                # Try transit first, then drive
                if "transit" in problem.transport_matrix.get(transport_key, {}):
                    transport_data = problem.transport_matrix[transport_key]["transit"]
                    transit_duration = transport_data["duration"]
                    transit_cost = transport_data["price"]
                    transport_mode = "transit"
                elif "drive" in problem.transport_matrix.get(transport_key, {}):
                    transport_data = problem.transport_matrix[transport_key]["drive"]
                    transit_duration = transport_data["duration"]
                    transit_cost = transport_data["price"]
                    transport_mode = "drive"
            
            # Calculate return time
            return_time = prev_departure_time + transit_duration
            return_time_str = format_time(return_time)
            hotel_location = problem.locations[0]
            
            # Add hotel return
            day_data["locations"].append({
                "name": problem.locations[0]["name"],
                "type": "hotel",
                "position": "end",
                "arrival_time": return_time_str,
                "departure_time": return_time_str,  # Same as arrival for hotel
                "lat": hotel_location.get("lat"),
                "lng": hotel_location.get("lng"),
                "transit_from_prev": transport_mode,
                "transit_duration": round(transit_duration),
                "transit_cost": round(transit_cost, 2),
                "satisfaction": 0,
                "cost": 0,
                "rest_duration": 0
            })
        
        # Add day to itinerary
        itinerary["days"].append(day_data)
    
    # Add attractions visited summary
    itinerary["attractions_visited"] = evaluation.get("visited_attractions", [])
    
    # Calculate budget breakdown
    budget_breakdown = {
        "attractions": 0,
        "meals": 0,
        "transportation": 0
    }
    
    # Process each day to calculate budget breakdown
    total_transport_duration = 0
    total_rest_duration = 0
    for day_data in itinerary["days"]:
        for location in day_data["locations"]:
            if location["type"] == "attraction":
                budget_breakdown["attractions"] += location["cost"]
            elif location["type"] == "hawker":
                budget_breakdown["meals"] += location["cost"]
            
            # Add transportation cost
            if "transit_cost" in location and location["transit_cost"] > 0:
                budget_breakdown["transportation"] += location["transit_cost"]
                total_transport_duration += location.get("transit_duration", 0)
            
            # Add rest duration to total
            total_rest_duration += location.get("rest_duration", 0)
    
    # Add budget breakdown to itinerary
    itinerary["budget_breakdown"] = budget_breakdown
    
    # Add transport summary
    itinerary["transport_summary"] = {
        "total_duration": total_transport_duration,
        "total_cost": budget_breakdown["transportation"]
    }
    
    # Add rest time summary
    itinerary["rest_summary"] = {
        "total_rest_duration": total_rest_duration
    }
    
    # Add constraint violations if the solution is not feasible
    if not evaluation["is_feasible"]:
        # Get inequality and equality violations
        inequality_violations = evaluation.get("inequality_violations", [])
        equality_violations = evaluation.get("equality_violations", [])
        
        # Combine all violations
        all_violations = []
        for violation in inequality_violations + equality_violations:
            all_violations.append({
                "type": violation.get("type", "unknown"),
                "details": violation.get("details", "No details provided")
            })
        
        itinerary["constraint_violations"] = all_violations
    
    # Generate a default filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)
        filename = f"results/itinerary_{timestamp}.json"
    else:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Write JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(itinerary, f, indent=2, ensure_ascii=True)
    
    logger.info(f"JSON itinerary exported to {filename}")
    return filename, itinerary