import numpy as np
import logging

logger = logging.getLogger(__name__)

class ConstraintValidator:
    """
    Comprehensive constraint validation for travel itinerary optimization
    """
    @staticmethod
    def validate_time_windows(problem, solution):
        """
        Validate time window constraints for the entire solution
        
        Args:
            problem: TravelItineraryProblem instance
            solution: Solution vector to validate
        
        Returns:
            list: List of time window constraint violations
        """
        violations = []
        
        # Reshape solution into x_var and u_var
        x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                                problem.num_locations, problem.num_locations)
        u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
        
        # Check daily start time constraint
        for day in range(problem.NUM_DAYS):
            if u_var[day, 0] < problem.START_TIME:
                violations.append({
                    "type": "start_time_violation",
                    "day": day,
                    "current_time": u_var[day, 0],
                    "required_time": problem.START_TIME
                })
        
        return violations
    
    @staticmethod
    def validate_meal_constraints(problem, solution):
        """
        Validate hawker center (meal) constraints
        
        Args:
            problem: TravelItineraryProblem instance
            solution: Solution vector to validate
        
        Returns:
            list: List of meal constraint violations
        """
        violations = []
        
        # Reshape solution into x_var and u_var
        x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                                problem.num_locations, problem.num_locations)
        u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
        
        # Get hawker indices
        hawker_indices = [i for i in range(problem.num_locations) 
                        if problem.locations[i]["type"] == "hawker"]
        
        # Check meal constraints for each day
        for day in range(problem.NUM_DAYS):
            # Track lunch and dinner visits
            lunch_visits = 0
            dinner_visits = 0
            hawker_sum = 0
            
            # Check hawker visits
            for k in range(problem.num_locations):
                if problem.locations[k]["type"] == "hawker":
                    # Count visits as destination
                    hawker_sum += np.sum(x_var[day, :, :, k])
                    
                    # Check time of visit
                    arrival_time = u_var[day, k]
                    if arrival_time >= problem.LUNCH_START and arrival_time <= problem.LUNCH_END:
                        lunch_visits += 1
                    
                    if arrival_time >= problem.DINNER_START and arrival_time <= problem.DINNER_END:
                        dinner_visits += 1
            
            # Check total hawker visits
            if hawker_sum < 2:
                violations.append({
                    "type": "insufficient_hawker_visits",
                    "day": day,
                    "current_visits": hawker_sum,
                    "required_visits": 2
                })
            
            # Check lunch visit
            if lunch_visits < 1:
                violations.append({
                    "type": "no_lunch_hawker_visit",
                    "day": day,
                    "current_visits": lunch_visits,
                    "required_visits": 1
                })
            
            # Check dinner visit
            if dinner_visits < 1:
                violations.append({
                    "type": "no_dinner_hawker_visit",
                    "day": day,
                    "current_visits": dinner_visits,
                    "required_visits": 1
                })
        
        return violations
    
    @staticmethod
    def validate_attraction_constraints(problem, solution):
        """
        Validate attraction visit constraints
        
        Args:
            problem: TravelItineraryProblem instance
            solution: Solution vector to validate
        
        Returns:
            list: List of attraction constraint violations
        """
        violations = []
        
        # Reshape solution into x_var
        x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                                problem.num_locations, problem.num_locations)
        
        # Get attraction indices
        attraction_indices = [i for i in range(problem.num_locations) 
                            if problem.locations[i]["type"] == "attraction"]
        
        # Check attraction visit constraints
        for k in attraction_indices:
            # Total visits as source and destination
            visits_as_source = np.sum(x_var[:, :, k, :])
            visits_as_dest = np.sum(x_var[:, :, :, k])
            
            # Check source visits
            if visits_as_source > 1:
                violations.append({
                    "type": "attraction_max_once_source",
                    "attraction": k,
                    "name": problem.locations[k]["name"],
                    "visits": visits_as_source
                })
            
            # Check destination visits
            if visits_as_dest > 1:
                violations.append({
                    "type": "attraction_max_once_dest",
                    "attraction": k,
                    "name": problem.locations[k]["name"],
                    "visits": visits_as_dest
                })
            
            # Check symmetry (source visits == destination visits)
            if visits_as_source != visits_as_dest:
                violations.append({
                    "type": "attraction_source_dest_mismatch",
                    "attraction": k,
                    "name": problem.locations[k]["name"],
                    "source_visits": visits_as_source,
                    "dest_visits": visits_as_dest
                })
        
        return violations
    
    @staticmethod
    def validate_budget_constraint(problem, solution):
        """
        Validate budget constraints
        
        Args:
            problem: TravelItineraryProblem instance
            solution: Solution vector to validate
        
        Returns:
            list: List of budget constraint violations
        """
        # Evaluate the solution to get total cost
        evaluation = problem.evaluate_solution(solution)
        
        violations = []
        
        # Check total cost against budget
        if evaluation["total_cost"] > problem.budget:
            violations.append({
                "type": "budget_exceeded",
                "total_cost": evaluation["total_cost"],
                "budget_limit": problem.budget
            })
        
        return violations
    
    @staticmethod
    def validate_flow_conservation(problem, solution):
        """
        Validate flow conservation constraints
        
        Args:
            problem: TravelItineraryProblem instance
            solution: Solution vector to validate
        
        Returns:
            list: List of flow conservation violations
        """
        violations = []
        
        # Reshape solution into x_var
        x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                                problem.num_locations, problem.num_locations)
        
        # Check flow conservation for each day and each location
        for day in range(problem.NUM_DAYS):
            for k in range(problem.num_locations):
                # Count incoming and outgoing routes
                incoming = np.sum(x_var[day, :, :, k])
                outgoing = np.sum(x_var[day, :, k, :])
                
                # Check if incoming routes equal outgoing routes
                if incoming != outgoing:
                    violations.append({
                        "type": "flow_conservation_violated",
                        "day": day,
                        "location": k,
                        "location_name": problem.locations[k]["name"],
                        "incoming": float(incoming),
                        "outgoing": float(outgoing)
                    })
        
        return violations
    
    @staticmethod
    def validate_hotel_constraints(problem, solution):
        """
        Validate hotel-related constraints
        
        Args:
            problem: TravelItineraryProblem instance
            solution: Solution vector to validate
        
        Returns:
            list: List of hotel-related constraint violations
        """
        violations = []
        
        # Reshape solution into x_var and u_var
        x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                                problem.num_locations, problem.num_locations)
        u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
        
        # Hotel index is 0
        hotel_idx = 0
        
        # Check that each day starts from the hotel
        for day in range(problem.NUM_DAYS):
            # Check hotel as starting point (outgoing routes)
            hotel_outgoing = np.sum(x_var[day, :, hotel_idx, :])
            if hotel_outgoing != 1:
                violations.append({
                    "type": "hotel_not_starting_point",
                    "day": day,
                    "outgoing_routes": float(hotel_outgoing)
                })
            
            # Check hotel start time constraint - must start at or after START_TIME
            if u_var[day, hotel_idx] < problem.START_TIME:
                violations.append({
                    "type": "hotel_start_time_violation",
                    "day": day,
                    "start_time": float(u_var[day, hotel_idx]),
                    "required_time": problem.START_TIME
                })
        
        # Check that each day returns to the hotel
        for day in range(problem.NUM_DAYS):
            # Find last location (by time)
            last_time = np.max(u_var[day, :])
            last_locations = np.where(u_var[day, :] == last_time)[0]
            
            # If last location is not hotel, check if there's a route back
            if len(last_locations) > 0 and last_locations[0] != hotel_idx:
                last_loc = last_locations[0]
                returns_to_hotel = np.sum(x_var[day, :, last_loc, hotel_idx])
                
                if returns_to_hotel != 1:
                    violations.append({
                        "type": "not_returning_to_hotel",
                        "day": day,
                        "last_location": last_loc,
                        "last_location_name": problem.locations[last_loc]["name"]
                    })
            
            # Check hotel return time - must be before HARD_LIMIT_END_TIME
            if u_var[day, hotel_idx] > problem.HARD_LIMIT_END_TIME:
                violations.append({
                    "type": "hotel_end_time_violation",
                    "day": day,
                    "end_time": float(u_var[day, hotel_idx]),
                    "limit_time": problem.HARD_LIMIT_END_TIME
                })
        
        return violations
    
    @staticmethod
    def validate_transport_exclusivity(problem, solution):
        """
        Validate that only one transport type is used for each route
        
        Args:
            problem: TravelItineraryProblem instance
            solution: Solution vector to validate
        
        Returns:
            list: List of transport exclusivity violations
        """
        violations = []
        
        # Reshape solution into x_var
        x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                                problem.num_locations, problem.num_locations)
        
        # Check each day, source and destination
        for day in range(problem.NUM_DAYS):
            for k in range(problem.num_locations):
                for l in range(problem.num_locations):
                    if k == l:  # Skip self-routes
                        continue
                    
                    # Count how many transport types are used for this route
                    transport_count = sum(x_var[day, j, k, l] for j in range(problem.num_transport_types))
                    
                    # If more than one transport type is used, it's a violation
                    if transport_count > 1:
                        violations.append({
                            "type": "multiple_transport_types",
                            "day": day,
                            "from": k,
                            "to": l,
                            "from_name": problem.locations[k]["name"],
                            "to_name": problem.locations[l]["name"],
                            "transport_count": float(transport_count)
                        })
        
        return violations
    
    @staticmethod
    def validate_visits_per_day(problem, solution):
        """
        Validate the number of visits per day is within acceptable range (2-6)
        
        Args:
            problem: TravelItineraryProblem instance
            solution: Solution vector to validate
        
        Returns:
            list: List of daily visit count violations
        """
        violations = []
        
        # Reshape solution into x_var
        x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                                problem.num_locations, problem.num_locations)
        
        # Minimum and maximum visits per day
        min_visits = 2  # Minimum: lunch and dinner
        max_visits = 6  # Maximum reasonable visits per day
        
        # Check each day's visit count
        for day in range(problem.NUM_DAYS):
            # Count trips to non-hotel destinations
            visits = 0
            for j in range(problem.num_transport_types):
                for k in range(problem.num_locations):
                    for l in range(1, problem.num_locations):  # Start from 1 to skip hotel as destination
                        if k != l and x_var[day, j, k, l] > 0:
                            visits += 1
            
            # Check minimum visits
            if visits < min_visits:
                violations.append({
                    "type": "insufficient_daily_visits",
                    "day": day,
                    "visits": visits,
                    "minimum": min_visits
                })
            
            # Check maximum visits
            if visits > max_visits:
                violations.append({
                    "type": "excessive_daily_visits",
                    "day": day,
                    "visits": visits,
                    "maximum": max_visits
                })
        
        return violations
    
    @staticmethod
    def validate_time_brackets(problem, solution):
        """
        Validate that times for activities fall into appropriate time brackets
        and sequential visits have appropriate timing
        
        Args:
            problem: TravelItineraryProblem instance
            solution: Solution vector to validate
        
        Returns:
            list: List of time bracket violations
        """
        violations = []
        
        # Reshape solution into x_var and u_var
        x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                                problem.num_locations, problem.num_locations)
        u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
        
        # For each day, check time sequential logic
        for day in range(problem.NUM_DAYS):
            for j in range(problem.num_transport_types):
                for k in range(problem.num_locations):
                    for l in range(problem.num_locations):
                        if k == l or x_var[day, j, k, l] == 0:
                            continue
                        
                        # This route is used
                        try:
                            # Get transport hour
                            transport_hour = problem.get_transport_hour(u_var[day, k])
                            
                            # Get transport data
                            transport_key = (problem.locations[k]["name"], 
                                           problem.locations[l]["name"], 
                                           transport_hour)
                            
                            transport_data = problem.transport_matrix[transport_key][problem.transport_types[j]]
                            
                            # Calculate expected arrival time
                            departure_time = u_var[day, k]
                            expected_arrival = departure_time + transport_data["duration"]
                            
                            # Calculate minimum time needed for the activity
                            activity_duration = problem.locations[l]["duration"]
                            
                            # Calculate expected finish time
                            expected_finish = expected_arrival + activity_duration
                            
                            # Get actual finish time in solution
                            actual_finish = u_var[day, l]
                            
                            # Check if actual finish time is too early (impossible)
                            if actual_finish < expected_finish:
                                violations.append({
                                    "type": "sequential_time_violation",
                                    "day": day,
                                    "from": k,
                                    "to": l,
                                    "from_name": problem.locations[k]["name"],
                                    "to_name": problem.locations[l]["name"],
                                    "actual_finish": float(actual_finish),
                                    "expected_finish": float(expected_finish)
                                })
                        
                        except KeyError:
                            # Missing transport data, add a violation
                            violations.append({
                                "type": "missing_transport_data",
                                "day": day,
                                "from": k,
                                "to": l,
                                "from_name": problem.locations[k]["name"],
                                "to_name": problem.locations[l]["name"]
                            })
        
        return violations
        
    @classmethod
    def validate_solution(cls, problem, solution):
        """
        Comprehensive solution validation
        
        Args:
            problem: TravelItineraryProblem instance
            solution: Solution vector to validate
        
        Returns:
            dict: Comprehensive validation results
        """
        validation_results = {
            "is_valid": True,
            "time_window_violations": cls.validate_time_windows(problem, solution),
            "meal_violations": cls.validate_meal_constraints(problem, solution),
            "attraction_violations": cls.validate_attraction_constraints(problem, solution),
            "budget_violations": cls.validate_budget_constraint(problem, solution),
            "flow_conservation_violations": cls.validate_flow_conservation(problem, solution),
            "hotel_violations": cls.validate_hotel_constraints(problem, solution),
            "transport_exclusivity_violations": cls.validate_transport_exclusivity(problem, solution),
            "visits_per_day_violations": cls.validate_visits_per_day(problem, solution),
            "time_bracket_violations": cls.validate_time_brackets(problem, solution)
        }
        
        # Determine overall validity
        validation_results["is_valid"] = (
            len(validation_results["time_window_violations"]) == 0 and
            len(validation_results["meal_violations"]) == 0 and
            len(validation_results["attraction_violations"]) == 0 and
            len(validation_results["budget_violations"]) == 0 and
            len(validation_results["flow_conservation_violations"]) == 0 and
            len(validation_results["hotel_violations"]) == 0 and
            len(validation_results["transport_exclusivity_violations"]) == 0 and
            len(validation_results["visits_per_day_violations"]) == 0 and
            len(validation_results["time_bracket_violations"]) == 0
        )
        
        return validation_results