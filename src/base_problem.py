import numpy as np
from pymoo.core.problem import ElementwiseProblem
import os
import logging

# from utils.transport_utility import get_transport_hour

# Set up logging
os.makedirs("log", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/generated_problem.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("generated_problem")

class TravelItineraryProblem(ElementwiseProblem):

    def validate_inputs(self, budget, locations, transport_matrix, num_days):
        """Validate input data and print warnings/errors"""
        logging.info("Validating optimization inputs...")
        
        # Check if we have at least one hotel
        hotels = [loc for loc in locations if loc["type"] == "hotel"]
        if not hotels:
            logging.error("No hotels found in locations data!")
        else:
            logging.info(f"Found {len(hotels)} hotels in the data")
        
        # Check if we have hawkers for meals
        hawkers = [loc for loc in locations if loc["type"] == "hawker"]
        if not hawkers:
            logging.error("No hawker centers found - meal constraints cannot be satisfied!")
        else:
            logging.info(f"Found {len(hawkers)} hawker centers in the data")
        
        # Check for attractions
        attractions = [loc for loc in locations if loc["type"] == "attraction"]
        logging.info(f"Found {len(attractions)} attractions in the data")
        
        # Validate location data completeness
        for i, loc in enumerate(locations):
            missing = []
            if loc["type"] == "attraction":
                if "entrance_fee" not in loc or loc["entrance_fee"] is None:
                    missing.append("entrance_fee")
                if "satisfaction" not in loc or loc["satisfaction"] is None:
                    missing.append("satisfaction")
                if "duration" not in loc or loc["duration"] is None:
                    missing.append("duration")
            elif loc["type"] == "hawker":
                if "rating" not in loc or loc["rating"] is None:
                    missing.append("rating")
                if "duration" not in loc or loc["duration"] is None:
                    missing.append("duration")
            
            if missing:
                logging.warning(f"Location '{loc['name']}' is missing required fields: {', '.join(missing)}")
        
        # Check transport matrix completeness
        sample_routes = 0
        missing_routes = 0
        for i, src in enumerate(locations):
            for j, dest in enumerate(locations):
                if i != j:  # Skip self-routes
                    for hour in [8, 12, 16, 20]:  # Time brackets
                        key = (src["name"], dest["name"], hour)
                        if key not in transport_matrix:
                            missing_routes += 1
                            if missing_routes <= 5:  # Only log the first few missing routes
                                logging.error(f"Missing transport data: {key}")
                        else:
                            sample_routes += 1
        
        if missing_routes > 0:
            logging.error(f"Missing {missing_routes} routes in transport matrix!")
        else:
            logging.info(f"Transport matrix contains all required routes ({sample_routes} total)")
        
        # Check budget feasibility
        hotel_cost = num_days * 50  # Using your fixed HOTEL_COST of 50
        min_food_cost = num_days * 2 * 10  # Minimum 2 meals per day at 10 each
        
        min_cost = hotel_cost + min_food_cost
        if budget < min_cost:
            logging.error(f"Budget ({budget}) is too low! Minimum needed is {min_cost} for hotel and food alone")
        else:
            logging.info(f"Budget check passed: {budget} >= minimum {min_cost} for hotel and food")
    
    def __init__(self, budget, locations, transport_matrix, num_days=3):
        
        # Add validation checks before setup
        self.validate_inputs(budget, locations, transport_matrix, num_days)
        
        # constants
        self.NUM_DAYS = num_days
        self.HOTEL_COST = 50
        self.START_TIME = 9 * 60 # everyday starts from 9 AM
        self.HARD_LIMIT_END_TIME = 22 * 60 # everyday MUST return back to hotel by 10 PM

        # Define lunch and dinner time windows (in minutes since start of day at 9 AM)
        self.LUNCH_START = 11 * 60  # 11 AM (9 AM + 2 hours)
        self.LUNCH_END = 15 * 60    # 3 PM (9 AM + 6 hours)
        self.DINNER_START = 17 * 60  # 5 PM (9 AM + 8 hours)
        self.DINNER_END = 21 * 60   # 9 PM (9 AM + 12 hours)
        
        # hard limit money spent
        self.budget = budget
        # list of destinations
        self.locations = locations
        self.num_locations = len(locations)
        self.num_attractions = len([loc for loc in self.locations if loc["type"] == "attraction"])
        self.num_hawkers = len([loc for loc in self.locations if loc["type"] == "hawker"])
        self.num_hotels = 1 # always 1

        # transportation option prices and durations. The size MUST BE num_dest * num_dest, for 24 hours
        self.transport_types = ["transit", "drive"]
        self.num_transport_types = len(self.transport_types)
        self.transport_matrix = transport_matrix

        # x_ijkl = BINARY VAR, if the route goes in day i, use transport type j, from location k, to location l
        self.x_shape = self.NUM_DAYS * self.num_transport_types * self.num_locations * self.num_locations
        # u_ik  = CONTINUOUS VAR, tracking FINISH time of the person in day i, location k
        self.u_shape = self.NUM_DAYS * self.num_locations

        num_vars = self.x_shape + self.u_shape

        lower_bound = np.concatenate([np.zeros(self.x_shape), np.full(self.u_shape, 0)])
        upper_bound = np.concatenate([np.ones(self.x_shape), np.full(self.u_shape, self.HARD_LIMIT_END_TIME)])
        
        # Count constraints
        def calculate_constraints():
            ### For counting actual inequality constraints
            g_count = 0
            
            # For each attraction, must be visited at most once as source and at most once as destination
            g_count += 2 * self.num_attractions

            # For each hawker everyday, must be visited at most once as source and at most once as destination
            g_count += 2 * self.NUM_DAYS * self.num_hawkers

            # Every day, go from hotel at least START_TIME
            g_count += self.NUM_DAYS
            
            # For time constraints when a route is chosen
            g_count += self.NUM_DAYS * self.num_transport_types * (self.num_locations - 1)
            g_count += self.NUM_DAYS * self.num_transport_types * (self.num_locations - 1) * (self.num_locations - 2)
            
            # For hawker visits (at least twice per day)
            g_count += self.NUM_DAYS
            
            # For lunch and dinner time constraints
            g_count += self.NUM_DAYS * 2
            
            # For transport type constraints (can't use both transit and drive for the same route)
            g_count += self.NUM_DAYS * self.num_locations * self.num_locations
            
            # Budget constraint
            g_count += 1
            
            # Min/max total visits constraints
            g_count += 2
            
            ### For equality constraints
            h_count = 0
            
            # Flow conservation (in = out)
            h_count += self.NUM_DAYS * self.num_locations
            
            # Hotel must be starting point each day
            h_count += self.NUM_DAYS
            
            # Return to hotel constraint
            h_count += self.NUM_DAYS
            
            # If attraction is visited as source, it must be visited as destination
            h_count += self.NUM_DAYS * self.num_locations
            
            return g_count, h_count

        # Calculate actual constraints
        num_inequality_constraints, num_equality_constraints = calculate_constraints()
        
        super().__init__(
            n_var=num_vars,
            n_obj=3, # INEQUALITY_CONSTRAINT_LINE
            n_ieq_constr=num_inequality_constraints,
            n_eq_constr=num_equality_constraints,
            xl=lower_bound,
            xu=upper_bound,
        )
        
    def test_feasibility(self):
        """Test if the problem has any feasible solutions"""
        logging.info("Testing problem feasibility...")
        
        # Check if we have enough hawkers for lunch and dinner every day
        if self.num_hawkers == 0:
            logging.error("Infeasible: No hawker centers available for meals")
            return False
        
        # Check if we can meet the time constraints
        # This is a simplified check - minimum time would be:
        # - Start at hotel
        # - Travel to lunch hawker
        # - Eat lunch (60 min)
        # - Travel to attraction 
        # - Visit attraction
        # - Travel to dinner hawker
        # - Eat dinner (60 min)
        # - Travel back to hotel
        
        # Check if there's enough time in the day for this minimum itinerary
        available_time = self.HARD_LIMIT_END_TIME - self.START_TIME  # Minutes available
        logging.info(f"Available time per day: {available_time} minutes")
        
        # Simple feasibility test on time windows 
        lunch_window = self.LUNCH_END - self.LUNCH_START
        dinner_window = self.DINNER_END - self.DINNER_START
        logging.info(f"Lunch window: {lunch_window} minutes, Dinner window: {dinner_window} minutes")
        
        # Check if we can satisfy hawker constraints 
        if lunch_window < 60:
            logging.error(f"Infeasible: Lunch window ({lunch_window} min) too short for a 60 min meal")
            return False
        
        if dinner_window < 60:
            logging.error(f"Infeasible: Dinner window ({dinner_window} min) too short for a 60 min meal")
            return False
        
        return True

    def get_transport_hour(self, transport_time):
        # because the transport_matrix is only bracketed to 4 groups, we find the earliest it happens
        brackets = [8, 12, 16, 20]
        transport_hour = transport_time // 60

        for bracket in reversed(brackets):
            if transport_hour >= bracket:
                return bracket

        return brackets[-1] # from 8 PM to 8 AM next day
    
    def _evaluate(self, x, out, *args, **kwargs):
        # they're ensured to be integers
        x_var = x[:self.x_shape].reshape(self.NUM_DAYS, self.num_transport_types, self.num_locations, self.num_locations)
        u_var = x[self.x_shape:].reshape(self.NUM_DAYS, self.num_locations)

        # initialize constraints
        # equality constraints
        out["H"] = []
        # inequality constraints
        out["G"] = []

        total_cost = self.NUM_DAYS * self.HOTEL_COST  # Cost starts with hotel cost for number of nights
        total_travel_time = 0
        total_satisfaction = 0

        # for every attraction, must be a source & destination at most once
        # NOTE that this doesn't apply to hawkers
        for k in range(self.num_locations):
            if self.locations[k]["type"] == "attraction":
                out["G"].append(np.sum(x_var[:, :, k, :]) - 1)
                out["G"].append(np.sum(x_var[:, :, :, k]) - 1)
        
        # for hawkers, every day must be a source & destination at most once
        for i in range(self.NUM_DAYS):
            for k in range(self.num_locations):
                if self.locations[k]["type"] == "hawker":
                    out["G"].append(np.sum(x_var[i, :, k, :]) - 1)
                    out["G"].append(np.sum(x_var[i, :, :, k]) - 1)
                
                # If attraction is a source, it must also be a destination
                out["H"].append(np.sum(x_var[:, :, k, :]) - np.sum(x_var[:, :, :, k]))

        for i in range(self.NUM_DAYS):
            # u_var[i, 0] must be the smallest of u_var[i]
            non_zero_elements = u_var[i, u_var[i, :] > 0]
            if len(non_zero_elements) > 0:
                out["H"].append(np.min(non_zero_elements) - u_var[i, 0])
            else: # if everything is zero, this is violated already
                out["H"].append(-1)

            # day start must be 8 AM or more
            out["G"].append((self.START_TIME - u_var[i, 0]))

            for j in range(self.num_transport_types):
                for k in range(self.num_locations):

                    for l in range(1, self.num_locations): # NOTE here that hotel (index 0) isn't included in destination. It will be computed later
                        if k == l: continue
                        # every day, if the route is chosen, the time of finishing in this place is this

                        # if chosen, then time_should_finish_l must be time_finish_l, otherwise, don't matter (just put 0)
                        # you should finish attraction l at:
                        #   - time to finish k + 
                        #   - time to transport from k to l, using transport method j +
                        #   - time to play at l
                        time_finish_l = u_var[i, l]
                        transport_hour = self.get_transport_hour(u_var[i, k])
                        transport_value = self.transport_matrix[(self.locations[k]["name"], self.locations[l]["name"], transport_hour)][self.transport_types[j]]
                        time_should_finish_l = u_var[i, k] + transport_value["duration"] + self.locations[l]["duration"]
                        # if x_var[i, j, k, l] is not chosen, then this constraint don't matter
                        out["G"].append(x_var[i, j, k, l] * (time_should_finish_l - time_finish_l))
                        
                        # append to total travel time spent
                        if x_var[i, j, k, l] == 1:
                            # calculate the travel
                            total_travel_time += transport_value["duration"]
                            total_cost += transport_value["price"]

                            # calculate cost and satisfaction, based on what they come to
                            # Note: not counting anything for hotel.
                            if self.locations[l]["type"] == "attraction":
                                total_cost += self.locations[l]["entrance_fee"]
                                total_satisfaction += self.locations[l]["satisfaction"]
                            elif self.locations[l]["type"] == "hawker":
                                total_cost += 10 # ASSUME eating in a hawker is ALWAYS $10
                                total_satisfaction += self.locations[l]["rating"]

            # from last place, return to hotel.
            last_place = np.argmax(u_var[i, :])
            # THIS SHOULD NEVER HAPPEN: if last_place is already the hotel, then get the second last
            if last_place == 0: # doesn't make sense, because the first constraint is u[i, 0] must be the smallest (it's starting point)
                last_place = np.argsort(u_var[i, :])[-2]
                latest_hour = self.get_transport_hour(u_var[i, last_place])
                out["H"].append(np.sum(x_var[i, :, last_place, 0]) - 1)
            else:
                latest_hour = self.get_transport_hour(u_var[i, last_place])
                # MUST go back to hotel at the end of the day
                out["H"].append(np.sum(x_var[i, :, last_place, 0]) - 1)
                # go back to hotel
                # choose whether to use which transport type
                for j in range(self.num_transport_types):
                    if x_var[i, j, last_place, 0]:
                        # pull from google maps to get the transport details if the decision is here.
                        gohome_value = self.transport_matrix[(self.locations[last_place]["name"], self.locations[0]["name"], latest_hour)][self.transport_types[j]]
                        total_travel_time += gohome_value["duration"]
                        total_cost += gohome_value["price"]
                        break

        for i in range(self.NUM_DAYS):
            
            lunch_hawker_visit = 0
            dinner_hawker_visit = 0
            
            hawker_sum = 0
            for k in range(self.num_locations):
                # every day, for every location, if it's selected as source, it must be selected as destination
                out["H"].append(np.sum(x_var[i, :, :, k]) - np.sum(x_var[i, :, k, :]))

                if self.locations[k]["type"] == "hawker":
                    hawker_sum += np.sum(x_var[i, :, k, :])
                    
                    # For each route ending at this hawker, check if it's during lunch time or dinner time
                    for src in range(self.num_locations):
                        if src == k:
                            continue
                        for j_transport in range(self.num_transport_types):
                            if u_var[i, k] >= self.LUNCH_START and u_var[i, k] <= self.LUNCH_END:
                                lunch_hawker_visit += x_var[i, j_transport, src, k]
                            
                            if u_var[i, k] >= self.DINNER_START and u_var[i, k] <= self.DINNER_END:
                                dinner_hawker_visit += x_var[i, j_transport, src, k]

            # every day, must go to hawkers at least twice (lunch & dinner. Can go more times if they want to)
            out["G"].append(2 - hawker_sum)
            
            # every day, must visit a hawker during lunch time (at least one hawker visit with arrival/stay during lunch hours)
            if self.num_hawkers > 0:  # Only add constraint if there are hawkers available
                out["G"].append(1 - lunch_hawker_visit)
                
                # every day, must visit a hawker during dinner time (at least one hawker visit with arrival/stay during dinner hours)
                out["G"].append(1 - dinner_hawker_visit)

        for i in range(self.NUM_DAYS):
            for k in range(self.num_locations):
                for l in range(self.num_locations):
                    # for every day, if public transportation is chosen, taxi can't be chosen, and vice versa
                    # (sum j in transport_types x_ijkl <= 1) for all days, for all sources, for all destinations
                    out["G"].append(np.sum(x_var[i, :, k, l]) - 1)

        # finally, make sure everything is within budget
        out["G"].append(total_cost - self.budget)
        
        # Calculate reasonable minimum and maximum visits
        # Minimum: At least 2 hawker visits per day (lunch & dinner)
        min_visits = self.NUM_DAYS * 2
        # Maximum: Reasonable upper bound for visits per day (e.g., start at hotel + 2 meals + 2-3 attractions)
        max_visits_per_day = 6
        max_visits = self.NUM_DAYS * max_visits_per_day

        # Constraint: Total visits should be at least the minimum
        out["G"].append(min_visits - np.sum(x_var))
        # Constraint: Total visits should be at most the maximum
        out["G"].append(np.sum(x_var) - max_visits)
        # INDENTATION_COUNT_LINE
        # <ADD ADDITIONAL CONSTRAINTS HERE>

        # objectives
        out["F"] = [total_cost, total_travel_time, -total_satisfaction]