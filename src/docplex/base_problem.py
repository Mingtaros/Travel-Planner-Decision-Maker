import json
import random
import numpy as np
from docplex.mp.model import Model


def get_transport_matrix():
    BASE_PATH = "data/routeData/"
    daytimes = [("morning", 8), ("midday", 12), ("evening", 16), ("night", 20)]
    transport_matrix = {}

    for time_in_day, hour in daytimes:
        filepath = BASE_PATH + f"route_matrix_{time_in_day}.json"

        with open(filepath, 'r') as f:
            route_matrix = json.load(f)

        for route in route_matrix["routes"]:
            this_route = route_matrix["routes"][route]
            origin = this_route["origin_name"]
            destination = this_route["destination_name"]
            transport_matrix[(origin, destination, hour)] = {
                "transit": {
                    "duration": this_route["transit"]["duration_minutes"],
                    "price": this_route["transit"]["fare_sgd"],
                },
                "drive": {
                    "duration": this_route["drive"]["duration_minutes"],
                    "price": this_route["drive"]["fare_sgd"],
                }
            }
    
    return transport_matrix


def get_all_locations():
    BASE_PATH = "data/routeData/"

    with open(BASE_PATH + "route_matrix_morning.json", 'r') as f:
        route_matrix = json.load(f)
    
    locations = [route_matrix["locations"][location_id] for location_id in route_matrix["locations"]]
    return locations

class TravelItineraryProblem(object):
    def __init__(
            self,
            budget,
            locations,
            transport_matrix,
            num_days=3,
            priority_weights=[0.3, 0.3, 0.4],
        ):

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
        self.time_brackets = [8, 12, 16, 20]

        # add weighting priority for objective function
        self.priority_weights = priority_weights
        # the goal here is to make the objective work in the same axis, price $
        self.travel_time_penalty_per_minute = 0.1 # for every minute spent in transit, the "cost" is this in $
        self.satisfaction_offset = 10 # for the satisfaction, every point in satisfaction, offset the cost by this in $

        # define models and variables
        self.define_model()
        self.define_variables()

        # initialize solution as None before solving
        self.solution = None


    def define_model(self):
        self.mdl = Model("MITB-AI")


    def define_variables(self):
        self.x_var = {
            (day, transport_type, source["name"], dest["name"]): self.mdl.binary_var(f"Day_{day}_GoFrom_{source['name']}_To_{dest['name']}_using_{transport_type}")
            for day in range(self.NUM_DAYS)
            for transport_type in self.transport_types
            for source in self.locations
            for dest in self.locations
            if source != dest
        }

        self.u_var = {
            (day, location["name"]): self.mdl.integer_var(0, self.HARD_LIMIT_END_TIME, name=f"Day_{day}_FinishAttraction_{location['name']}_time")
            for day in range(self.NUM_DAYS)
            for location in self.locations
        }

        self.bracket_var = {
            (day, location["name"], time_bracket): self.mdl.binary_var(f"Day_{day}_FinishAttraction_{location['name']}_time_bracket_{time_bracket}")
            for day in range(self.NUM_DAYS)
            for location in self.locations
            for time_bracket in self.time_brackets
        }

        self.last_visit_var = {
            (day, location["name"]): self.mdl.binary_var(name=f"Day_{day}_LastVisitIs_{location['name']}")
            for day in range(self.NUM_DAYS)
            for location in self.locations
        }

        self.lunch_hawker_visit_var = {
            (day, location["name"]): self.mdl.binary_var(name=f"Day_{day}_VisitHawker_{location['name']}_inLunchTime")
            for day in range(self.NUM_DAYS)
            for location in self.locations
            if location["type"] == "hawker"
        }

        self.dinner_hawker_visit_var = {
            (day, location["name"]): self.mdl.binary_var(name=f"Day_{day}_VisitHawker_{location['name']}_inDinnerTime")
            for day in range(self.NUM_DAYS)
            for location in self.locations
            if location["type"] == "hawker"
        }

        self.cost_var = self.mdl.continuous_var(lb=0, name="total_cost")
        self.travel_time_var = self.mdl.continuous_var(lb=0, name="total_travel_time")
        self.satisfaction_var = self.mdl.continuous_var(lb=0, name="total_satisfaction")
    
    def define_constraints(self):
        # every day, for every location, the time bracket the guy goes in, must be only 1, cannot go to multiple
        for day in range(self.NUM_DAYS):
            for source in self.locations:
                self.mdl.add_constraint(
                    sum(self.bracket_var[(day, source["name"], time_bracket)] for time_bracket in self.time_brackets) == 1
                )

        # link bracket_var and u_var
        LARGE_M = 10e6  # Large enough big-M constant
        for day in range(self.NUM_DAYS):
            for source in self.locations:
                for time_bracket in self.time_brackets:
                    self.mdl.add_constraint( # ensure that if this time bracket is chosen, then u_var must be bigger than the next bracket
                        time_bracket * self.bracket_var[(day, source["name"], time_bracket)] <= self.u_var[(day, source["name"])].divide(60)
                    )
                    # if this time bracket is chosen, then it must be between this bracket and the next bracket
                    self.mdl.add_constraint(
                        self.u_var[(day, source["name"])].divide(60) <= (time_bracket + 4) * self.bracket_var[(day, source["name"], time_bracket)] +
                        LARGE_M * (1 - self.bracket_var[(day, source["name"], time_bracket)])
                    )

        # make sure there's only 1 last visit every day
        for day in range(self.NUM_DAYS):
            self.mdl.add_constraint(
                sum([self.last_visit_var[(day, loc["name"])] for loc in self.locations[1:]]) == 1
            )
            # link last visit with u_var
            for loc in self.locations:
                for other_loc in self.locations:
                    self.mdl.add_constraint(
                        self.u_var[(day, loc["name"])] <= self.u_var[(day, other_loc["name"])] +
                        LARGE_M * (1 - self.last_visit_var[(day, other_loc["name"])])
                    )
            # for everyday, last visit cannot be hotel
            self.mdl.add_constraint(self.last_visit_var[(day, self.locations[0]["name"])] == 0)

        # for every attraction, must be a source & destination at most once
        for loc in self.locations:
            if loc["type"] == "attraction":
                self.mdl.add_constraint(
                    sum([
                        self.x_var[(day, transport_type, loc["name"], dest["name"])]
                        for day in range(self.NUM_DAYS)
                        for transport_type in self.transport_types
                        for dest in self.locations
                        if loc != dest
                    ]) <= 1
                )
                self.mdl.add_constraint(
                    sum([
                        self.x_var[(day, transport_type, source["name"], loc["name"])]
                        for day in range(self.NUM_DAYS)
                        for transport_type in self.transport_types
                        for source in self.locations
                        if loc != source
                    ]) <= 1
                )

        # for hawkers, every day must be a source & destination at most once
        for day in range(self.NUM_DAYS):
            for loc in self.locations:
                if loc["type"] == "hawker":
                    self.mdl.add_constraint(
                        sum([
                            self.x_var[(day, transport_type, source["name"], loc["name"])]
                            for transport_type in self.transport_types
                            for source in self.locations
                            if source != loc
                        ]) <= 1
                    )
                    self.mdl.add_constraint(
                        sum([
                            self.x_var[(day, transport_type, loc["name"], dest["name"])]
                            for transport_type in self.transport_types
                            for dest in self.locations
                            if dest != loc
                        ]) <= 1
                    )

                # if attraction is a source, it must also be a destination
                self.mdl.add_constraint(
                    sum([
                        self.x_var[(day, transport_type, source["name"], loc["name"])]
                        for transport_type in self.transport_types
                        for source in self.locations
                        if source != loc
                    ]) == sum([
                        self.x_var[(day, transport_type, loc["name"], dest["name"])]
                        for transport_type in self.transport_types
                        for dest in self.locations
                        if dest != loc
                    ])
                )

        # link lunch & dinner hawker visit to u_var
        for day in range(self.NUM_DAYS):
            for loc in self.locations:
                if loc["type"] != "hawker": continue

                self.mdl.add_constraint(
                    self.LUNCH_START * self.lunch_hawker_visit_var[(day, loc["name"])] <= self.u_var[(day, loc["name"])]
                )
                self.mdl.add_constraint(
                    self.u_var[(day, loc["name"])] <= self.LUNCH_END + LARGE_M * (1 - self.lunch_hawker_visit_var[(day, loc["name"])])
                )

                self.mdl.add_constraint(
                    self.DINNER_START * self.dinner_hawker_visit_var[(day, loc["name"])] <= self.u_var[(day, loc["name"])]
                )
                self.mdl.add_constraint(
                    self.u_var[(day, loc["name"])] <= self.DINNER_END + LARGE_M * (1 - self.dinner_hawker_visit_var[(day, loc["name"])])
                )

        for day in range(self.NUM_DAYS):
            # everyday, hotel must be starting point
            for loc in self.locations:
                if loc["name"] != self.locations[0]["name"]:
                    self.mdl.add_constraint(
                        self.u_var[(day, self.locations[0]["name"])] <= self.u_var[(day, loc["name"])]
                    )

            # hotel starting must be at START_TIME earliest
            self.mdl.add_constraint(
                self.u_var[(day, self.locations[0]["name"])] >= self.START_TIME
            )

            for transport_type in self.transport_types:
                for source in self.locations:
                    for dest in self.locations[1:]: # NOTE here that hotel (index 0) isn't included in destination. It will be computed later
                        if source == dest: continue
                        # every day, if the route is chosen, the time of finishing in this place is this
                        for time_bracket in self.time_brackets:
                            transport_value = self.transport_matrix[(source["name"], dest["name"], time_bracket)][transport_type]
                            self.mdl.add_constraint(
                                self.u_var[(day, dest["name"])] >= self.u_var[(day, source["name"])] +
                                transport_value["duration"] + dest["duration"] -
                                LARGE_M * (1 - self.x_var[(day, transport_type, source["name"], dest["name"])]) -
                                LARGE_M * (1 - self.bracket_var[(day, source["name"], time_bracket)])
                            )

            # make sure to go back to hotel
            for loc in self.locations[1:]:
                self.mdl.add_constraint(
                    sum([
                        self.x_var[(day, transport_type, loc["name"], self.locations[0]["name"])]
                        for transport_type in self.transport_types
                    ]) == self.last_visit_var[(day, loc["name"])]
                )

            for loc in self.locations:
                self.mdl.add_constraint(
                    sum([
                        self.x_var[(day, transport_type, source["name"], loc["name"])]
                        for transport_type in self.transport_types
                        for source in self.locations
                        if source != loc
                    ]) == sum([
                        self.x_var[(day, transport_type, loc["name"], dest["name"])]
                        for transport_type in self.transport_types
                        for dest in self.locations
                        if dest != loc
                    ])
                )

            # everyday, must go to hawkers at least twice (lunch & dinner)
            self.mdl.add_constraint(sum([
                self.x_var[(day, transport_type, source["name"], dest["name"])]
                for transport_type in self.transport_types
                for source in self.locations
                for dest in self.locations
                if source != dest and dest["type"] == "hawker"
            ]) >= 2)

            # every day, have lunch exactly once
            self.mdl.add_constraint(
                sum(self.lunch_hawker_visit_var[(day, hawker["name"])] for hawker in self.locations if hawker["type"] == "hawker") == 1
            )

            # every day, have dinner exactly once
            self.mdl.add_constraint(
                sum(self.dinner_hawker_visit_var[(day, hawker["name"])] for hawker in self.locations if hawker["type"] == "hawker") == 1
            )

            for source in self.locations:
                for dest in self.locations:
                    if source == dest: continue
                    # for every day, if public transportation is chosen, taxi can't be chosen, and vice versa
                    self.mdl.add_constraint(
                        sum([
                            self.x_var[(day, transport_type, source["name"], dest["name"])]
                            for transport_type in self.transport_types
                        ]) <= 1
                    )

        self.mdl.add_constraint(self.cost_var ==
            self.NUM_DAYS * self.HOTEL_COST + 
            sum(
                self.x_var[(day, transport_type, source["name"], dest["name"])] * (
                    self.transport_matrix[(source["name"], dest["name"], time_bracket)][transport_type]["price"] 
                    + (dest["entrance_fee"] if dest["type"] == "attraction" else 0)
                    + (dest["avg_food_price"] if dest["type"] == "hawker" else 0)
                )
                for day in range(self.NUM_DAYS)
                for transport_type in self.transport_types
                for source in self.locations
                for dest in self.locations
                for time_bracket in self.time_brackets
                if source["name"] != dest["name"]
            )
        )
        self.mdl.add_constraint(self.travel_time_var == 
            sum(
                self.x_var[(day, transport_type, source["name"], dest["name"])] * 
                self.transport_matrix[(source["name"], dest["name"], time_bracket)][transport_type]["duration"]
                for day in range(self.NUM_DAYS)
                for transport_type in self.transport_types
                for source in self.locations
                for dest in self.locations
                for time_bracket in self.time_brackets
                if source["name"] != dest["name"]
            )
        )
        self.mdl.add_constraint(self.satisfaction_var == 
            sum(
                self.x_var[(day, transport_type, source["name"], dest["name"])] * (
                    dest["satisfaction"] if dest["type"] == "attraction" else 0
                )
                for day in range(self.NUM_DAYS)
                for transport_type in self.transport_types
                for source in self.locations
                for dest in self.locations
                if source["name"] != dest["name"]
            ) + sum(
                self.x_var[(day, transport_type, source["name"], dest["name"])] * (
                    dest["rating"] if dest["type"] == "hawker" else 0
                )
                for day in range(self.NUM_DAYS)
                for transport_type in self.transport_types
                for source in self.locations
                for dest in self.locations
                if source["name"] != dest["name"]
            )
        )
        # finally, make sure everything is within budget
        self.mdl.add_constraint(self.cost_var <= self.budget)

        # you must visit 2-6 places per day
        min_visits = self.NUM_DAYS * 2
        max_visits = self.NUM_DAYS * 6
        self.mdl.add_constraint(
            sum([
                self.x_var[(day, transport_type, source["name"], dest["name"])]
                for day in range(self.NUM_DAYS)
                for transport_type in self.transport_types
                for source in self.locations
                for dest in self.locations
                if source != dest
            ]) <= max_visits
        )
        self.mdl.add_constraint(
            sum([
                self.x_var[(day, transport_type, source["name"], dest["name"])]
                for day in range(self.NUM_DAYS)
                for transport_type in self.transport_types
                for source in self.locations
                for dest in self.locations
                if source != dest
            ]) >= min_visits
        )

        # <ADD CONSTRAINTS HERE>


    def solve(self):
        # return, get solution
        self.define_constraints()
        # minimizing
        self.mdl.minimize(
            self.priority_weights[0] * self.cost_var +
            self.priority_weights[1] * self.travel_time_var * self.travel_time_penalty_per_minute -
            self.priority_weights[2] * self.satisfaction_var * self.satisfaction_offset
        )

        self.solution = self.mdl.solve(log_output=True)
        # store the solution inside self.solution
    
    def print_solution(self):
        if self.solution is None:
            print("ERROR: No Feasible Solution or solve not run yet.")
            return
        
        for day in range(self.NUM_DAYS):
            print(f"Day {day}:")
            
            # Find starting location (hotel) and its start time
            hotel = self.locations[0]
            start_location = hotel["name"]
            start_time = round(self.u_var[(day, start_location)].solution_value)
            print(f"{int(start_time // 60):02d}:{int(start_time % 60):02d} - Start from {start_location}")

            current_location = start_location
            while True:
                next_location = None
                chosen_transport = None
                travel_time = None
                price = None
                entrance_fee = None
                duration = None
                food_price = None
                satisfaction_score = None
                rating = None

                # Find the next location and transport used
                for transport_type in self.transport_types:
                    for dest in self.locations:
                        if dest["name"] == current_location:
                            continue
                        
                        for time_bracket in self.time_brackets:
                            if self.x_var[(day, transport_type, current_location, dest["name"])].solution_value > 0.5:
                                next_location = dest["name"]
                                chosen_transport = transport_type
                                travel_time = self.transport_matrix[(current_location, next_location, time_bracket)][transport_type]["duration"]
                                price = self.transport_matrix[(current_location, next_location, time_bracket)][transport_type]["price"]
                                entrance_fee = dest.get("entrance_fee", 0) if dest["type"] == "attraction" else None
                                duration = (dest.get("duration", 0) // 60, dest.get("duration", 0) % 60)
                                food_price = dest.get("avg_food_price", 0) if dest["type"] == "hawker" else None
                                satisfaction_score = dest.get("satisfaction", 0) if dest["type"] == "attraction" else None
                                rating = dest.get("rating", 0) if dest["type"] == "hawker" else None
                                break
                        if next_location:
                            break
                    if next_location:
                        break
                
                if not next_location:
                    break  # No more locations for the day

                # Arrival time
                arrival_time = self.u_var[(day, current_location)].solution_value

                # Print movement details
                print(f"{int(arrival_time // 60):02d}:{int(arrival_time % 60):02d} - From {current_location}, go to {next_location}")
                print("        Details:")
                print(f"         - use {chosen_transport}, price = ${price:.2f}, duration = {travel_time // 60} hours {travel_time % 60} minutes")
                
                if entrance_fee is not None:
                    print(f"         - {next_location} entrance fee = ${entrance_fee:.2f}")
                if duration is not None:
                    print(f"         - {next_location} duration = {duration[0]} hours {duration[1]} minutes")
                if satisfaction_score is not None:
                    print(f"         - {next_location} satisfaction score = {satisfaction_score:.2f} / 10")
                if food_price is not None:
                    print(f"         - {next_location} average food price = ${food_price:.2f}")
                if rating is not None:
                    print(f"         - {next_location} Rating = {rating:.2f} / 5")

                if next_location == hotel["name"]:
                    print(f"{int(arrival_time // 60):02d}:{int(arrival_time % 60):02d} - Go back to hotel")
                    print("        Details:")
                    print(f"         - use {chosen_transport}, price = ${price:.2f}, duration = {travel_time // 60} hours {travel_time % 60} minutes")
                    break

                current_location = next_location

            print()  # Blank line between days
        travel_time_total = self.travel_time_var.solution_value
        travel_time_hm = (travel_time_total//60, travel_time_total%60)
        print("Overall:")
        print(f"    Total Cost             = ${self.cost_var.solution_value:.2f}")
        print("    Total Travel Time      =", travel_time_hm[0], "hours", travel_time_hm[1], "minutes")
        print("    Estimated Satisfaction =", self.satisfaction_var.solution_value)


if __name__ == "__main__":
    random.seed(42)
    # load locations
    all_locations = get_all_locations()
    # for all locations, get satisfaction and rating
    for loc in all_locations:
        if loc["type"] == "hawker":
            loc["rating"] = np.random.uniform(0, 5)
            loc["avg_food_price"] = np.random.uniform(5, 15)
            loc["duration"] = 60 # just standardize 60 mins
        elif loc["type"] == "attraction":
            loc["satisfaction"] = np.random.uniform(0, 10)
            loc["entrance_fee"] = np.random.uniform(5, 100)
            loc["duration"] = np.random.randint(30, 90)
    # get hotel, add it to selected locations
    dummy_hotel = {
        "type": "hotel",
        "name": "DUMMY HOTEL",
        "lat": 1.2852044,
        "lng": 103.8610313,
    }
    # load transport_matrix
    transport_matrix = get_transport_matrix()
    # add dummy hotel to transport_matrix
    for loc in all_locations:
        for time_ in [8, 12, 16, 20]:
            transport_matrix[(dummy_hotel["name"], loc["name"], time_)] = {
                "transit": {
                    "duration": 50,
                    "price": 1.93,
                },
                "drive": {
                    "duration": 20,
                    "price": 10.1,
                }
            }
            transport_matrix[(loc["name"], dummy_hotel["name"], time_)] = {
                "transit": {
                    "duration": 50,
                    "price": 1.93,
                },
                "drive": {
                    "duration": 20,
                    "price": 10.1,
                }
            }

    locations = [dummy_hotel] + all_locations

    # select sublocations for smaller problem size
    hotels = [loc for loc in locations if loc["type"] == "hotel"]
    attractions = [loc for loc in locations if loc["type"] == "attraction"]
    hawkers = [loc for loc in locations if loc["type"] == "hawker"]

    selected_hotel = random.sample(hotels, 1)
    selected_attractions = random.sample(attractions, 5)
    selected_hawkers = random.sample(hawkers, 2)

    selected_locations = selected_hotel + selected_attractions + selected_hawkers

    problem = TravelItineraryProblem(
        budget=1000,
        locations=locations,
        # locations=selected_locations,
        transport_matrix=transport_matrix,
        num_days=3,
        priority_weights=[0.3, 0.3, 0.4],
    )

    problem.solve()
    problem.print_solution()
