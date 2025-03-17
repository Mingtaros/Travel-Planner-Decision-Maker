import numpy as np
from pymoo.core.problem import ElementwiseProblem

# from utils.transport_utility import get_transport_hour


class TravelItineraryProblem(ElementwiseProblem):

    def __init__(self, budget, locations, transport_matrix):
        # constants
        self.NUM_DAYS = 3
        self.HOTEL_COST = 50
        self.START_TIME = 9 * 60 # everyday starts from 9 AM
        self.HARD_LIMIT_END_TIME = 22 * 60 # everyday MUST return back to hotel by 10 PM

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

        lower_bound = np.concatenate([np.zeros(self.x_shape), np.full(self.u_shape, self.START_TIME)])
        upper_bound = np.concatenate([np.ones(self.x_shape), np.full(self.u_shape, self.HARD_LIMIT_END_TIME)])

        # inequality constraints
        #   - for every attraction, must be a source at least once
        #   - vice versa, for every attraction, must be a destination at least once
        #   - if source k to destination l chosen, make sure the time in var u is correct
        #       - from hotel, it can go to N-1 destinations (exclude hotel)
        #       - from any other destinations, it can go to N-2 destinations (exclude itself AND hotel)
        #   - every day, must go to hawker at least twice
        #   - if public transport is chosen, taxi can't be chosen
        #   - make sure everything is within budget
        num_inequality_constraints = \
            self.num_attractions + \
            self.num_attractions + \
            self.NUM_DAYS * self.num_transport_types * (self.num_locations - 1) + \
            self.NUM_DAYS * self.num_transport_types * (self.num_locations - 1) * (self.num_locations - 2) + \
            self.NUM_DAYS + \
            self.NUM_DAYS * self.num_locations * self.num_locations + \
            1

        # equality constraints
        #   - everyday, hotel must be starting
        #   - everyday, for every location, if selected as source, must be selected as destination
        #   - everyday, must return to hotel from last destination
        num_equality_constraints = \
            self.NUM_DAYS + \
            self.NUM_DAYS * self.num_locations + \
            self.NUM_DAYS

        super().__init__(
            n_var=num_vars,
            n_obj=3,
            n_ieq_constr=num_inequality_constraints + 1,
            n_eq_constr=num_equality_constraints + 1,
            xl=lower_bound,
            xu=upper_bound,
        )

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

        # for every attraction, must be a source at most once
        # for every attraction, must be a destination at most once
        # NOTE that this doesn't apply to hawkers
        for k in range(self.num_locations):
            if self.locations[k]["type"] == "attraction":
                out["G"].append(np.sum(x_var[:, :, k, :]) - 1)
                out["G"].append(np.sum(x_var[:, :, :, k]) - 1)

        for i in range(self.NUM_DAYS):
            # u_var[i, 0] must be the smallest of u_var[i]
            out["H"].append(np.min(u_var[i, :]) - u_var[i, 0])

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
                        out["G"].append(x_var[i, j, k, l] * time_finish_l - time_should_finish_l)
                        
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
            hawker_sum = 0
            for k in range(self.num_locations):
                # every day, for every location, if it's selected as source, it must be selected as destination
                out["H"].append(np.sum(x_var[i, :, :, k]) - np.sum(x_var[i, :, k, :]))

                if self.locations[k]["type"] == "hawker":
                    hawker_sum += np.sum(x_var[i, :, k, :])

            # every day, must go to hawkers at least twice (lunch & dinner. Can go more times if they want to)
            out["G"].append(2 - hawker_sum)

        for i in range(self.NUM_DAYS):
            for k in range(self.num_locations):
                for l in range(self.num_locations):
                    # for every day, if public transportation is chosen, taxi can't be chosen, and vice versa
                    # (sum j in transport_types x_ijkl <= 1) for all days, for all sources, for all destinations
                    out["G"].append(np.sum(x_var[i, :, k, l]) - 1)

        # finally, make sure everything is within budget
        out["G"].append(self.budget - total_cost)

        # <ADD ADDITIONAL CONSTRAINTS HERE>
        out["H"].append(np.sum(x_var) - 5) # should be == 5        
        day_one_attraction_limit = np.sum(x_var[0, :, :, :]) - 3 # should be <= 3
        out["G"].append(day_one_attraction_limit)
        
        # objectives
        out["F"] = [total_cost, total_travel_time, -total_satisfaction]

# TODO:
#   - fix the constraints..., for some reason no feasible solution
#   - make sure the hawker is visited DURING LUNCHTIME and DINNERTIME
#       - the "twice-a-day formulation" is there already, but not at the correct time
