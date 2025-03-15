import numpy as np
from pymoo.core.problem import ElementwiseProblem

class TravelItineraryProblem(ElementwiseProblem):

    def __init__(self, budget, destinations, public_transport_prices, taxi_prices, public_transport_durations, taxi_durations):
        # constants
        self.NUM_DAYS = 3
        self.HOTEL_COST = 50
        self.START_TIME = 9 * 60 # everyday starts from 9 AM
        self.HARD_LIMIT_END_TIME = 22 * 60 # everyday MUST return back to hotel by 10 PM

        # hard limit money spent
        self.budget = budget
        # list of destinations
        self.places = destinations
        self.num_destinations = len(destinations)
        self.num_attractions = len([place for place in self.places if place["type"] == "attraction"])
        self.num_hawkers = len([place for place in self.places if place["type"] == "hawker"])

        # transportation option prices and durations. The size MUST BE num_dest * num_dest, for 24 hours
        self.transport_types = 2
        self.transport_prices = np.array(public_transport_prices + taxi_prices) \
            .reshape(self.transport_types, self.num_destinations, self.num_destinations, 24)
        self.transport_durations = np.array(public_transport_durations + taxi_durations) \
            .reshape(self.transport_types, self.num_destinations, self.num_destinations, 24)

        # x_ijkl = BINARY VAR, if the route goes in day i, use transport type j, from location k, to location l
        self.x_shape = self.NUM_DAYS * self.transport_types * self.num_destinations * self.num_destinations
        # u_ik  = CONTINUOUS VAR, tracking FINISH time of the person in day i, location k
        self.u_shape = self.NUM_DAYS * self.num_destinations

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
            self.NUM_DAYS * self.transport_types * (self.num_destinations - 1) + \
            self.NUM_DAYS * self.transport_types * (self.num_destinations - 1) * (self.num_destinations - 2) + \
            self.NUM_DAYS + \
            self.NUM_DAYS * self.num_destinations * self.num_destinations + \
            1

        # equality constraints
        #   - everyday, hotel must be starting
        #   - everyday, for every attraction, if selected as source, must be selected as destination
        #   - everyday, must return to hotel from last destination
        num_equality_constraints = \
            self.NUM_DAYS + \
            self.NUM_DAYS + \
            self.NUM_DAYS

        super().__init__(
            n_var=num_vars,
            n_obj=3,
            n_ieq_constr=num_inequality_constraints,
            n_eq_constr=num_equality_constraints,
            xl=lower_bound,
            xu=upper_bound,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # they're ensured to be integers
        x_var = x[:self.x_shape].reshape(self.NUM_DAYS, self.transport_types, self.num_destinations, self.num_destinations)
        u_var = x[self.x_shape:].reshape(self.NUM_DAYS, self.num_destinations)

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
        for k in range(self.num_destinations):
            if self.places[k]["type"] == "attraction":
                out["G"].append(np.sum(x_var[:, :, k, :]) - 1)
                out["G"].append(np.sum(x_var[:, :, :, k]) - 1)

        for i in range(self.NUM_DAYS):
            # every day, hotel must be starting
            out["H"].append(self.START_TIME - u_var[i, 0])

            # every day, for every attraction, if it's selected as source, it must be selected as destination
            out["H"].append(np.sum(x_var[i, :, :, k]) - np.sum(x_var[i, :, k, :]))

            for j in range(self.transport_types):
                for k in range(self.num_destinations):
                    for l in range(1, self.num_destinations): # NOTE here that hotel (index 0) isn't included in destination. It will be computed later
                        if k == l: continue
                        # every day, if the route is chosen, the time of finishing in this place is this

                        # if chosen, then time_should_finish_l must be time_finish_l, otherwise, don't matter (just put 0)
                        # you should finish attraction l at:
                        #   - time to finish k + 
                        #   - time to transport from k to l, using transport method j +
                        #   - time to play at l
                        time_finish_l = u_var[i, l]
                        transport_hour = u_var[i, k] // 60
                        time_should_finish_l = u_var[i, k] + self.transport_durations[j][k][l][transport_hour] + self.places[l]["duration"]
                        # if x_var[i, j, k, l] is not chosen, then this constraint don't matter
                        out["G"].append(x_var[i, j, k, l] * time_finish_l - time_should_finish_l)
                        
                        # append to total travel time spent
                        if x_var[i, j, k, l] == 1:
                            # calculate the travel
                            total_travel_time += self.transport_durations[j][k][l][transport_hour]
                            total_cost += self.transport_prices[j][k][l][transport_hour]

                            # calculate cost and satisfaction, based on what they come to
                            # Note: not counting anything for hotel.
                            if self.places[l]["type"] == "attraction":
                                total_cost += self.places[l]["entrance_fee"]
                                total_satisfaction += self.places[l]["satisfaction"]
                            elif self.places[l]["type"] == "hawker":
                                total_cost += 10 # ASSUME eating in a hawker is ALWAYS $10
                                total_satisfaction += self.places[l]["ratings"]

            # from last place, return to hotel.
            last_place = np.argmax(u_var[i, :])
            latest_hour = u_var[i, last_place] // 60
            # MUST go back to hotel at the end of the day
            out["H"].append(np.sum(x_var[i, :, last_place, 0]) - 1)
            # go back to hotel
            # choose whether to use which transport type
            for j in range(self.transport_types):
                if x_var[i, j, last_place, 0]:
                    total_travel_time += self.transport_durations[j][last_place][0][latest_hour]
                    total_cost += self.transport_prices[j][last_place][0][latest_hour]
                    break

        for i in range(self.NUM_DAYS):
            hawker_sum = 0
            for k in range(self.num_destinations):
                if self.places[k]["type"] == "hawker":
                    hawker_sum += np.sum(x_var[i, :, k, :])

            # every day, must go to hawkers at least twice (lunch & dinner. Can go more times if they want to)
            out["G"].append(2 - hawker_sum)

        for i in range(self.NUM_DAYS):
            for k in range(self.num_destinations):
                for l in range(self.num_destinations):
                    # for every day, if public transportation is chosen, taxi can't be chosen, and vice versa
                    # (sum j in transport_types x_ijkl <= 1) for all days, for all sources, for all destinations
                    out["G"].append(np.sum(x_var[i, :, k, l]) - 1)

        # finally, make sure everything is within budget
        out["G"].append(self.budget - total_cost)

        # <ADD ADDITIONAL CONSTRAINTS HERE>

        # objectives
        out["F"] = [total_cost, total_travel_time, -total_satisfaction]

# TODO:
#   - fix the constraints..., for some reason no feasible solution
#   - make sure the hawker is visited DURING LUNCHTIME and DINNERTIME
#       - the "twice-a-day formulation" is there already, but not at the correct time
