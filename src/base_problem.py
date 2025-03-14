import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination

class TravelItineraryProblem(ElementwiseProblem):

    def __init__(self, budget, destinations, public_transport_prices, taxi_prices, public_transport_durations, taxi_durations):
        self.budget = budget
        self.destinations = destinations
        self.public_transport_prices = public_transport_prices
        self.taxi_prices = taxi_prices
        self.public_transport_durations = public_transport_durations
        self.taxi_durations = taxi_durations
        self.num_destinations = len(destinations)
        super().__init__(
            n_var=3 * 2 * self.num_destinations, # [day][use taxi / public transport][destination]
            n_obj=3,
            n_constr=23,
            xl=0,
            xu=1
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x = x.reshape(3, 2, self.num_destinations)
        # explanation on x:
        #   - index 1: day
        #   - index 2: whether to use taxi or not
        #   - index 3: destination
        total_cost = 0
        total_time = 0
        total_satisfaction = 0
        visited = np.zeros(self.num_destinations)
        current_time = 9 * 60  # Start at 9 AM
        current_location = 0  # Start at the hotel

        for day in range(3):  # Iterate over 3 days
            current_time = 9 * 60  # Start each day at 9 AM
            current_location = 0  # Start each day at the hotel
            for i in range(self.num_destinations):
                if x[day][0][i] or x[day][1][i]: # Check if the destination is visited on this day
                    destination = self.destinations[i]
                    
                    # Decide whether to use taxi or public transport
                    if x[day][1][i]:
                        travel_time = self.taxi_durations[current_location][i][int(current_time // 60)]
                        travel_cost = self.taxi_prices[current_location][i][int(current_time // 60)]
                    else:  # Use public transport otherwise
                        travel_time = self.public_transport_durations[current_location][i][int(current_time // 60)]
                        travel_cost = self.public_transport_prices[current_location][i][int(current_time // 60)]
                    
                    current_time += travel_time
                    total_time += travel_time
                    total_cost += travel_cost
                    
                    # Spend time and money at destination
                    if destination["type"] == "attraction":
                        current_time += destination["duration"]
                        total_cost += destination["entrance_fee"]
                        total_satisfaction += destination["satisfaction"]
                    elif destination["type"] == "hawker":
                        current_time += 60
                        total_cost += 10
                        total_satisfaction += destination["ratings"]
                    
                    # Update location
                    visited[i] = 1
                    current_location = i

            # Ensure the day ends back at the hotel around 9 PM
            if current_location != 0:
                if x[day][1][0]:  # Use taxi to return to hotel
                    travel_time = self.taxi_durations[current_location][i][int(current_time // 60)]
                    travel_cost = self.taxi_prices[current_location][i][int(current_time // 60)]
                else:  # Use public transport otherwise
                    travel_time = self.public_transport_durations[current_location][i][int(current_time // 60)]
                    travel_cost = self.public_transport_prices[current_location][i][int(current_time // 60)]
                
                current_time += travel_time
                total_time += travel_time
                total_cost += travel_cost

            # Check if the return time is around 9 PM
            out["G"] = [total_cost - self.budget, np.sum(visited) - self.num_destinations, current_time - 21 * 60]

        total_cost += 3 * 50  # Hotel cost for 3 nights

        # visited constraint (each destination should only be visited once)
        visited_count = np.sum(x, axis=(0, 1))  # Sum over all days and transport modes
        constraint_visit_once = visited_count - 1  # Should be <= 0 (i.e., at most 1 visit)
        out["G"].extend(constraint_visit_once.tolist())

        # transportation constraint (if taxi is used, public transportation is not, and vice versa)
        conflicting_transport = np.sum(x, axis=1) - 1 # Should be <= 0 (sum per day should be at most 1)
        out["G"].extend(conflicting_transport.flatten().tolist())

        # print(out["G"])

        out["F"] = [total_cost, total_time, -total_satisfaction]
