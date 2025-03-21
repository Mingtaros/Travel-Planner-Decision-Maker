"""
n_gen: This is the generation number/ aka itinerary set is explored. Each generation represents one iteration of the genetic algorithm 
        where new candidate solutions (itineraries) are evaluated.
n_eval: This column shows the cumulative number of evaluations (i.e., how many candidate solutions have been assessed) up to that generation.
f_avg: This is the average objective function value (in this case, the travel cost) across all solutions in the current generation.
f_min: This is the minimum (i.e., best) objective function value found in that generation. 
"""

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA

# Define a simple cost matrix for 5 cities
cost_matrix = np.array([
    [0, 10, 15, 20, 10],
    [10, 0, 35, 25, 15],
    [15, 35, 0, 30, 20],
    [20, 25, 30, 0, 25],
    [10, 15, 20, 25, 0]
])

n_cities = cost_matrix.shape[0]

class TSPProblem(ElementwiseProblem):
    def __init__(self):
        # Each decision variable is in the range [0, 1]
        super().__init__(n_var=n_cities, n_obj=1, n_constr=0, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        # Convert continuous solution to a permutation using the "random keys" method:
        # The permutation is determined by sorting the vector x.
        permutation = np.argsort(x)
        
        # Calculate total travel cost following the order of cities in 'permutation'
        cost = 0
        for i in range(n_cities - 1):
            cost += cost_matrix[permutation[i], permutation[i+1]]
        # Optionally, return to the starting city
        cost += cost_matrix[permutation[-1], permutation[0]]
        out["F"] = cost

# Use a Genetic Algorithm (GA) provided by PyMOO
algorithm = GA(pop_size=50)

# Instantiate and solve the problem
problem = TSPProblem()
res = minimize(problem, algorithm, seed=1, termination=('n_gen', 100), verbose=True)

# The best solution's permutation can be derived by sorting the best found vector
best_permutation = np.argsort(res.X)
print("Best permutation found:", best_permutation)
print("Total cost:", res.F)