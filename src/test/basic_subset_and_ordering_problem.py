"""
 example code using PyMOO that tackles a subset selection itinerary problem. 
 In this formulation, you decide which attractions (from a total of n attractions) 
 to include and in what order to visit them, with the objective of minimizing the total travel cost.
"""
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA

# Define a cost matrix where:
# - Index 0 represents the start location.
# - Indices 1 to n represent attractions.
cost_matrix = np.array([
    [ 0, 12, 10, 15, 20, 10],
    [12,  0,  8, 10, 25, 15],
    [10,  8,  0, 12, 18, 20],
    [15, 10, 12,  0, 10, 22],
    [20, 25, 18, 10,  0, 15],
    [10, 15, 20, 22, 15,  0]
])

n_attractions = 5  # Number of available attractions (indices 1 to 5)
budget = 50         # Set your budget constraint here

class SubsetItineraryProblemWithBudget(ElementwiseProblem):
    def __init__(self, budget):
        self.budget = budget
        # Now we have 2*n_attractions decision variables and 1 constraint.
        super().__init__(n_var=2 * n_attractions, n_obj=1, n_constr=1, xl=0, xu=1)
    
    def _evaluate(self, x, out, *args, **kwargs):
        # Determine which attractions are selected (threshold > 0.5).
        selected = []
        for i in range(n_attractions):
            if x[i] > 0.5:
                selected.append((i, x[i + n_attractions]))
        
        # If no attraction is selected, assign a heavy penalty to both objective and constraint.
        if not selected:
            out["F"] = 1e6
            out["G"] = 1e6  # Constraint violated
            return
        
        # Sort the selected attractions based on their ordering key.
        selected.sort(key=lambda tup: tup[1])
        order = [t[0] for t in selected]
        
        # Build the complete route:
        # - Start at index 0.
        # - Visit selected attractions (shifted by 1 since attractions are indices 1..n).
        # - Return to start (index 0).
        route = [0] + [i + 1 for i in order] + [0]
        
        # Compute the total travel cost along the route.
        total_cost = 0
        for j in range(len(route) - 1):
            total_cost += cost_matrix[route[j], route[j + 1]]
        
        out["F"] = total_cost
        # Constraint: total_cost must not exceed the budget.
        # Constraint is satisfied if total_cost - budget <= 0.
        out["G"] = total_cost - self.budget

# Use a Genetic Algorithm (GA) with a population size of 50.
algorithm = GA(pop_size=50)

# Instantiate and solve the problem with the budget constraint.
problem = SubsetItineraryProblemWithBudget(budget)  
res = minimize(problem, algorithm, seed=1, termination=('n_gen', 100), verbose=True)

# Decode the best solution found.
best_x = res.X
selected = []
for i in range(n_attractions):
    if best_x[i] > 0.5:
        selected.append((i, best_x[i + n_attractions]))

if not selected:
    print("No attraction selected.")
else:
    selected.sort(key=lambda tup: tup[1])
    order = [i for i, _ in selected]
    # Build the final route with the start at index 0.
    route = [0] + [i + 1 for i in order] + [0]
    print("Selected attractions (0-indexed):", order)
    print("Complete route (with start as 0):", route)
    print("Total travel cost:", res.F)
    # Optionally, show the constraint value:
    print("Constraint value (should be <= 0):", res.G)