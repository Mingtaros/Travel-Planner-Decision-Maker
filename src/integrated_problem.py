import re
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from utils.transport_utility import get_transport_matrix, get_all_locations

np.random.seed(42)

def count_indentation(line_of_code):
    # given line of code, get the indentation to replicate in added constraints
    num_indent = 0
    for chara in line_of_code:
        if chara != " ":
            break
        num_indent += 1
    
    return num_indent

def integrate_problem(base_problem_str, inequality_constraints, equality_constraints):
    # update the number of constraints in class initialization
    base_problem_str[69] = base_problem_str[69].replace(",", " + " + str(len(inequality_constraints)) + ",")
    base_problem_str[70] = base_problem_str[70].replace(",", " + " + str(len(equality_constraints)) + ",")

    # add additional constraints
    # known location of <ADD ADDITIONAL CONSTRAINTS HERE> is in this line
    num_indent = count_indentation(base_problem_str[178]) # see indentation there, match in every added constraints
    for constraint in inequality_constraints: # inequality constraints
        # add indent for each line
        constraint = [" " * num_indent + constraint_line.strip() for constraint_line in constraint.split("\n")]
        constraint = "\n".join(constraint) # re-join to make new constraint
        # add the constraint to the code
        base_problem_str.insert(179, constraint)

    for constraint in equality_constraints: # equality constraints
        # add indent for each line
        constraint = [" " * num_indent + constraint_line.strip() for constraint_line in constraint.split("\n")]
        constraint = "\n".join(constraint) # re-join to make new constraint
        # add the constraint to the code
        base_problem_str.insert(179, constraint)
    
    return base_problem_str


if __name__ == "__main__":
    with open("src/base_problem.py", 'r') as base_problem_file:
        base_problem_str = base_problem_file.readlines()


    inequality_constraints = [
        """
        day_one_attraction_limit = np.sum(x_var[0, :, :, :]) - 3 # should be <= 3
        out["G"].append(day_one_attraction_limit)
        """,
    ]
    equality_constraints = [
        """out["H"].append(np.sum(x_var) - 5) # should be == 5""",
    ]

    base_problem_str = integrate_problem(base_problem_str, inequality_constraints, equality_constraints)

    # have base problem set as None for defaulting in case of error
    class TravelItineraryProblem():
        def __init__(self, **kwargs):
            pass

    with open("src/test_integrated.py", 'w') as f:
        f.writelines(base_problem_str)
    # execute the code inside base_problem_str, importing the Problem class.
    exec("".join(base_problem_str))

    all_locations = get_all_locations()

    # for all locations, get necessary data
    for loc in all_locations:
        if loc["type"] == "hawker":
            loc["rating"] = np.random.uniform(0, 5)
            loc["avg_food_price"] = np.random.uniform(5, 15)
            loc["duration"] = 60 # just standardize 60 mins
        elif loc["type"] == "attraction":
            loc["satisfaction"] = np.random.uniform(0, 10)
            loc["entrance_fee"] = np.random.uniform(5, 100)
            loc["duration"] = np.random.randint(30, 90)

    dummy_hotel = {
        "type": "hotel",
        "name": "DUMMY HOTEL",
        "lat": 1.2852044,
        "lng": 103.8610313,
    }
    transport_matrix = get_transport_matrix()
    # add dummy hotel to transport_matrix
    for loc in all_locations:
        time_brackets = [8, 12, 16, 20]
        for time_ in time_brackets:
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

    # make the problemset and solve it.
    problem = TravelItineraryProblem(
        budget=300,
        locations=locations,
        transport_matrix=transport_matrix,
    )

    algorithm = NSGA2(
        pop_size=100,
        sampling=IntegerRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 200)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        save_history=True,
        verbose=True
    )

    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
