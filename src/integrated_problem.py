import re
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination


def count_indentation(line_of_code):
    # given line of code, get the indentation to replicate in added constraints
    num_indent = 0
    for chara in line_of_code:
        if chara != " ":
            break
        num_indent += 1
    
    return num_indent


with open("src/base_problem.py", 'r') as base_problem_file:
    base_problem_str = base_problem_file.readlines()


inequality_constraints = [
    """
    day_one_attraction_limit = np.sum(x[0, :, :]) - 3 # should be <= 3
    out["G"].append(day_one_attraction_limit)
    """,
]
equality_constraints = [
    """out["H"].append(np.sum(x) - 5) # should be == 5""",
]

# update the number of constraints in class initialization
num_ieq_constrain_line = base_problem_str[16]
current_num_ieq_constraints = int(re.findall("n_ieq_constr=(.*),", num_ieq_constrain_line)[0])
new_num_ieq_constraints = current_num_ieq_constraints + len(inequality_constraints)
base_problem_str[16] = base_problem_str[16].replace(str(current_num_ieq_constraints), str(new_num_ieq_constraints))

num_eq_constrain_line = base_problem_str[17]
current_num_eq_constraints = int(re.findall("n_eq_constr=(.*),", num_eq_constrain_line)[0])
new_num_eq_constraints = current_num_eq_constraints + len(equality_constraints)
base_problem_str[17] = base_problem_str[17].replace(str(current_num_eq_constraints), str(new_num_eq_constraints))

# add additional constraints
# known location of <ADD ADDITIONAL CONSTRAINTS HERE> is in line 101
num_indent = count_indentation(base_problem_str[101]) # see indentation there, match in every added constraints
for constraint in inequality_constraints: # inequality constraints
    # add indent for each line
    constraint = [" " * num_indent + constraint_line.strip() for constraint_line in constraint.split("\n")]
    constraint = "\n".join(constraint) # re-join to make new constraint
    # add the constraint to the code
    base_problem_str.insert(102, constraint)

for constraint in equality_constraints: # equality constraints
    # add indent for each line
    constraint = [" " * num_indent + constraint_line.strip() for constraint_line in constraint.split("\n")]
    constraint = "\n".join(constraint) # re-join to make new constraint
    # add the constraint to the code
    base_problem_str.insert(102, constraint)

# have base problem set as None for defaulting in case of error
class TravelItineraryProblem():
    def __init__(self, **kwargs):
        pass

# execute the code inside base_problem_str, importing the Problem class.
exec("".join(base_problem_str))

# make the problemset and solve it.
problem = TravelItineraryProblem(
    budget=300,
    destinations=[
        {"type": "attraction", "duration": 120, "entrance_fee": 20, "satisfaction": 8},
        {"type": "attraction", "duration": 90, "entrance_fee": 15, "satisfaction": 7},
        {"type": "hawker", "ratings": 4.3},
        {"type": "hawker", "ratings": 3.4},
        {"type": "hotel"}
    ],
    public_transport_prices=np.random.rand(5, 5, 24).tolist(),
    taxi_prices=np.random.rand(5, 5, 24).tolist(),
    public_transport_durations=(np.random.rand(5, 5, 24) * 120).tolist(),
    taxi_durations=(np.random.rand(5, 5, 24) * 120).tolist()
)

algorithm = NSGA2(
    pop_size=100,
    sampling=BinaryRandomSampling(),
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
