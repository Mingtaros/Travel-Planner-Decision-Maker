from pymoo.core.initialization import Initialization
from pymoo.operators.sampling.rnd import FloatRandomSampling
import numpy as np

class HeuristicInitialization(Initialization):
    def __init__(self, heuristic_solution, n_random=99):
        """
        Initialize with a heuristic solution and randomly generated solutions
        
        Args:
            heuristic_solution: The pre-computed heuristic solution(s)
            n_random: Number of random solutions to generate (total pop = n_random + len(heuristic_solution))
        """
        super().__init__(sampling=FloatRandomSampling())
        self.heuristic_solution = heuristic_solution
        self.n_random = n_random
        
    def do(self, problem, n_samples, **kwargs):
        # Generate random solutions
        random_sols = np.random.random((self.n_random, problem.n_var))
        # Scale random solutions to problem bounds
        xl, xu = problem.xl, problem.xu
        random_sols = xl + random_sols * (xu - xl)
        
        # If heuristic solution is a single solution, reshape to 2D
        if self.heuristic_solution.ndim == 1:
            heuristic_sols = self.heuristic_solution.reshape(1, -1)
        else:
            heuristic_sols = self.heuristic_solution
            
        # Combine heuristic and random solutions
        pop = np.vstack([heuristic_sols, random_sols])
        
        return pop
    
    def create_heuristic_solution(problem):
        """Create a heuristic solution for the travel itinerary problem"""
        
        xl_dtype = problem.xl.dtype
        xu_dtype = problem.xu.dtype
        
        # Initialize solution vector with zeros
        n_var = problem.n_var
        solution = np.zeros(n_var, dtype=xl_dtype)
        
        # Extract problem parameters
        x_shape = problem.x_shape
        u_shape = problem.u_shape
        num_days = problem.NUM_DAYS
        num_locations = problem.num_locations
        
        # For each day, create a simple schedule:
        # 1. Start at hotel
        # 2. Visit a hawker for lunch during lunch time
        # 3. Visit 1-2 attractions
        # 4. Visit another hawker for dinner during dinner time
        # 5. Return to hotel
        
        current_time = problem.START_TIME
        
        for day in range(num_days):
            # Step 1: Start at hotel (location 0)
            current_loc = 0
            
            # Find indices of hawkers and attractions
            hawkers = [i for i, loc in enumerate(problem.locations) if loc["type"] == "hawker"]
            attractions = [i for i, loc in enumerate(problem.locations) if loc["type"] == "attraction"]
            
            # Randomize to get some variety
            np.random.shuffle(hawkers)
            np.random.shuffle(attractions)
            
            # Step 2: Go to a hawker for lunch during lunch time
            if hawkers:
                lunch_hawker = hawkers[0]
                # Select transit mode (index 0)
                transport_mode = 0
                # Set the route from hotel to lunch hawker
                solution[day * problem.num_transport_types * problem.num_locations * problem.num_locations + 
                    transport_mode * problem.num_locations * problem.num_locations +
                    current_loc * problem.num_locations +
                    lunch_hawker] = 1
                
                # Update current location
                current_loc = lunch_hawker
                
                # Set lunch time to be within lunch window
                lunch_time = problem.LUNCH_START + 30  # 30 minutes into lunch window
                # Set u variable for lunch hawker
                solution[x_shape + day * problem.num_locations + lunch_hawker] = lunch_time + problem.locations[lunch_hawker]["duration"]
            
            # Step 3: Visit 1-2 attractions
            for i in range(min(2, len(attractions))):
                attraction = attractions[i]
                # Set the route from current location to attraction
                solution[day * problem.num_transport_types * problem.num_locations * problem.num_locations + 
                    transport_mode * problem.num_locations * problem.num_locations +
                    current_loc * problem.num_locations +
                    attraction] = 1
                    
                # Update current location
                current_loc = attraction
                
                # Set time for attraction visit (after lunch)
                attraction_time = lunch_time + problem.locations[lunch_hawker]["duration"] + 30 + i * 90  # 30 min transport + 90 min per attraction
                # Set u variable for attraction
                solution[x_shape + day * problem.num_locations + attraction] = attraction_time + problem.locations[attraction]["duration"]
            
            # Step 4: Go to a hawker for dinner
            if len(hawkers) > 1:
                dinner_hawker = hawkers[1]
                # Set the route from current location to dinner hawker
                solution[day * problem.num_transport_types * problem.num_locations * problem.num_locations + 
                    transport_mode * problem.num_locations * problem.num_locations +
                    current_loc * problem.num_locations +
                    dinner_hawker] = 1
                    
                # Update current location
                current_loc = dinner_hawker
                
                # Set dinner time to be within dinner window
                dinner_time = problem.DINNER_START + 30  # 30 minutes into dinner window
                # Set u variable for dinner hawker
                solution[x_shape + day * problem.num_locations + dinner_hawker] = dinner_time + problem.locations[dinner_hawker]["duration"]
            
            # Step 5: Return to hotel
            solution[day * problem.num_transport_types * problem.num_locations * problem.num_locations + 
                transport_mode * problem.num_locations * problem.num_locations +
                current_loc * problem.num_locations +
                0] = 1  # 0 is the hotel index
        
        if np.issubdtype(solution.dtype, np.floating):
            # For floating point arrays, round to ensure exact 0.0 or 1.0
            binary_part = solution[:problem.x_shape]
            binary_part = np.round(binary_part)
            solution[:problem.x_shape] = binary_part
            print("Rounded binary variables to exact 0.0 or 1.0 values")
        
        print(f'solution:{solution}')
        
        return solution