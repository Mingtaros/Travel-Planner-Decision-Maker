import random
import math
import numpy as np
import logging
import os
import time
import gc
import concurrent.futures
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from alns.alns_init_solution import create_initial_solution

logger = logging.getLogger("alns")

class ALNS:
    """
    Enhanced Adaptive Large Neighborhood Search implementation for the Travel Itinerary Problem
    with parallel evaluation, adaptive parameter tuning, early termination, and memory optimization.
    """
    def __init__(
        self, 
        problem, 
        initial_solution=None, 
        destroy_operators=None, 
        repair_operators=None,
        weights_destroy=None, 
        weights_repair=None,
        max_iterations=1000, 
        segment_size=100, 
        reaction_factor=0.5, 
        decay_factor=0.8,
        temperature_control=0.95, 
        initial_temperature=None,
        time_limit=None, 
        seed=None,
        early_termination_iterations=200,  # Iterations without improvement for early termination
        parallel_evaluations=True,         # Enable parallel evaluations
        max_workers=4,                     # Maximum number of worker threads
        adaptive_parameters=True,          # Enable adaptive parameter tuning
        memory_efficient=True              # Enable memory optimization
    ):
        """
        Initialize the ALNS algorithm with enhanced features
        
        Args:
            problem: The TravelItineraryProblem instance
            initial_solution: Initial solution (if None, will create a heuristic solution)
            destroy_operators: List of destroy operator functions
            repair_operators: List of repair operator functions
            weights_destroy: Initial weights for destroy operators
            weights_repair: Initial weights for repair operators
            max_iterations: Maximum number of iterations
            segment_size: Number of iterations before weights are updated
            reaction_factor: How strongly to adjust weights based on performance
            decay_factor: How much to decay weights over time
            temperature_control: Temperature decrease factor for simulated annealing
            initial_temperature: Initial temperature for simulated annealing
            time_limit: Time limit in seconds
            seed: Random seed for reproducibility
            early_termination_iterations: Number of iterations without improvement to trigger early termination
            parallel_evaluations: Whether to use parallel evaluation of solutions
            max_workers: Maximum number of worker threads for parallel evaluation
            adaptive_parameters: Whether to use adaptive parameter tuning
            memory_efficient: Whether to use memory-efficient data structures
        """
        self.problem = problem
        self.max_iterations = max_iterations
        self.segment_size = segment_size
        self.reaction_factor = reaction_factor
        self.decay_factor = decay_factor
        self.temperature_control = temperature_control
        self.time_limit = time_limit
        self.early_termination_iterations = early_termination_iterations
        self.parallel_evaluations = parallel_evaluations
        self.max_workers = max_workers
        self.adaptive_parameters = adaptive_parameters
        self.memory_efficient = memory_efficient
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize destroy operators
        if destroy_operators is None:
            from alns.destroy_operators import (
                destroy_random_days,
                destroy_random_attractions,
                destroy_worst_attractions,
                destroy_random_meals,
                destroy_random_routes,
                # destroy_time_aware_routes  
            )
            self.destroy_operators = [
                destroy_random_days,
                destroy_random_attractions,
                destroy_worst_attractions,
                destroy_random_meals,
                destroy_random_routes,
                # destroy_time_aware_routes  
            ]
        else:
            self.destroy_operators = destroy_operators
        
        # Initialize repair operators
        if repair_operators is None:
            from alns.repair_operators import (
                repair_greedy,
                repair_random,
                repair_regret,
                repair_satisfaction_based,
                repair_time_based,
                # repair_time_consistent
            )
            self.repair_operators = [
                repair_greedy,
                repair_random,
                repair_regret,
                repair_satisfaction_based,
                repair_time_based,
                # repair_time_consistent  
            ]
        else:
            self.repair_operators = repair_operators
        
        # Initialize weights for operators
        if weights_destroy is None:
            self.weights_destroy = [1.0] * len(self.destroy_operators)
        else:
            self.weights_destroy = weights_destroy
            
        if weights_repair is None:
            self.weights_repair = [1.0] * len(self.repair_operators)
        else:
            self.weights_repair = weights_repair
        
        # Initialize operator scores for this segment
        self.scores_destroy = [0] * len(self.destroy_operators)
        self.scores_repair = [0] * len(self.repair_operators)
        
        # Initialize counters
        self.destroy_count = [0] * len(self.destroy_operators)
        self.repair_count = [0] * len(self.repair_operators)
        
        # Initialize performance metrics for operators
        self.destroy_performance = [[] for _ in range(len(self.destroy_operators))]
        self.repair_performance = [[] for _ in range(len(self.repair_operators))]
        
        # Initialize current solution
        if initial_solution is not None:
            self.current_solution = initial_solution
        else:
            self.current_solution = create_initial_solution(self.problem, self.memory_efficient, self.optimize_numpy_arrays)
        
        # Evaluate current solution
        self.current_evaluation = self.problem.evaluate_solution(self.current_solution)
        self.current_objective = self.calculate_objective(self.current_evaluation)
        
        # Initialize best solution
        self.best_solution = self.current_solution.copy()
        if self.memory_efficient:
            self.best_solution_sparse = self.create_sparse_solution_representation(self.best_solution)
        self.best_evaluation = self.current_evaluation
        self.best_objective = self.current_objective
        
        # Initialize simulated annealing parameters
        if initial_temperature is None:
            # Auto-calibrate initial temperature
            self.initial_temperature = self.calibrate_temperature()
        else:
            self.initial_temperature = initial_temperature
            
        self.temperature = self.initial_temperature
        
        # Tracking performance
        self.iteration_history = []
        self.objective_history = [self.current_objective]
        self.temperature_history = [self.temperature]
        
        # For early termination
        self.iterations_since_improvement = 0
        
        # For diverse solutions
        self.diverse_solutions = []
        
        # For memory metrics
        self.memory_metrics = {
            "peak_memory": 0,
            "current_memory": 0
        }
        
        # For parallel execution
        if self.parallel_evaluations:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
    
    def __del__(self):
        """Clean up resources when the object is deleted"""
        if hasattr(self, 'executor') and self.parallel_evaluations:
            self.executor.shutdown()
    
    def calibrate_temperature(self, num_samples=100, acceptance_rate=0.5):
        """
        Auto-calibrate the initial temperature for simulated annealing
        
        Args:
            num_samples: Number of random perturbations to sample
            acceptance_rate: Target acceptance rate
            
        Returns:
            float: Calibrated initial temperature
        """
        logger.info("Calibrating initial temperature...")
        
        # Sample objective value differences from random destroy-repair operations
        objective_diffs = []
        
        # Create a copy of the current solution for experiments
        solution_copy = self.current_solution.copy()
        
        for _ in range(num_samples):
            # Apply random destroy operator
            destroy_idx = random.randrange(len(self.destroy_operators))
            destroyed_solution = self.destroy_operators[destroy_idx](self.problem, solution_copy.copy())
            
            # Apply random repair operator
            repair_idx = random.randrange(len(self.repair_operators))
            repaired_solution = self.repair_operators[repair_idx](self.problem, destroyed_solution.copy())
            
            # Evaluate new solution
            new_evaluation = self.problem.evaluate_solution(repaired_solution)
            new_objective = self.calculate_objective(new_evaluation)
            
            # Record the objective difference (only if it's worse)
            diff = new_objective - self.current_objective
            if diff > 0:
                objective_diffs.append(diff)
        
        if not objective_diffs:
            # If all samples were better, use a low temperature
            logger.info("All calibration samples were better than current solution. Using default temperature.")
            return 1.0
        
        # Calculate temperature where acceptance_rate of worse moves are accepted
        objective_diffs.sort()
        diff_idx = int(len(objective_diffs) * acceptance_rate)
        if diff_idx >= len(objective_diffs):
            diff_idx = len(objective_diffs) - 1
            
        # Temperature calculation: -diff/ln(acceptance_rate)
        temperature = -objective_diffs[diff_idx] / math.log(acceptance_rate)
        
        logger.info(f"Calibrated initial temperature: {temperature:.4f}")
        return temperature
    
    def calculate_objective(self, evaluation):
        """
        Calculate a single objective value from a solution evaluation
        
        Args:
            evaluation: Solution evaluation dictionary
            
        Returns:
            float: Weighted objective value (lower is better)
        """
        # Extract objectives
        cost = evaluation["total_cost"]
        travel_time = evaluation["total_travel_time"]
        satisfaction = evaluation["total_satisfaction"]
        
        # Normalize and combine objectives (weighted sum)
        # Cost weight: 0.3, Travel time weight: 0.3, Satisfaction weight: 0.4 (negative since we maximize)
        objective = 0.3 * (cost / self.problem.budget) + 0.3 * (travel_time / (self.problem.NUM_DAYS * 12 * 60)) - 0.4 * (satisfaction / (self.problem.num_attractions * 10 + self.problem.num_hawkers * 5))
        
        # Add large penalty for infeasible solutions
        if not evaluation["is_feasible"]:
            # Count total violations
            constraint_violations = len(evaluation["inequality_violations"]) + len(evaluation["equality_violations"])
            objective += 10.0 * constraint_violations
        
        return objective
    
    def select_destroy_operator(self):
        """
        Select a destroy operator using roulette wheel selection
        
        Returns:
            int: Index of the selected destroy operator
        """
        weights = self.weights_destroy
        if sum(weights) == 0:
            # If all weights are zero, use equal weights
            weights = [1] * len(self.destroy_operators)
            
        # Roulette wheel selection
        cum_weights = np.cumsum(weights)
        r = random.random() * cum_weights[-1]
        for i, w in enumerate(cum_weights):
            if r <= w:
                return i
    
    def select_repair_operator(self):
        """
        Select a repair operator using roulette wheel selection
        
        Returns:
            int: Index of the selected repair operator
        """
        weights = self.weights_repair
        if sum(weights) == 0:
            # If all weights are zero, use equal weights
            weights = [1] * len(self.repair_operators)
            
        # Roulette wheel selection
        cum_weights = np.cumsum(weights)
        r = random.random() * cum_weights[-1]
        for i, w in enumerate(cum_weights):
            if r <= w:
                return i
    
    def accept_solution(self, new_objective):
        """
        Decide whether to accept a new solution using simulated annealing
        
        Args:
            new_objective: Objective value of the new solution
            
        Returns:
            bool: True if the new solution is accepted, False otherwise
        """
        # Calculate objective difference (minimize objective)
        delta = new_objective - self.current_objective
        
        # Always accept better solutions
        if delta <= 0:
            return True
        
        # Accept worse solutions with a probability based on temperature
        acceptance_prob = math.exp(-delta / self.temperature)
        return random.random() < acceptance_prob
    
    def update_weights(self):
        """
        Update the weights of destroy and repair operators based on their performance
        """
        # Decay existing weights
        self.weights_destroy = [w * self.decay_factor for w in self.weights_destroy]
        self.weights_repair = [w * self.decay_factor for w in self.weights_repair]
        
        # Update weights based on scores in this segment
        for i in range(len(self.destroy_operators)):
            if self.destroy_count[i] > 0:
                self.weights_destroy[i] += self.reaction_factor * (self.scores_destroy[i] / self.destroy_count[i])
                # Store performance metrics for adaptive parameter tuning
                self.destroy_performance[i].append(self.scores_destroy[i] / self.destroy_count[i])
        
        for i in range(len(self.repair_operators)):
            if self.repair_count[i] > 0:
                self.weights_repair[i] += self.reaction_factor * (self.scores_repair[i] / self.repair_count[i])
                # Store performance metrics for adaptive parameter tuning
                self.repair_performance[i].append(self.scores_repair[i] / self.repair_count[i])
        
        # Reset scores and counts for the next segment
        self.scores_destroy = [0] * len(self.destroy_operators)
        self.scores_repair = [0] * len(self.repair_operators)
        self.destroy_count = [0] * len(self.destroy_operators)
        self.repair_count = [0] * len(self.repair_operators)
    
    def update_temperature(self, iterations_remaining, total_iterations):
        """
        Update the temperature for simulated annealing with adaptive cooling
        
        Args:
            iterations_remaining: Number of iterations remaining
            total_iterations: Total number of iterations
        """
        if self.adaptive_parameters:
            # Adaptive cooling - cool faster when approaching the end
            progress = 1 - (iterations_remaining / total_iterations)
            if progress > 0.75:
                # Accelerated cooling in the final stages
                self.temperature *= self.temperature_control**2
            else:
                # Normal cooling in the early stages
                self.temperature *= self.temperature_control
        else:
            # Standard cooling
            self.temperature *= self.temperature_control
    
    def adjust_parameters(self, current_segment, max_segments):
        """
        Adaptively adjust parameters based on search progress
        
        Args:
            current_segment: Current segment number
            max_segments: Maximum number of segments
        """
        if not self.adaptive_parameters:
            return
        
        # Calculate search progress
        progress = current_segment / max_segments
        
        # Adaptive reaction factor
        # Initially high for exploration, then lower for exploitation
        if progress < 0.3:
            # Exploration phase - high reaction
            self.reaction_factor = 0.6
        elif progress < 0.7:
            # Transition phase - moderate reaction
            self.reaction_factor = 0.4
        else:
            # Exploitation phase - low reaction
            self.reaction_factor = 0.2
        
        # Adaptive segment size
        # Smaller at the beginning for faster learning, larger in the middle for stability
        if progress < 0.2:
            self.segment_size = max(50, self.segment_size // 2)
        elif progress < 0.6:
            self.segment_size = min(200, self.segment_size * 2)
        else:
            self.segment_size = max(75, self.segment_size // 1.5)
    
    def evaluate_and_create_candidate(self, destroy_idx, repair_idx):
        """
        Create and evaluate a candidate solution
        
        Args:
            destroy_idx: Index of destroy operator
            repair_idx: Index of repair operator
            
        Returns:
            tuple: (new_solution, new_evaluation, new_objective)
        """
        # Apply destroy operator
        destroyed_solution = self.destroy_operators[destroy_idx](self.problem, self.current_solution.copy())
        
        # Apply repair operator
        new_solution = self.repair_operators[repair_idx](self.problem, destroyed_solution)
        
        # Evaluate new solution
        new_evaluation = self.problem.evaluate_solution(new_solution)
        new_objective = self.calculate_objective(new_evaluation)
        
        return new_solution, new_evaluation, new_objective
    
    def create_sparse_solution_representation(self, solution):
        """
        Convert a dense solution vector to a sparse representation to save memory
        
        Args:
            solution: Original dense solution vector
            
        Returns:
            dict: Sparse representation with non-zero elements only
        """
        # Reshape solution into x_var and u_var
        x_var = solution[:self.problem.x_shape].reshape(self.problem.NUM_DAYS, 
                                                      self.problem.num_transport_types, 
                                                      self.problem.num_locations, 
                                                      self.problem.num_locations)
        u_var = solution[self.problem.x_shape:].reshape(self.problem.NUM_DAYS, 
                                                      self.problem.num_locations)
        
        # Create sparse representation for x_var (only store 1s)
        sparse_x = []
        for day in range(self.problem.NUM_DAYS):
            for j in range(self.problem.num_transport_types):
                for k in range(self.problem.num_locations):
                    for l in range(self.problem.num_locations):
                        if x_var[day, j, k, l] == 1:
                            sparse_x.append((day, j, k, l))
        
        # Create sparse representation for u_var (only store non-zero times)
        sparse_u = {}
        for day in range(self.problem.NUM_DAYS):
            for k in range(self.problem.num_locations):
                if u_var[day, k] > 0:
                    sparse_u[(day, k)] = u_var[day, k]
        
        return {"x": sparse_x, "u": sparse_u}
    
    def expand_sparse_solution(self, sparse_solution):
        """
        Convert a sparse solution back to the original dense format
        
        Args:
            sparse_solution: Sparse solution representation
            
        Returns:
            np.ndarray: Dense solution vector
        """
        # Initialize empty solution
        solution = np.zeros(self.problem.n_var)
        
        # Reshape into x_var and u_var for easier manipulation
        x_var = np.zeros((self.problem.NUM_DAYS, self.problem.num_transport_types, 
                      self.problem.num_locations, self.problem.num_locations))
        u_var = np.zeros((self.problem.NUM_DAYS, self.problem.num_locations))
        
        # Fill in x_var from sparse representation
        for day, j, k, l in sparse_solution["x"]:
            x_var[day, j, k, l] = 1
        
        # Fill in u_var from sparse representation
        for (day, k), time in sparse_solution["u"].items():
            u_var[day, k] = time
        
        # Flatten and combine
        solution[:self.problem.x_shape] = x_var.flatten()
        solution[self.problem.x_shape:] = u_var.flatten()
        
        return solution
    
    def optimize_numpy_arrays(self, solution):
        """
        Optimize memory usage of numpy arrays in the solution
        
        Args:
            solution: Original solution array
            
        Returns:
            np.ndarray: Memory-optimized solution
        """
        # Convert to more memory-efficient data type
        # For binary variables (x_var), use np.int8 instead of default np.int64
        x_part = solution[:self.problem.x_shape].astype(np.int8)
        
        # For continuous variables (u_var), use np.float32 instead of np.float64
        u_part = solution[self.problem.x_shape:].astype(np.float32)
        
        # Combine
        optimized = np.concatenate([x_part, u_part])
        
        return optimized
    
    def limit_history_size(self):
        """
        Limit the size of history data structures to prevent memory growth
        """
        max_history_size = 1000  # Adjust based on memory constraints
        
        # Trim iteration history if it gets too large
        if len(self.iteration_history) > max_history_size:
            # Keep the first few and the most recent entries
            keep_beginning = 100
            keep_end = max_history_size - keep_beginning
            self.iteration_history = (
                self.iteration_history[:keep_beginning] + 
                self.iteration_history[-keep_end:]
            )
        
        # Limit operator performance history
        for i in range(len(self.destroy_performance)):
            if len(self.destroy_performance[i]) > max_history_size:
                self.destroy_performance[i] = self.destroy_performance[i][-max_history_size:]
        
        for i in range(len(self.repair_performance)):
            if len(self.repair_performance[i]) > max_history_size:
                self.repair_performance[i] = self.repair_performance[i][-max_history_size:]
        
        # Also trim objective and temperature history
        max_stat_history = 5000  # Can be larger since these are simple values
        if len(self.objective_history) > max_stat_history:
            sample_interval = len(self.objective_history) // max_stat_history
            self.objective_history = [
                self.objective_history[i] 
                for i in range(0, len(self.objective_history), sample_interval)
            ]
            self.temperature_history = [
                self.temperature_history[i] 
                for i in range(0, len(self.temperature_history), sample_interval)
            ]
    
    def store_diverse_solution(self, solution, evaluation, objective, iteration, max_solutions=10):
        """
        Store a solution in the diverse solutions list with memory efficiency
        
        Args:
            solution: Solution vector
            evaluation: Solution evaluation
            objective: Objective value
            iteration: Current iteration number
            max_solutions: Maximum number of solutions to store
        """
        # Skip if not feasible
        if not evaluation["is_feasible"]:
            return
            
        # Convert to memory-efficient representation
        if self.memory_efficient:
            sparse_solution = self.create_sparse_solution_representation(solution)
        else:
            sparse_solution = solution.copy()
        
        # Store only essential evaluation data
        compact_evaluation = {
            "is_feasible": evaluation["is_feasible"],
            "total_cost": evaluation["total_cost"],
            "total_travel_time": evaluation["total_travel_time"],
            "total_satisfaction": evaluation["total_satisfaction"]
        }
        
        # Add to diverse solutions
        self.diverse_solutions.append({
            "solution": sparse_solution,
            "evaluation": compact_evaluation,
            "objective": objective,
            "iteration": iteration
        })
        
        # Limit the size of the diverse solutions list
        if len(self.diverse_solutions) > max_solutions:
            # Keep solutions sorted by objective
            self.diverse_solutions.sort(key=lambda x: x["objective"])
            
            # Remove worst solutions if we have too many
            if len(self.diverse_solutions) > max_solutions:
                self.diverse_solutions = self.diverse_solutions[:max_solutions]
    
    def filter_diverse_solutions(self, solutions, max_solutions=5):
        """
        Filter solutions to keep only diverse ones
        
        Args:
            solutions: List of solution dictionaries
            max_solutions: Maximum number of solutions to keep
        
        Returns:
            list: Filtered list of diverse solutions
        """
        if not solutions:
            return []
        
        # Always include the best solution
        result = [solutions[0]]
        
        # Define a distance function between solutions
        def solution_distance(sol1, sol2):
            # Calculate distance based on solution representation
            if self.memory_efficient:
                # For sparse representations
                sol1_sparse = sol1["solution"]
                sol2_sparse = sol2["solution"]
                
                # Calculate set difference for x variables
                sol1_set = set([(d, j, k, l) for d, j, k, l in sol1_sparse["x"]])
                sol2_set = set([(d, j, k, l) for d, j, k, l in sol2_sparse["x"]])
                
                diff_count = len(sol1_set.symmetric_difference(sol2_set))
                total_count = len(sol1_set.union(sol2_set))
                
                return diff_count / max(1, total_count)  # Avoid division by zero
            else:
                # For dense representations
                x_var1 = sol1["solution"][:self.problem.x_shape]
                x_var2 = sol2["solution"][:self.problem.x_shape]
                hamming_dist = np.sum(x_var1 != x_var2)
                
                # Normalize by the length
                return hamming_dist / len(x_var1)
        
        # Remaining candidates
        candidates = solutions[1:]
        
        # Add diverse solutions one by one
        while len(result) < max_solutions and candidates:
            # Find the candidate that is most different from all chosen solutions
            best_candidate = None
            best_min_distance = -1
            
            for candidate in candidates:
                # Calculate minimum distance to all chosen solutions
                min_distance = min(solution_distance(candidate, chosen) for chosen in result)
                
                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_candidate = candidate
            
            # Add the most diverse candidate
            if best_candidate and best_min_distance > 0.05:  # Require at least 5% difference
                result.append(best_candidate)
                candidates.remove(best_candidate)
            else:
                # No more sufficiently diverse candidates
                break
        
        return result
    
    def analyze_operator_performance(self):
        """
        Analyze and report on operator performance
        """
        logger.info("Operator Performance Analysis:")
        
        # Analyze destroy operators
        logger.info("Destroy Operators:")
        for i, operator in enumerate(self.destroy_operators):
            operator_name = operator.__name__
            usage_count = sum(self.destroy_count)
            usage_percentage = 0 if usage_count == 0 else self.destroy_count[i] / usage_count * 100
            avg_score = 0 if not self.destroy_performance[i] else sum(self.destroy_performance[i]) / len(self.destroy_performance[i])
            
            logger.info(f"  {operator_name}: Used {self.destroy_count[i]} times ({usage_percentage:.1f}%), Avg Score: {avg_score:.2f}")
        
        # Analyze repair operators
        logger.info("Repair Operators:")
        for i, operator in enumerate(self.repair_operators):
            operator_name = operator.__name__
            usage_count = sum(self.repair_count)
            usage_percentage = 0 if usage_count == 0 else self.repair_count[i] / usage_count * 100
            avg_score = 0 if not self.repair_performance[i] else sum(self.repair_performance[i]) / len(self.repair_performance[i])
            
            logger.info(f"  {operator_name}: Used {self.repair_count[i]} times ({usage_percentage:.1f}%), Avg Score: {avg_score:.2f}")
    
    def estimate_memory_usage(self):
        """
        Estimate current memory usage of key data structures
        
        Returns:
            dict: Memory usage estimates in MB
        """
        try:
            import sys
            
            def get_size_mb(obj):
                # Estimate size of an object in MB
                return sys.getsizeof(obj) / (1024 * 1024)
            
            memory_usage = {
                "current_solution": get_size_mb(self.current_solution),
                "best_solution": get_size_mb(self.best_solution),
                "iteration_history": get_size_mb(self.iteration_history),
                "diverse_solutions": get_size_mb(self.diverse_solutions),
                "objective_history": get_size_mb(self.objective_history),
                "temperature_history": get_size_mb(self.temperature_history)
            }
            
            # Update memory metrics
            total_memory = sum(memory_usage.values())
            self.memory_metrics["current_memory"] = total_memory
            self.memory_metrics["peak_memory"] = max(self.memory_metrics["peak_memory"], total_memory)
            
            return memory_usage
        except:
            # In case of error, return empty dict
            return {}
        
    def analyze_constraint_violations(self, evaluation):
        """
        Analyze constraint violations and provide clear explanations
        
        Args:
            evaluation: Solution evaluation dictionary with constraint violations
            
        Returns:
            dict: Summary of constraint violations with explanations
        """
        if evaluation.get("is_feasible", True):
            return {"is_feasible": True, "violations": []}
        
        # Initialize violation summary
        violation_summary = {
            "is_feasible": False,
            "violations": [],
            "violation_count": 0,
            "primary_issue": None
        }
        
        # Process inequality violations
        if "inequality_violations" in evaluation:
            for violation in evaluation["inequality_violations"]:
                violation_type = violation.get("type", "unknown")
                
                # Create human-readable explanation based on violation type
                explanation = ""
                if violation_type == "start_time_violation":
                    explanation = f"Day {violation.get('day', '?')} starts too early: {violation.get('current_time', 0):.1f} mins vs. required {violation.get('required_time', 0):.1f} mins"
                
                elif violation_type == "insufficient_hawker_visits":
                    explanation = f"Day {violation.get('day', '?')} has insufficient hawker visits: {violation.get('current_visits', 0)} vs. required {violation.get('required_visits', 0)}"
                
                elif violation_type == "no_lunch_hawker_visit":
                    explanation = f"Day {violation.get('day', '?')} is missing a lunch visit to a hawker center"
                
                elif violation_type == "no_dinner_hawker_visit":
                    explanation = f"Day {violation.get('day', '?')} is missing a dinner visit to a hawker center"
                
                elif violation_type == "attraction_max_once_source" or violation_type == "attraction_max_once_dest":
                    explanation = f"Attraction '{violation.get('name', '?')}' is visited multiple times ({violation.get('visits', 0)}) but should be visited at most once"
                
                elif violation_type == "budget_exceeded":
                    explanation = f"Total cost (${violation.get('total_cost', 0):.2f}) exceeds budget limit (${violation.get('budget_limit', 0):.2f})"
                
                elif violation_type == "hotel_not_starting_point":
                    explanation = f"Day {violation.get('day', '?')} does not start from the hotel"
                
                elif violation_type == "not_returning_to_hotel":
                    explanation = f"Day {violation.get('day', '?')} does not end at the hotel (ends at {violation.get('last_location_name', '?')})"
                
                elif violation_type == "multiple_transport_types":
                    explanation = f"Multiple transport types used for the same route from {violation.get('from_name', '?')} to {violation.get('to_name', '?')} on day {violation.get('day', '?')}"
                
                elif violation_type == "insufficient_daily_visits":
                    explanation = f"Day {violation.get('day', '?')} has too few visits: {violation.get('visits', 0)} vs. minimum {violation.get('minimum', 0)}"
                
                elif violation_type == "excessive_daily_visits":
                    explanation = f"Day {violation.get('day', '?')} has too many visits: {violation.get('visits', 0)} vs. maximum {violation.get('maximum', 0)}"
                
                elif violation_type == "sequential_time_violation":
                    explanation = f"Time sequence violation on day {violation.get('day', '?')}: arrival at {violation.get('to_name', '?')} is too early"
                
                elif violation_type == "flow_conservation_violated":
                    explanation = f"Flow conservation violated at {violation.get('location_name', '?')} on day {violation.get('day', '?')}: incoming routes ({violation.get('incoming', 0)}) ≠ outgoing routes ({violation.get('outgoing', 0)})"
                
                elif violation_type == "hotel_start_time_violation":
                    explanation = f"Start time from hotel on day {violation.get('day', '?')} is too early: {violation.get('start_time', 0):.1f} vs. required {violation.get('required_time', 0):.1f}"
                
                elif violation_type == "hotel_end_time_violation":
                    explanation = f"Return time to hotel on day {violation.get('day', '?')} is too late: {violation.get('end_time', 0):.1f} vs. limit {violation.get('limit_time', 0):.1f}"
                
                else:
                    # Generic explanation for unknown violation types
                    explanation = f"Constraint violation of type '{violation_type}'"
                
                # Add to summary
                violation_summary["violations"].append({
                    "type": violation_type,
                    "explanation": explanation,
                    "details": violation
                })
                violation_summary["violation_count"] += 1
        
        # Process equality violations
        if "equality_violations" in evaluation:
            for violation in evaluation["equality_violations"]:
                violation_type = violation.get("type", "unknown")
                
                # Create human-readable explanation based on violation type
                explanation = ""
                if violation_type == "attraction_source_dest_mismatch":
                    explanation = f"Attraction '{violation.get('name', '?')}' has mismatched visits: {violation.get('source_visits', 0)} as source vs. {violation.get('dest_visits', 0)} as destination"
                
                elif violation_type == "hotel_not_starting_point":
                    explanation = f"Day {violation.get('day', '?')} must start from hotel but doesn't"
                
                else:
                    # Generic explanation for unknown violation types
                    explanation = f"Constraint violation of type '{violation_type}'"
                
                # Add to summary
                violation_summary["violations"].append({
                    "type": violation_type,
                    "explanation": explanation,
                    "details": violation
                })
                violation_summary["violation_count"] += 1
        
        # Identify primary issue
        if violation_summary["violations"]:
            # Sort violations by criticality
            critical_types = [
                "budget_exceeded", 
                "no_lunch_hawker_visit", 
                "no_dinner_hawker_visit",
                "hotel_not_starting_point",
                "not_returning_to_hotel"
            ]
            
            # Find the most critical violation
            for critical_type in critical_types:
                critical_violations = [v for v in violation_summary["violations"] if v["type"] == critical_type]
                if critical_violations:
                    violation_summary["primary_issue"] = critical_violations[0]["explanation"]
                    break
            
            # If no critical violations found, use the first one
            if not violation_summary["primary_issue"] and violation_summary["violations"]:
                violation_summary["primary_issue"] = violation_summary["violations"][0]["explanation"]
        
        return violation_summary
    
    def run(self, verbose=True, track_solutions=False):
        """
        Run the enhanced ALNS algorithm
        
        Args:
            verbose: Whether to print progress information
            track_solutions: Whether to track all accepted solutions for diversity
            
        Returns:
            dict: Results including best solution and statistics
        """
        logger.info("Starting enhanced ALNS optimization...")
        
        start_time = time.time()
        iteration = 0
        segment_iteration = 0
        
        # Score constants for updating weights
        SCORE_BEST = 3     # New best solution
        SCORE_BETTER = 2   # Better than current but not best
        SCORE_ACCEPTED = 1 # Worse but accepted
        SCORE_REJECTED = 0 # Rejected
        
        # Track statistics
        stats = {
            "iterations": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "best_objective": self.best_objective,
            "best_cost": self.best_evaluation["total_cost"],
            "best_travel_time": self.best_evaluation["total_travel_time"],
            "best_satisfaction": self.best_evaluation["total_satisfaction"],
            "runtime": 0,
            "best_found_at": 0,
            "initial_objective": self.current_objective,
            "objective_history": [self.current_objective],
            "temperature_history": [self.temperature],
            "best_is_feasible": self.best_evaluation["is_feasible"],
            "early_termination": False,
            "segment_sizes": [self.segment_size],
            "reaction_factors": [self.reaction_factor],
            "adaptive_params": self.adaptive_parameters,
            "memory_efficient": self.memory_efficient
        }
        
        # Calculate max segments
        max_segments = self.max_iterations // self.segment_size + 1
        current_segment = 0
        
        # Main ALNS loop
        while True:
            # Check termination conditions
            if iteration >= self.max_iterations:
                logger.info("Terminating: Maximum iterations reached")
                break
                
            if self.time_limit and time.time() - start_time > self.time_limit:
                logger.info("Terminating: Time limit reached")
                stats["timeout"] = True
                break
            
            if self.iterations_since_improvement >= self.early_termination_iterations:
                logger.info(f"Terminating: No improvement for {self.iterations_since_improvement} iterations")
                stats["early_termination"] = True
                break
            
            # Select and apply operators
            if self.parallel_evaluations:
                # Parallel generation and evaluation of candidates
                candidates = []
                futures = []
                
                # Create multiple candidates in parallel
                num_candidates = min(self.max_workers, 4)  # Generate up to 4 candidates
                for _ in range(num_candidates):
                    destroy_idx = self.select_destroy_operator()
                    repair_idx = self.select_repair_operator()
                    candidates.append((destroy_idx, repair_idx))
                    
                    # Submit task to executor
                    future = self.executor.submit(
                        self.evaluate_and_create_candidate, destroy_idx, repair_idx
                    )
                    futures.append(future)
                
                # Wait for all candidates to complete
                concurrent.futures.wait(futures)
                
                # Process results
                best_candidate_idx = -1
                best_candidate_objective = float('inf')
                
                for i, future in enumerate(futures):
                    new_solution, new_evaluation, new_objective = future.result()
                    
                    # Check if this is the best candidate
                    if new_objective < best_candidate_objective:
                        best_candidate_idx = i
                        best_candidate_objective = new_objective
                
                # Use the best candidate
                destroy_idx, repair_idx = candidates[best_candidate_idx]
                new_solution, new_evaluation, new_objective = futures[best_candidate_idx].result()
                
                # Update counters for all explored operators
                for destroy_idx, repair_idx in candidates:
                    self.destroy_count[destroy_idx] += 1
                    self.repair_count[repair_idx] += 1
            else:
                # Sequential evaluation (original approach)
                destroy_idx = self.select_destroy_operator()
                repair_idx = self.select_repair_operator()
                
                # Update counters
                self.destroy_count[destroy_idx] += 1
                self.repair_count[repair_idx] += 1
                
                # Generate and evaluate candidate
                new_solution, new_evaluation, new_objective = self.evaluate_and_create_candidate(
                    destroy_idx, repair_idx
                )
            
            # Apply memory optimization if enabled
            if self.memory_efficient:
                new_solution = self.optimize_numpy_arrays(new_solution)
            
            # Decide whether to accept the new solution
            if self.accept_solution(new_objective):
                # Update current solution
                self.current_solution = new_solution
                self.current_evaluation = new_evaluation
                self.current_objective = new_objective
                
                # Update best solution if improvement
                if new_objective < self.best_objective:
                    if verbose:
                        logger.info(f"Iteration {iteration}: New best solution found (objective: {new_objective:.4f}, feasible: {new_evaluation['is_feasible']})")
                    
                    # Store best solution
                    self.best_solution = new_solution.copy()
                    if self.memory_efficient:
                        self.best_solution_sparse = self.create_sparse_solution_representation(new_solution)
                    self.best_evaluation = new_evaluation
                    self.best_objective = new_objective
                    
                    # Update weight scores - new best solution
                    self.scores_destroy[destroy_idx] += SCORE_BEST
                    self.scores_repair[repair_idx] += SCORE_BEST
                    
                    # Update statistics
                    stats["best_objective"] = self.best_objective
                    stats["best_cost"] = self.best_evaluation["total_cost"]
                    stats["best_travel_time"] = self.best_evaluation["total_travel_time"]
                    stats["best_satisfaction"] = self.best_evaluation["total_satisfaction"]
                    stats["best_found_at"] = iteration
                    stats["best_is_feasible"] = self.best_evaluation["is_feasible"]
                    
                    # Reset counter for early termination
                    self.iterations_since_improvement = 0
                    
                    # Store diverse solution if tracking enabled
                    if track_solutions:
                        self.store_diverse_solution(
                            new_solution, new_evaluation, new_objective, iteration
                        )
                else:
                    # Update weight scores - better than current but not best
                    self.scores_destroy[destroy_idx] += SCORE_BETTER
                    self.scores_repair[repair_idx] += SCORE_BETTER
                    
                    # Store slightly worse solutions with some probability for diversity
                    if track_solutions and random.random() < 0.1 and new_evaluation["is_feasible"]:
                        # Only keep diverse solutions that are feasible
                        self.store_diverse_solution(
                            new_solution, new_evaluation, new_objective, iteration, max_solutions=20
                        )
                
                stats["accepted_count"] += 1
                
                # Increment counter for early termination
                self.iterations_since_improvement += 1
            else:
                # Update weight scores - solution rejected
                self.scores_destroy[destroy_idx] += SCORE_REJECTED
                self.scores_repair[repair_idx] += SCORE_REJECTED
                
                stats["rejected_count"] += 1
                
                # Increment counter for early termination
                self.iterations_since_improvement += 1
            
            # Record iteration history
            self.iteration_history.append({
                "iteration": iteration,
                "objective": self.current_objective,
                "temperature": self.temperature,
                "destroy_operator": destroy_idx,
                "repair_operator": repair_idx,
                "accepted": self.current_objective == new_objective
            })
            
            self.objective_history.append(self.current_objective)
            self.temperature_history.append(self.temperature)
            
            # Update segment iterations
            segment_iteration += 1
            
            # Check if we've completed a segment
            if segment_iteration >= self.segment_size:
                # Update operator weights
                self.update_weights()
                
                # Update simulated annealing temperature
                self.update_temperature(self.max_iterations - iteration, self.max_iterations)
                
                # Adjust parameters if adaptive mode is enabled
                current_segment += 1
                self.adjust_parameters(current_segment, max_segments)
                
                # Track parameter changes for analysis
                stats["segment_sizes"].append(self.segment_size)
                stats["reaction_factors"].append(self.reaction_factor)
                
                # Reset segment iteration counter
                segment_iteration = 0
                
                # Memory management
                if self.memory_efficient:
                    self.limit_history_size()
                    # Force garbage collection
                    gc.collect()
                    
                    # Estimate memory usage
                    memory_usage = self.estimate_memory_usage()
                    stats["memory_usage"] = memory_usage
                
                if verbose:
                    # Log progress at end of segment
                    elapsed = time.time() - start_time
                    logger.info(f"Segment {current_segment}/{max_segments} complete - Iteration {iteration}, "
                               f"Temp: {self.temperature:.4f}, Current: {self.current_objective:.4f}, "
                               f"Best: {self.best_objective:.4f}, Time: {elapsed:.1f}s")
                    
                    # Provide ETA if time limit is specified
                    if self.time_limit:
                        remaining_time = self.time_limit - elapsed
                        if remaining_time > 0:
                            estimated_completion = elapsed * self.max_iterations / iteration
                            logger.info(f"ETA: {estimated_completion:.1f}s, Time remaining: {remaining_time:.1f}s")
            
            # Increment iteration counter
            iteration += 1
        
        # Calculate final statistics
        stats["iterations"] = iteration
        stats["runtime"] = time.time() - start_time
        stats["iterations_per_second"] = iteration / stats["runtime"] if stats["runtime"] > 0 else 0
        
        # Process diverse solutions if tracking was enabled
        if track_solutions:
            # Process and prepare diverse solutions for return
            filtered_solutions = self.filter_diverse_solutions(self.diverse_solutions, max_solutions=5)
            
            # Expand sparse solutions if needed
            processed_solutions = []
            for sol_data in filtered_solutions:
                solution_copy = sol_data.copy()
                
                # Expand sparse solution if using memory efficiency
                if self.memory_efficient and isinstance(sol_data["solution"], dict):
                    solution_copy["solution"] = self.expand_sparse_solution(sol_data["solution"])
                
                processed_solutions.append(solution_copy)
            
            stats["diverse_solutions"] = processed_solutions
        
        # Final memory cleanup
        if self.memory_efficient:
            # Clean up memory before returning
            gc.collect()
            
            # Final memory usage
            memory_usage = self.estimate_memory_usage()
            stats["final_memory_usage"] = memory_usage
            stats["peak_memory_usage"] = self.memory_metrics["peak_memory"]
        
        # Log final results
        logger.info(f"ALNS optimization completed in {stats['runtime']:.2f} seconds ({stats['iterations_per_second']:.1f} iterations/second)")
        logger.info(f"Best objective: {self.best_objective:.4f} (iteration {stats['best_found_at']})")
        logger.info(f"Best solution cost: ${self.best_evaluation['total_cost']:.2f}")
        logger.info(f"Best solution travel time: {self.best_evaluation['total_travel_time']:.2f} minutes")
        logger.info(f"Best solution satisfaction: {self.best_evaluation['total_satisfaction']:.2f}")
        logger.info(f"Best solution feasible: {self.best_evaluation['is_feasible']}")

        # Analyze constraint violations if the best solution is not feasible
        if not self.best_evaluation.get("is_feasible", True):
            try:
                violation_summary = self.analyze_constraint_violations(self.best_evaluation)
                stats["violation_summary"] = violation_summary
                
                if verbose:
                    logger.warning(f"Best solution is not feasible. Primary issue: {violation_summary.get('primary_issue', 'Unknown')}")
                    logger.warning(f"Total violations: {violation_summary.get('violation_count', 0)}")
                    for i, violation in enumerate(violation_summary.get('violations', [])[:5]):  # Show top 5 violations
                        logger.warning(f"  {i+1}. {violation.get('explanation', 'Unknown violation')}")
                    if len(violation_summary.get('violations', [])) > 5:
                        logger.warning(f"  ... plus {len(violation_summary.get('violations', [])) - 5} more violations")
            except Exception as e:
                logger.error(f"Error analyzing constraint violations: {e}")
                stats["violation_analysis_error"] = str(e)

        # Analyze operator performance if verbose
        if verbose:
            self.analyze_operator_performance()
        
        # Prepare the best solution for return
        # If using memory efficiency, expand the sparse representation
        if self.memory_efficient and hasattr(self, 'best_solution_sparse'):
            self.best_solution = self.expand_sparse_solution(self.best_solution_sparse)
        
        # Return best solution and statistics
        return {
            "best_solution": self.best_solution,
            "best_evaluation": self.best_evaluation,
            "stats": stats
        }