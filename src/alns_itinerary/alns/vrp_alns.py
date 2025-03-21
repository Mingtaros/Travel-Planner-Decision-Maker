"""
Enhanced Adaptive Large Neighborhood Search (ALNS) algorithm for the VRP-based travel itinerary problem.
This implementation uses the position-based VRP representation for more effective time constraint management.
"""

import random
import math
import time
import logging
import numpy as np
from datetime import datetime
import copy
import gc
from collections import defaultdict

from alns.vrp_solution import VRPSolution
from alns.vrp_operators import *

logger = logging.getLogger(__name__)

class VRPALNS:
    """
    Enhanced ALNS implementation using the position-based VRP representation
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
        early_termination_iterations=200
    ):
        """
        Initialize the ALNS algorithm with the VRP approach
        
        Args:
            problem: The TravelItineraryProblem instance
            initial_solution: Initial VRPSolution (if None, will create a heuristic solution)
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
        """
        self.problem = problem
        self.max_iterations = max_iterations
        self.segment_size = segment_size
        self.reaction_factor = reaction_factor
        self.decay_factor = decay_factor
        self.temperature_control = temperature_control
        self.time_limit = time_limit
        self.early_termination_iterations = early_termination_iterations
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize destroy operators
        if destroy_operators is None:
            self.destroy_operators = [
                destroy_random_day_subsequence,
                destroy_worst_attractions,
                destroy_random_attractions,
                destroy_random_meals,
                destroy_time_window_violations,
                destroy_day_shuffle
            ]
        else:
            self.destroy_operators = destroy_operators
        
        # Initialize repair operators
        if repair_operators is None:
            self.repair_operators = [
                repair_greedy_insertion,
                repair_regret_insertion,
                repair_time_based_insertion
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
            # Use the improved method for creating a valid initial solution
            self.current_solution = self.create_initial_solution()
            
            # If not feasible, log a warning
            if not self.current_solution.is_feasible():
                logger.warning("Initial solution is not feasible")
        
        # Evaluate current solution
        self.current_evaluation = self.current_solution.evaluate()
        self.current_objective = self.calculate_objective(self.current_evaluation)
        
        # Initialize best solution
        self.best_solution = self.current_solution.clone()
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
    
    def create_initial_solution(self):
        """
        Create a valid initial solution that adheres to all constraints:
        - Each day has exactly one lunch and one dinner at appropriate times
        - Each day starts and ends at the hotel
        - Attractions are distributed sensibly between meals
        
        Returns:
            VRPSolution: Valid initial solution
        """
        solution = VRPSolution(self.problem)
        
        # Get attraction and hawker lists
        attractions = [i for i in range(self.problem.num_locations) 
                    if self.problem.locations[i]["type"] == "attraction"]
        hawkers = [i for i in range(self.problem.num_locations) 
                if self.problem.locations[i]["type"] == "hawker"]
        
        # Rank attractions by value
        attraction_values = []
        for attr_idx in attractions:
            satisfaction = self.problem.locations[attr_idx].get("satisfaction", 0)
            cost = self.problem.locations[attr_idx].get("entrance_fee", 1)
            duration = self.problem.locations[attr_idx].get("duration", 60)
            
            # Calculate value ratio (higher is better)
            value_ratio = satisfaction / (cost + duration/60)
            attraction_values.append((attr_idx, value_ratio))
        
        # Sort by value ratio (highest first)
        attraction_values.sort(key=lambda x: x[1], reverse=True)
        
        # Rank hawkers by rating
        hawker_ratings = [(h, self.problem.locations[h].get("rating", 0)) for h in hawkers]
        hawker_ratings.sort(key=lambda x: x[1], reverse=True)  # Sort by rating (highest first)
        
        # Make sure we have enough hawkers for each day's lunch and dinner
        if len(hawker_ratings) < self.problem.NUM_DAYS * 2:
            # If not enough unique hawkers, allow reuse across days
            hawker_ratings = hawker_ratings * (1 + (self.problem.NUM_DAYS * 2 // len(hawker_ratings)))
        
        # Initialize each day with proper structure
        for day in range(solution.num_days):
            # Skip if we don't have enough hawkers
            if len(hawker_ratings) < 2:
                logger.warning("Not enough unique hawkers for each day's lunch and dinner")
                continue
                
            # 1. Create lunch at appropriate time
            lunch_time = self.problem.LUNCH_START + 60  # 12:00 PM
            lunch_hawker_idx, _ = hawker_ratings.pop(0)  # Get highest rated hawker for lunch
            
            # Find transport mode that works
            lunch_inserted = False
            for transport_mode in ["transit", "drive"]:
                if solution.insert_location(day, 1, lunch_hawker_idx, transport_mode):
                    lunch_inserted = True
                    break
            
            if not lunch_inserted:
                # Try another hawker if insertion failed
                if hawker_ratings:
                    lunch_hawker_idx, _ = hawker_ratings.pop(0)
                    for transport_mode in ["transit", "drive"]:
                        if solution.insert_location(day, 1, lunch_hawker_idx, transport_mode):
                            lunch_inserted = True
                            break
            
            # 2. Create dinner at appropriate time
            dinner_time = self.problem.DINNER_START + 60  # 6:00 PM
            
            # For dinner timing, we need to calculate appropriate position after lunch
            dinner_position = 2  # After lunch by default
            
            if lunch_inserted:
                # Aim for dinner time
                dinner_hawker_idx, _ = hawker_ratings.pop(0)  # Get highest rated hawker for dinner
                
                # Find transport mode that works
                dinner_inserted = False
                for transport_mode in ["transit", "drive"]:
                    if solution.insert_location(day, dinner_position, dinner_hawker_idx, transport_mode):
                        dinner_inserted = True
                        break
                
                if not dinner_inserted:
                    # Try another hawker if insertion failed
                    if hawker_ratings:
                        dinner_hawker_idx, _ = hawker_ratings.pop(0)
                        for transport_mode in ["transit", "drive"]:
                            if solution.insert_location(day, dinner_position, dinner_hawker_idx, transport_mode):
                                dinner_inserted = True
                                break
            
            # 3. Add hotel return at the end
            hotel_idx = 0
            hotel_position = dinner_position + 1 if dinner_inserted else dinner_position
            
            # Find transport mode that works
            hotel_return_inserted = False
            for transport_mode in ["transit", "drive"]:
                if solution.insert_location(day, hotel_position, hotel_idx, transport_mode):
                    hotel_return_inserted = True
                    break
            
            # 4. Try to insert attractions between lunch and dinner
            if lunch_inserted and dinner_inserted:
                attr_position = 2  # After lunch, before dinner
                
                # Try to add 1-2 attractions between lunch and dinner
                added_attractions = 0
                max_attractions = 2
                
                for attr_idx, _ in attraction_values[:]:
                    if added_attractions >= max_attractions:
                        break
                    
                    # Try both transit and drive
                    for transport_mode in ["transit", "drive"]:
                        if solution.is_feasible_insertion(day, attr_position, attr_idx, transport_mode):
                            if solution.insert_location(day, attr_position, attr_idx, transport_mode):
                                # Update attraction position for next insertion
                                attr_position += 1
                                added_attractions += 1
                                
                                # Remove this attraction from consideration for other days
                                attraction_values = [(a, v) for a, v in attraction_values if a != attr_idx]
                                break
        
        # Post-process to fix any issues
        solution = solution.post_process_solution()
        
        return solution
    
    def calibrate_temperature(self, num_samples=20, acceptance_rate=0.5):
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
        solution_copy = self.current_solution.clone()
        
        for _ in range(num_samples):
            # Apply random destroy operator
            destroy_idx = random.randrange(len(self.destroy_operators))
            destroyed_solution = self.destroy_operators[destroy_idx](self.problem, solution_copy.clone())
            
            # Apply random repair operator
            repair_idx = random.randrange(len(self.repair_operators))
            repaired_solution = self.repair_operators[repair_idx](self.problem, destroyed_solution)
            
            # Evaluate new solution
            new_evaluation = repaired_solution.evaluate()
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
            objective += 10.0
        
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
                # Store performance metrics
                self.destroy_performance[i].append(self.scores_destroy[i] / self.destroy_count[i])
        
        for i in range(len(self.repair_operators)):
            if self.repair_count[i] > 0:
                self.weights_repair[i] += self.reaction_factor * (self.scores_repair[i] / self.repair_count[i])
                # Store performance metrics
                self.repair_performance[i].append(self.scores_repair[i] / self.repair_count[i])
        
        # Reset scores and counts for the next segment
        self.scores_destroy = [0] * len(self.destroy_operators)
        self.scores_repair = [0] * len(self.repair_operators)
        self.destroy_count = [0] * len(self.destroy_operators)
        self.repair_count = [0] * len(self.repair_operators)
    
    def update_temperature(self):
        """
        Update the temperature for simulated annealing
        """
        self.temperature *= self.temperature_control
    
    def run(self, verbose=True):
        """
        Run the VRP-based ALNS algorithm
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            dict: Results including best solution and statistics
        """
        logger.info("Starting VRP-based ALNS optimization...")
        
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
            "early_termination": False
        }
        
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
            destroy_idx = self.select_destroy_operator()
            repair_idx = self.select_repair_operator()
            
            # Update counters
            self.destroy_count[destroy_idx] += 1
            self.repair_count[repair_idx] += 1
            
            # Apply destroy operator
            destroyed_solution = self.destroy_operators[destroy_idx](self.problem, self.current_solution.clone())
            
            # Apply repair operator
            new_solution = self.repair_operators[repair_idx](self.problem, destroyed_solution)
            
            # Evaluate new solution
            new_evaluation = new_solution.evaluate()
            new_objective = self.calculate_objective(new_evaluation)
            
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
                    self.best_solution = new_solution.clone()
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
                    
                    # Store solution for diversity
                    self.diverse_solutions.append({
                        "solution": new_solution.clone(),
                        "evaluation": new_evaluation,
                        "objective": new_objective,
                        "iteration": iteration
                    })
                    # Limit the size of diverse solutions list
                    if len(self.diverse_solutions) > 10:
                        self.diverse_solutions.sort(key=lambda x: x["objective"])
                        self.diverse_solutions = self.diverse_solutions[:10]
                else:
                    # Update weight scores - better than current but not best
                    self.scores_destroy[destroy_idx] += SCORE_BETTER
                    self.scores_repair[repair_idx] += SCORE_BETTER
                
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
                self.update_temperature()
                
                # Reset segment iteration counter
                segment_iteration = 0
                
                # Force garbage collection
                gc.collect()
                
                if verbose:
                    # Log progress at end of segment
                    elapsed = time.time() - start_time
                    logger.info(f"Segment {iteration//self.segment_size + 1} complete - Iteration {iteration}, "
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
        
        # Log final results
        logger.info(f"VRP-based ALNS optimization completed in {stats['runtime']:.2f} seconds ({stats['iterations_per_second']:.1f} iterations/second)")
        logger.info(f"Best objective: {self.best_objective:.4f} (iteration {stats['best_found_at']})")
        logger.info(f"Best solution cost: ${self.best_evaluation['total_cost']:.2f}")
        logger.info(f"Best solution travel time: {self.best_evaluation['total_travel_time']:.2f} minutes")
        logger.info(f"Best solution satisfaction: {self.best_evaluation['total_satisfaction']:.2f}")
        logger.info(f"Best solution feasible: {self.best_evaluation['is_feasible']}")
        
        # Analyze operator performance if verbose
        if verbose:
            self.analyze_operator_performance()
        
        # Post-process the best solution before returning it
        logger.info("Post-processing best solution to fix any remaining issues...")
        post_processed_solution = self.best_solution.post_process_solution()
        
        # Re-evaluate the post-processed solution
        post_processed_evaluation = post_processed_solution.evaluate()
        post_processed_objective = self.calculate_objective(post_processed_evaluation)
        
        # Check if the post-processed solution is better
        if post_processed_objective <= self.best_objective and post_processed_evaluation["is_feasible"]:
            logger.info("Post-processed solution is better or equal to the original best solution.")
            self.best_solution = post_processed_solution
            self.best_evaluation = post_processed_evaluation
            self.best_objective = post_processed_objective
        elif post_processed_evaluation["is_feasible"] and not self.best_evaluation["is_feasible"]:
            logger.info("Post-processed solution is feasible while original wasn't. Using post-processed solution.")
            self.best_solution = post_processed_solution
            self.best_evaluation = post_processed_evaluation
            self.best_objective = post_processed_objective
        else:
            logger.info("Original solution is better than post-processed. Keeping original solution.")
        
        # Return best solution and statistics
        return {
            "best_solution": self.best_solution,
            "best_evaluation": self.best_evaluation,
            "stats": stats
        }
    
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