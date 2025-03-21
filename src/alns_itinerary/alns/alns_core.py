import random
import math
import numpy as np
import logging
import os
import time
from collections import defaultdict

logger = logging.getLogger("alns")

class ALNS:
    """
    Adaptive Large Neighborhood Search implementation for the Travel Itinerary Problem
    """
    def __init__(self, problem, initial_solution=None, 
                 destroy_operators=None, repair_operators=None,
                 weights_destroy=None, weights_repair=None,
                 max_iterations=1000, segment_size=100, 
                 reaction_factor=0.5, decay_factor=0.8,
                 temperature_control=0.95, initial_temperature=None,
                 time_limit=None, seed=None):
        """
        Initialize the ALNS algorithm
        
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
        """
        self.problem = problem
        self.max_iterations = max_iterations
        self.segment_size = segment_size
        self.reaction_factor = reaction_factor
        self.decay_factor = decay_factor
        self.temperature_control = temperature_control
        self.time_limit = time_limit
        
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
                destroy_random_routes
            )
            self.destroy_operators = [
                destroy_random_days,
                destroy_random_attractions,
                destroy_worst_attractions,
                destroy_random_meals,
                destroy_random_routes
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
                repair_time_based
            )
            self.repair_operators = [
                repair_greedy,
                repair_random,
                repair_regret,
                repair_satisfaction_based,
                repair_time_based
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
        
        # Initialize current solution
        if initial_solution is not None:
            self.current_solution = initial_solution
        else:
            self.current_solution = self.create_initial_solution()
        
        # Evaluate current solution
        self.current_evaluation = self.problem.evaluate_solution(self.current_solution)
        self.current_objective = self.calculate_objective(self.current_evaluation)
        
        # Initialize best solution
        self.best_solution = self.current_solution.copy()
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
    
    def create_initial_solution(self):
        """
        Create an initial solution using a heuristic approach
        
        Returns:
            np.ndarray: Initial solution vector
        """
        logger.info("Creating initial heuristic solution...")
        
        # Initialize solution vectors
        solution = np.zeros(self.problem.n_var, dtype=int)
        x_var = np.zeros((self.problem.NUM_DAYS, self.problem.num_transport_types, 
                         self.problem.num_locations, self.problem.num_locations), dtype=int)
        u_var = np.zeros((self.problem.NUM_DAYS, self.problem.num_locations), dtype=float)
        
        # Set the hotel as always starting from 9 AM
        hotel_index = 0  # Assuming hotel is at index 0
        
        # Track attractions already visited
        attractions_visited = set()
        
        # List of available hawkers
        hawkers_available = [i for i in range(self.problem.num_locations) 
                            if i != hotel_index and self.problem.locations[i]["type"] == "hawker"]
        
        # Get all attraction indices
        attractions = [i for i in range(self.problem.num_locations) 
                      if self.problem.locations[i]["type"] == "attraction"]
        
        # Sort attractions by satisfaction to cost+time ratio (greedy heuristic)
        attraction_values = []
        for attr_idx in attractions:
            attr = self.problem.locations[attr_idx]
            satisfaction = attr.get("satisfaction", 0)
            cost = attr.get("entrance_fee", 0)
            duration = attr.get("duration", 60)
            
            # Avoid division by zero
            if cost == 0:
                cost = 1
            
            # Higher ratio means better value
            value_ratio = satisfaction / (cost + duration/10)
            attraction_values.append((attr_idx, value_ratio))
        
        # Sort attractions by value (highest to lowest)
        attraction_values.sort(key=lambda x: x[1], reverse=True)
        
        # Track total cost
        total_cost = self.problem.NUM_DAYS * self.problem.HOTEL_COST
        
        # For each day
        for day in range(self.problem.NUM_DAYS):
            # Start at hotel
            current_location = hotel_index
            current_time = self.problem.START_TIME
            
            # Set the initial time for hotel
            u_var[day, hotel_index] = current_time
            
            # 1. Go to lunch hawker
            available_lunch_hawkers = hawkers_available.copy()
            np.random.shuffle(available_lunch_hawkers)
            
            if available_lunch_hawkers:
                lunch_hawker = available_lunch_hawkers[0]
                
                # Choose between transit and driving
                transport_choice = 0  # Default to transit (index 0)
                transport_hour = self.problem.get_transport_hour(current_time)
                
                try:
                    # Get transport data
                    transit_data = self.problem.transport_matrix[
                        (self.problem.locations[current_location]["name"], 
                         self.problem.locations[lunch_hawker]["name"], 
                         transport_hour)]["transit"]
                    
                    drive_data = self.problem.transport_matrix[
                        (self.problem.locations[current_location]["name"], 
                         self.problem.locations[lunch_hawker]["name"], 
                         transport_hour)]["drive"]
                    
                    # Choose driving if it saves significant time and budget allows
                    if (drive_data["duration"] < transit_data["duration"] * 0.7 and 
                        total_cost + drive_data["price"] <= self.problem.budget):
                        transport_choice = 1  # Drive (index 1)
                
                    # Calculate arrival time
                    if transport_choice == 0:
                        transit_time = transit_data["duration"]
                        transit_cost = transit_data["price"]
                        current_time += transit_time
                        total_cost += transit_cost
                    else:
                        drive_time = drive_data["duration"]
                        drive_cost = drive_data["price"]
                        current_time += drive_time
                        total_cost += drive_cost
                    
                    # Set route in solution
                    x_var[day, transport_choice, current_location, lunch_hawker] = 1
                    
                    # Add meal cost
                    lunch_cost = 10  # Assumed fixed cost
                    total_cost += lunch_cost
                    
                    # Set finish time at hawker (include duration)
                    lunch_duration = self.problem.locations[lunch_hawker]["duration"]
                    current_time += lunch_duration
                    
                    # Ensure lunch time is within the lunch window
                    if current_time < self.problem.LUNCH_START:
                        current_time = self.problem.LUNCH_START + lunch_duration
                    
                    u_var[day, lunch_hawker] = current_time
                    current_location = lunch_hawker
                except KeyError:
                    # Missing transport data, skip this lunch hawker
                    pass
            
            # 2. Visit attractions based on value
            todays_attractions = []
            
            # Calculate max time available before dinner
            max_time_before_dinner = self.problem.DINNER_START - current_time
            
            # Add attractions while time permits
            for attr_idx, _ in attraction_values:
                # Skip if already visited
                if attr_idx in attractions_visited:
                    continue
                
                try:
                    # Check if we still have time before dinner
                    transport_hour = self.problem.get_transport_hour(current_time)
                    
                    # Get transit data
                    transit_data = self.problem.transport_matrix[
                        (self.problem.locations[current_location]["name"], 
                         self.problem.locations[attr_idx]["name"], 
                         transport_hour)]["transit"]
                    
                    drive_data = self.problem.transport_matrix[
                        (self.problem.locations[current_location]["name"], 
                         self.problem.locations[attr_idx]["name"], 
                         transport_hour)]["drive"]
                    
                    # Choose transport method
                    transport_choice = 0
                    transport_time = transit_data["duration"]
                    transport_cost = transit_data["price"]
                    
                    # If driving saves a lot of time and budget allows, choose driving
                    if (drive_data["duration"] < transit_data["duration"] * 0.7 and 
                        total_cost + drive_data["price"] <= self.problem.budget):
                        transport_choice = 1
                        transport_time = drive_data["duration"]
                        transport_cost = drive_data["price"]
                    
                    # Calculate time needed
                    attraction_duration = self.problem.locations[attr_idx]["duration"]
                    total_time_needed = transport_time + attraction_duration
                    
                    # If there's not enough time before dinner, skip this attraction
                    if total_time_needed > max_time_before_dinner:
                        continue
                    
                    # Check if adding this attraction would exceed budget
                    attraction_cost = self.problem.locations[attr_idx]["entrance_fee"]
                    if total_cost + transport_cost + attraction_cost > self.problem.budget:
                        continue
                    
                    # Add this attraction to today's itinerary
                    todays_attractions.append(attr_idx)
                    
                    # Set route in solution
                    x_var[day, transport_choice, current_location, attr_idx] = 1
                    
                    # Update time and cost
                    current_time += transport_time + attraction_duration
                    total_cost += transport_cost + attraction_cost
                    
                    # Update current location
                    u_var[day, attr_idx] = current_time
                    current_location = attr_idx
                    
                    # Update remaining time
                    max_time_before_dinner = self.problem.DINNER_START - current_time
                    
                    # Mark as visited
                    attractions_visited.add(attr_idx)
                    
                    # Limit to 2-3 attractions per day
                    if len(todays_attractions) >= 2:
                        break
                except KeyError:
                    # Missing transport data, skip this attraction
                    continue
            
            # 3. Go to dinner hawker
            dinner_hawkers = [h for h in hawkers_available if h != lunch_hawker]
            np.random.shuffle(dinner_hawkers)
            
            if dinner_hawkers:
                dinner_hawker = dinner_hawkers[0]
                
                try:
                    # Choose transport method
                    transport_hour = self.problem.get_transport_hour(current_time)
                    
                    transit_data = self.problem.transport_matrix[
                        (self.problem.locations[current_location]["name"], 
                         self.problem.locations[dinner_hawker]["name"], 
                         transport_hour)]["transit"]
                    
                    drive_data = self.problem.transport_matrix[
                        (self.problem.locations[current_location]["name"], 
                         self.problem.locations[dinner_hawker]["name"], 
                         transport_hour)]["drive"]
                    
                    transport_choice = 0
                    transport_time = transit_data["duration"]
                    transport_cost = transit_data["price"]
                    
                    # If driving saves time and budget allows, choose driving
                    if (drive_data["duration"] < transit_data["duration"] * 0.7 and 
                        total_cost + drive_data["price"] <= self.problem.budget):
                        transport_choice = 1
                        transport_time = drive_data["duration"]
                        transport_cost = drive_data["price"]
                    
                    # Calculate arrival time
                    current_time += transport_time
                    total_cost += transport_cost
                    
                    # Ensure dinner time is within dinner window
                    if current_time < self.problem.DINNER_START:
                        current_time = self.problem.DINNER_START
                    elif current_time > self.problem.DINNER_END:
                        # If we're past dinner window, adjust
                        current_time = self.problem.DINNER_END - self.problem.locations[dinner_hawker]["duration"]
                    
                    # Set route in solution
                    x_var[day, transport_choice, current_location, dinner_hawker] = 1
                    
                    # Add meal cost
                    dinner_cost = 10  # Assumed fixed cost
                    total_cost += dinner_cost
                    
                    # Set finish time at hawker
                    dinner_duration = self.problem.locations[dinner_hawker]["duration"]
                    current_time += dinner_duration
                    u_var[day, dinner_hawker] = current_time
                    
                    # Update location
                    current_location = dinner_hawker
                except KeyError:
                    # Missing transport data, skip this dinner hawker
                    pass
            
            # 4. Return to hotel
            try:
                transport_hour = self.problem.get_transport_hour(current_time)
                
                transit_data = self.problem.transport_matrix[
                    (self.problem.locations[current_location]["name"], 
                     self.problem.locations[hotel_index]["name"], 
                     transport_hour)]["transit"]
                
                drive_data = self.problem.transport_matrix[
                    (self.problem.locations[current_location]["name"], 
                     self.problem.locations[hotel_index]["name"], 
                     transport_hour)]["drive"]
                
                # Choose fastest transport method to get back
                if drive_data["duration"] < transit_data["duration"]:
                    transport_choice = 1
                    return_time = drive_data["duration"]
                    return_cost = drive_data["price"]
                else:
                    transport_choice = 0
                    return_time = transit_data["duration"]
                    return_cost = transit_data["price"]
                
                # Set route back to hotel
                x_var[day, transport_choice, current_location, hotel_index] = 1
                
                # Update time and cost
                current_time += return_time
                total_cost += return_cost
                
                # Set hotel return time
                u_var[day, hotel_index] = max(u_var[day, hotel_index], current_time)
            except KeyError:
                # Missing transport data for return to hotel
                pass
        
        # Reshape x_var and u_var into the flat solution vector
        solution[:self.problem.x_shape] = x_var.flatten()
        solution[self.problem.x_shape:] = u_var.flatten()
        
        logger.info(f"Created heuristic solution with estimated cost: ${total_cost:.2f}")
        
        return solution
    
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
        
        for i in range(len(self.repair_operators)):
            if self.repair_count[i] > 0:
                self.weights_repair[i] += self.reaction_factor * (self.scores_repair[i] / self.repair_count[i])
        
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
        Run the ALNS algorithm
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            dict: Results including best solution and statistics
        """
        logger.info("Starting ALNS optimization...")
        
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
            "best_is_feasible": self.best_evaluation["is_feasible"]
        }
        
        # Main ALNS loop
        while True:
            # Check termination conditions
            if iteration >= self.max_iterations:
                logger.info("Terminating: Maximum iterations reached")
                break
                
            if self.time_limit and time.time() - start_time > self.time_limit:
                logger.info("Terminating: Time limit reached")
                break
            
            # Select destroy and repair operators
            destroy_idx = self.select_destroy_operator()
            repair_idx = self.select_repair_operator()
            
            # Update counters
            self.destroy_count[destroy_idx] += 1
            self.repair_count[repair_idx] += 1
            
            # Apply destroy operator
            destroyed_solution = self.destroy_operators[destroy_idx](self.problem, self.current_solution.copy())
            
            # Apply repair operator
            new_solution = self.repair_operators[repair_idx](self.problem, destroyed_solution)
            
            # Evaluate new solution
            new_evaluation = self.problem.evaluate_solution(new_solution)
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
                    
                    self.best_solution = new_solution.copy()
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
                else:
                    # Update weight scores - better than current but not best
                    self.scores_destroy[destroy_idx] += SCORE_BETTER
                    self.scores_repair[repair_idx] += SCORE_BETTER
                
                stats["accepted_count"] += 1
            else:
                # Update weight scores - solution rejected
                self.scores_destroy[destroy_idx] += SCORE_REJECTED
                self.scores_repair[repair_idx] += SCORE_REJECTED
                
                stats["rejected_count"] += 1
            
            # Record iteration history
            self.iteration_history.append({
                "iteration": iteration,
                "objective": self.current_objective,
                "temperature": self.temperature,
                "destroy_operator": destroy_idx,
                "repair_operator": repair_idx,
                "accepted": self.current_objective == new_objective
            })
            
            stats["objective_history"].append(self.current_objective)
            stats["temperature_history"].append(self.temperature)
            
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
                
                if verbose:
                    # Log progress at end of segment
                    elapsed = time.time() - start_time
                    logger.info(f"Segment complete - Iteration {iteration}, Temp: {self.temperature:.4f}, "
                               f"Current: {self.current_objective:.4f}, Best: {self.best_objective:.4f}, "
                               f"Time: {elapsed:.1f}s")
            
            # Increment iteration counter
            iteration += 1
        
        # Calculate final statistics
        stats["iterations"] = iteration
        stats["runtime"] = time.time() - start_time
        
        # Log final results
        logger.info(f"ALNS optimization completed in {stats['runtime']:.2f} seconds")
        logger.info(f"Best objective: {self.best_objective:.4f} (iteration {stats['best_found_at']})")
        logger.info(f"Best solution cost: ${self.best_evaluation['total_cost']:.2f}")
        logger.info(f"Best solution travel time: {self.best_evaluation['total_travel_time']:.2f} minutes")
        logger.info(f"Best solution satisfaction: {self.best_evaluation['total_satisfaction']:.2f}")
        logger.info(f"Best solution feasible: {self.best_evaluation['is_feasible']}")
        
        # Return best solution and statistics
        return {
            "best_solution": self.best_solution,
            "best_evaluation": self.best_evaluation,
            "stats": stats
        }