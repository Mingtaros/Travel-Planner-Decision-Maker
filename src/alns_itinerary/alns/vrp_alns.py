"""
Adaptive Large Neighborhood Search for Travel Itinerary Optimization
===================================================================

This module implements an enhanced Adaptive Large Neighborhood Search (ALNS)
algorithm specifically designed for optimizing travel itineraries using a
Vehicle Routing Problem (VRP) approach.

Key Features:
- Position-based VRP representation for time-constrained scheduling
- Simulated annealing acceptance criteria with auto-calibrated temperature
- Weighted adaptive operator selection
- Multiple destroy and repair strategies
- Support for complex constraints (meals, budget, time windows)
- Early termination and time limit capabilities

The ALNS algorithm works by iteratively destroying and repairing solutions,
using operators that are weighted based on their historical performance.
The algorithm balances exploration and exploitation through simulated annealing.
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
from alns.vrp_operators import VRPOperators

logger = logging.getLogger(__name__)

class VRPALNS:
    """
    Enhanced Adaptive Large Neighborhood Search (ALNS) for travel itinerary optimization.
    
    This class provides a complete implementation of the ALNS metaheuristic
    adapted to the travel itinerary problem. It manages the optimization process,
    operator selection, solution acceptance, and performance tracking.
    
    The algorithm balances three main objectives:
    1. Minimizing total cost (transportation and attractions)
    2. Minimizing travel time between locations
    3. Maximizing total satisfaction/enjoyment
    
    Attributes:
        problem: The TravelItineraryProblem instance containing constraints
        current_solution: The current working solution
        best_solution: The best solution found so far
        destroy_operators: List of destroy operator functions
        repair_operators: List of repair operator functions
        weights_destroy/weights_repair: Adaptive weights for operator selection
        temperature: Current simulated annealing temperature
        objective_weights: Weights for the multi-objective function
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
        infeasible_penalty=10.0,
        early_termination_iterations=200,
        objective_weights=[0.3, 0.3, 0.4],
        weights_scores=[3, 2, 1, 0],
        attraction_per_day=4,
        rich_threshold=100,
        avg_hawker_cost=15,
        rating_max=10,
        meal_buffer_time=90,
        approx_hotel_travel_cost=10,
        destroy_remove_percentage=0.3,
        destroy_distant_loc_weights=[0.5, 0.5],
        destroy_expensive_threshold=0.9,
        destroy_day_hawker_preserve=0.7,
        repair_transit_weights=[0.5, 0.5],
        repair_satisfaction_weights=[0.5, 0.5],
    ):
        """
        Initialize the ALNS algorithm for travel itinerary optimization.
        
        Args:
            problem: The TravelItineraryProblem instance containing constraints
            initial_solution: Optional starting solution (default: creates a heuristic solution)
            destroy_operators: List of destroy operators (default: uses standard operators)
            repair_operators: List of repair operators (default: uses standard operators)
            weights_destroy: Initial weights for destroy operators (default: equal weights)
            weights_repair: Initial weights for repair operators (default: equal weights)
            max_iterations: Maximum number of iterations to perform (default: 1000)
            segment_size: Number of iterations before weight updates (default: 100)
            reaction_factor: Learning rate for weight adjustments (default: 0.5)
            decay_factor: Weight decay rate (default: 0.8)
            temperature_control: Cooling rate for simulated annealing (default: 0.95)
            initial_temperature: Starting temperature (default: auto-calibrated)
            time_limit: Maximum runtime in seconds (default: None)
            seed: Random seed for reproducibility (default: None)
            infeasible_penalty: Penalty factor for infeasible solutions (default: 10.0)
            early_termination_iterations: Iterations without improvement before stopping (default: 200)
            objective_weights: Weights for [cost, travel_time, satisfaction] (default: [0.3, 0.3, 0.4])
            weights_scores: Scores for [new_best, better, accepted, rejected] solutions (default: [3, 2, 1, 0])
            attraction_per_day: Target number of attractions per day (default: 4)
            rich_threshold: Budget threshold for using more expensive transport (default: 100)
            avg_hawker_cost: Average cost of meals at hawker centers (default: 15)
            rating_max: Maximum possible satisfaction rating (default: 10)
            meal_buffer_time: Buffer time around meals in minutes (default: 90)
            approx_hotel_travel_cost: Estimated cost for hotel transit (default: 10)
            destroy_remove_percentage: Percentage of solution to destroy (default: 0.3)
            destroy_distant_loc_weights: Weights for distance-based destruction (default: [0.5, 0.5])
            destroy_expensive_threshold: Threshold for expensive location removal (default: 0.9)
            destroy_day_hawker_preserve: Probability to preserve hawkers during destruction (default: 0.7)
            repair_transit_weights: Weights for transit-based insertion (default: [0.5, 0.5])
            repair_satisfaction_weights: Weights for satisfaction-based insertion (default: [0.5, 0.5])
        """
        self.problem = problem
        self.max_iterations = max_iterations
        self.segment_size = segment_size
        self.reaction_factor = reaction_factor
        self.decay_factor = decay_factor
        self.temperature_control = temperature_control
        self.time_limit = time_limit
        self.early_termination_iterations = early_termination_iterations
        self.objective_weights = objective_weights
        self.infeasible_penalty = infeasible_penalty
        self.rating_max = rating_max
        self.approx_hotel_travel_cost = approx_hotel_travel_cost
        self.rich_threshold = rich_threshold
        self.avg_hawker_cost = avg_hawker_cost
        self.attraction_per_day = attraction_per_day
        self.meal_buffer_time = meal_buffer_time
        self.weights_scores = weights_scores
        self.seed = seed
        self.hotel_idx = 0
        self.diverse_solution_limit = 10
        self.destroy_remove_percentage = destroy_remove_percentage
        self.destroy_distant_loc_weights = destroy_distant_loc_weights
        self.destroy_expensive_threshold = destroy_expensive_threshold
        self.destroy_day_hawker_preserve = destroy_day_hawker_preserve
        self.repair_satisfaction_weights = repair_satisfaction_weights
        self.repair_transit_weights = repair_transit_weights
        
        operators_config = {
            "destroy_remove_percentage": self.destroy_remove_percentage,
            "destroy_distant_loc_weights": self.destroy_distant_loc_weights,
            "destroy_expensive_threshold": self.destroy_expensive_threshold,
            "destroy_day_hawker_preserve": self.destroy_day_hawker_preserve,
            "repair_satisfaction_weights": self.repair_satisfaction_weights,
            "repair_transit_weights": self.repair_transit_weights,
            "objective_weights": self.objective_weights,
            "seed": self.seed
        }
        
        operators = VRPOperators(**operators_config)
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize destroy operators
        if destroy_operators is None:
            self.destroy_operators = [
                operators.destroy_targeted_subsequence,
                operators.destroy_worst_attractions,
                operators.destroy_distant_locations,
                operators.destroy_expensive_attractions,
                operators.destroy_selected_day
            ]
        else:
            self.destroy_operators = destroy_operators
        
        # Initialize repair operators
        if repair_operators is None:
            self.repair_operators = [
                operators.repair_regret_insertion,
                operators.repair_transit_efficient_insertion,
                operators.repair_satisfaction_driven_insertion
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
        Create a valid initial solution for the itinerary problem.
        
        Constructs a feasible initial solution with a greedy heuristic approach:
        1. Selects attractions based on value (satisfaction/cost ratio)
        2. Ensures each day has exactly one lunch and dinner at appropriate times
        3. Properly schedules attractions between meals
        4. Respects budget and time constraints
        5. Returns to hotel at the end of each day
        
        Returns:
            VRPSolution: A valid initial solution that satisfies all hard constraints
            
        Note:
            The quality of this initial solution significantly impacts the
            efficiency of the optimization process.
        """

        solution = VRPSolution(self.problem)
        
        budget_left = self.problem.budget
        
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
            
            latest_completion_time = self.problem.START_TIME
            current_position = 1  # Start at position 1 (after hotel)
            num_days_left = solution.num_days - day
            lunch_inserted = 0
            dinner_inserted = 0
            approx_mandatory_cost = ((num_days_left * 2) - lunch_inserted - dinner_inserted) * self.avg_hawker_cost + self.approx_hotel_travel_cost
            attraction_count = 0
            
            logger.info(f"Approx mandatory cost: {approx_mandatory_cost}")
            
            # Skip if we don't have enough hawkers
            if len(hawker_ratings) < 2:
                # logger.warning("Not enough unique hawkers for each day's lunch and dinner")
                continue
            
            if budget_left/num_days_left < self.rich_threshold:
                transport_mode = "transit"
            else:
                transport_mode = "drive"
            
            logger.info(f"Day {day+1}: Starting with ${round(budget_left, 2)} budget, using {transport_mode} transport")
            
            count = 0
            
            # Visit attractions until lunch time
            while (latest_completion_time + self.meal_buffer_time) < self.problem.LUNCH_END and approx_mandatory_cost < budget_left and attraction_count < self.attraction_per_day and attraction_values:
                
                count += 1
                attr_idx, attr_ratio = attraction_values.pop(0)
                
                attr_cost, attr_duration = solution.get_cost_duration(day, attr_idx, current_position, transport_mode)
                
                if approx_mandatory_cost + attr_cost <= budget_left and attr_duration + latest_completion_time <= self.problem.LUNCH_END:
                    # Insert the attraction
                    check, departure_time = solution.insert_location(day, current_position, attr_idx, transport_mode)
                    if check:
                        attraction_count += 1
                        current_position += 1
                        latest_completion_time = departure_time
                        budget_left -= attr_cost
                        # Remove this attraction from consideration for other days
                        attraction_values = [(a, v) for a, v in attraction_values if a != attr_idx]
                        logger.info(f"Day {day+1}: Inserted attraction {attr_idx} with cost {attr_cost} and duration {attr_duration}")
                else:
                    attraction_values.append((attr_idx, attr_ratio))  # Add back to the end of the list
                
                if count >= 10:
                    # logger.warning("Could not insert enough attractions before lunch time")
                    break
            
            logger.info(f"Day {day+1}: Leaving for Lunch at {latest_completion_time//60}:{latest_completion_time%60} with ${round(budget_left, 2)} budget left")
            
            lunch_hawker_idx = -1
            # Have lunch at appropriate time
            while lunch_inserted == 0:
                if hawker_ratings:
                    lunch_hawker_idx, hwk_ratio = hawker_ratings.pop(0)
                    hwk_cost, hwk_duration = solution.get_cost_duration(day, lunch_hawker_idx, current_position, transport_mode)
                    
                    if approx_mandatory_cost + hwk_cost - self.avg_hawker_cost <= budget_left: # Ensure we can afford the hawker
                        # Insert the hawker center
                        check, lunch_departure_time = solution.insert_location(day, current_position, lunch_hawker_idx, transport_mode, 'Lunch')
                        if check:
                            current_position += 1
                            latest_completion_time = lunch_departure_time
                            budget_left -= hwk_cost
                            lunch_inserted = 1
                            logger.info(f"Day {day+1}: Inserted lunch at hawker {lunch_hawker_idx} with cost {hwk_cost} and duration {hwk_duration}")
                            # hawker_ratings.append((lunch_hawker_idx, hwk_ratio))
            
            approx_mandatory_cost = ((num_days_left * 2) - lunch_inserted - dinner_inserted) * self.avg_hawker_cost + self.approx_hotel_travel_cost
            
            logger.info(f"Day {day+1}: Lunch completed at {latest_completion_time//60}:{latest_completion_time%60} with ${round(budget_left, 2)} budget left")
            logger.info(f"Approx mandatory cost: {approx_mandatory_cost}")
            
            count = 0
            # Visit attractions until dinner time
            while (latest_completion_time + self.meal_buffer_time) < self.problem.DINNER_END and approx_mandatory_cost < budget_left and attraction_count < self.attraction_per_day and attraction_values:
                
                count += 1
                attr_idx, attr_ratio = attraction_values.pop(0)
                
                attr_cost, attr_duration = solution.get_cost_duration(day, attr_idx, current_position, transport_mode)
                
                if approx_mandatory_cost + attr_cost <= budget_left and attr_duration + latest_completion_time <= self.problem.DINNER_END:
                    # Insert the attraction
                    check, departure_time = solution.insert_location(day, current_position, attr_idx, transport_mode)
                    if check:
                        attraction_count += 1
                        current_position += 1
                        latest_completion_time = departure_time
                        budget_left -= attr_cost
                        # Remove this attraction from consideration for other days
                        attraction_values = [(a, v) for a, v in attraction_values if a != attr_idx]
                        logger.info(f"Day {day+1}: Inserted attraction {attr_idx} with cost {attr_cost} and duration {attr_duration}")
                else:
                    attraction_values.append((attr_idx, attr_ratio))  # Add back to the end of the list
                    
                if count >= 10:
                    # logger.warning("Could not insert enough attractions before dinner time")
                    break
            
            logger.info(f"Day {day+1}: Leaving for Dinner at {latest_completion_time//60}:{latest_completion_time%60} with ${round(budget_left, 2)} budget left")
                        
            # Have dinner at appropriate time
            while dinner_inserted == 0:
                if hawker_ratings:
                    dinner_hawker_idx, hwk_ratio = hawker_ratings.pop(0)
                    hwk_cost, hwk_duration = solution.get_cost_duration(day, dinner_hawker_idx, current_position, transport_mode)
                    
                    if approx_mandatory_cost + hwk_cost - self.avg_hawker_cost <= budget_left and dinner_hawker_idx != lunch_hawker_idx: # Ensure we can afford the hawker
                        # Insert the hawker center
                        check, dinner_departure_time = solution.insert_location(day, current_position, dinner_hawker_idx, transport_mode, 'Dinner')
                        if check:
                            current_position += 1
                            latest_completion_time = dinner_departure_time
                            budget_left -= hwk_cost
                            dinner_inserted = 1
                            logger.info(f"Day {day+1}: Inserted dinner at hawker {dinner_hawker_idx} with cost {hwk_cost} and duration {hwk_duration}")
                            # hawker_ratings.append((dinner_hawker_idx, hwk_ratio))
            
            # Add hotel return at the end
            approx_mandatory_cost = ((num_days_left * 2) - lunch_inserted - dinner_inserted) * self.avg_hawker_cost + self.approx_hotel_travel_cost
            logger.info(f"Day {day+1}: Dinner completed at {latest_completion_time//60}:{latest_completion_time%60} with ${round(budget_left, 2)} budget left")
            logger.info(f"Approx mandatory cost: {approx_mandatory_cost}")
            
            # Find transport mode that works
            hotel_return_inserted = False
            for transport_mode in ["drive", "transit"]:
                hotel_cost, hotel_duration = solution.get_cost_duration(day, self.hotel_idx, current_position, transport_mode)
                logger.info(f"Day {day+1}: Hotel return cost: {hotel_cost}, duration: {hotel_duration}, transport: {transport_mode}")
                if (latest_completion_time + hotel_duration) < self.problem.HARD_LIMIT_END_TIME and approx_mandatory_cost + hotel_cost <= budget_left:
                    hotel_return_inserted = True
                    solution.hotel_return_transport[day] = transport_mode
                    solution.hotel_transit_duration = hotel_duration
                    budget_left -= hotel_cost
                    break
            
            logger.info(f"Day {day+1}: Hotel return inserted: {hotel_return_inserted}")
        
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
        Calculate the overall objective value for a solution.
        
        Combines multiple objectives (cost, travel time, satisfaction) into
        a single value for simulated annealing-based optimization. Lower
        values indicate better solutions.
        
        Args:
            evaluation (dict): Evaluation metrics from solution.evaluate()
            
        Returns:
            float: Combined objective value (lower is better)
            
        Note:
            Infeasible solutions receive a large penalty to ensure the
            algorithm prioritizes finding feasible solutions.
        """
        # Extract objectives
        cost = evaluation["total_cost"]
        travel_time = evaluation["total_travel_time"]
        satisfaction = evaluation["total_satisfaction"]
        max_satisfaction = (self.problem.NUM_DAYS * 6) * self.rating_max
        reference_time = 6 * 60  # 6 hours as a reference point
        
        budget_objective = self.objective_weights[0] * (cost / self.problem.budget)
        travel_time_objective = self.objective_weights[1] * (travel_time / (travel_time + reference_time))
        satisfaction_objective = self.objective_weights[2] * (satisfaction / max_satisfaction)
        
        # logger.info("Budget_Objective: {}, Satisfaction_Objective: {}, Travel_Time_Objective: {}".format(budget_objective, satisfaction_objective, travel_time_objective))
        
        # Normalize and combine objectives (weighted sum)
        objective =  budget_objective + travel_time_objective - satisfaction_objective
        
        # Add large penalty for infeasible solutions
        if not evaluation["is_feasible"]:
            objective += self.infeasible_penalty
        
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
    
    def run(self, verbose=True):
        """
        Execute the ALNS optimization process.
        
        This is the main method that runs the complete ALNS algorithm:
        1. Iteratively destroys and repairs the current solution
        2. Uses simulated annealing to accept or reject new solutions
        3. Updates operator weights based on performance
        4. Tracks the best solution found
        5. Terminates upon reaching stopping criteria
        
        Args:
            verbose (bool): Whether to output detailed progress logs
            
        Returns:
            dict: Results dictionary containing:
                - best_solution: The best VRPSolution found
                - best_evaluation: Detailed metrics of the best solution
                - stats: Comprehensive statistics about the optimization process
                
        Note:
            The algorithm terminates when it reaches max_iterations, time_limit,
            or early_termination_iterations without improvement.
        """
        logger.info("Starting VRP-based ALNS optimization...")
        
        start_time = time.time()
        iteration = 0
        segment_iteration = 0
        
        # Score constants for updating weights
        SCORE_BEST = self.weights_scores[0]     # New best solution
        SCORE_BETTER = self.weights_scores[1]   # Better than current but not best
        SCORE_ACCEPTED = self.weights_scores[2] # Worse but accepted
        SCORE_REJECTED = self.weights_scores[3] # Rejected
        
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
                if new_objective < self.best_objective and new_evaluation["is_feasible"]:  # Only update if feasible
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
                    if len(self.diverse_solutions) > self.diverse_solution_limit:
                        self.diverse_solutions.sort(key=lambda x: x["objective"])
                        self.diverse_solutions = self.diverse_solutions[:self.diverse_solution_limit]
                elif new_evaluation["is_feasible"]:
                    if verbose:
                        logger.info(f"Iteration {iteration}: New solution accepted (objective: {new_objective:.4f}, feasible: {new_evaluation['is_feasible']})")
                    # Update weight scores - feasible but not best
                    self.scores_destroy[destroy_idx] += SCORE_BETTER
                    self.scores_repair[repair_idx] += SCORE_BETTER
                else:
                    if verbose:
                        logger.info(f"Iteration {iteration}: New worse solution accepted (objective: {new_objective:.4f}, feasible: {new_evaluation['is_feasible']})")
                    # Update weight scores - feasible but not best
                    self.scores_destroy[destroy_idx] += SCORE_ACCEPTED
                    self.scores_repair[repair_idx] += SCORE_ACCEPTED
                stats["accepted_count"] += 1
                
                # Increment counter for early termination
                self.iterations_since_improvement += 1
            else:
                if verbose:
                    logger.info(f"Iteration {iteration}: New solution rejected (objective: {new_objective:.4f})")
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
                self.temperature *= self.temperature_control
                
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