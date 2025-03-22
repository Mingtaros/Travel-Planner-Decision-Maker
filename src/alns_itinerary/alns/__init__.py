"""
ALNS module for the VRP-based Travel Itinerary Optimizer.
Implements position-based representation and specialized operators.
"""

# Import key components for easy access
from .vrp_alns import VRPALNS
from .vrp_solution import VRPSolution
from .vrp_operators import (
    # Destroy operators
    destroy_random_day_subsequence,
    destroy_worst_attractions,
    destroy_random_attractions,
    destroy_random_meals,
    destroy_time_window_violations,
    destroy_day_shuffle,
    
    # Repair operators
    repair_greedy_insertion,
    repair_regret_insertion,
    repair_time_based_insertion
)

# Specify which symbols to export when using "from alns import *"
__all__ = [
    # Core components
    'VRPALNS',
    'VRPSolution',
    
    # Destroy operators
    'destroy_random_day_subsequence',
    'destroy_worst_attractions',
    'destroy_random_attractions', 
    'destroy_random_meals',
    'destroy_time_window_violations',
    'destroy_day_shuffle',
    
    # Repair operators
    'repair_greedy_insertion',
    'repair_regret_insertion',
    'repair_time_based_insertion'
]