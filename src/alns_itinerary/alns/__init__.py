"""
ALNS module for the VRP-based Travel Itinerary Optimizer.
Implements position-based representation and specialized operators.
"""

# Import key components for easy access
from .vrp_alns import VRPALNS
from .vrp_solution import VRPSolution
from .vrp_operators import (
    # Destroy operators
    destroy_targeted_subsequence,
    destroy_worst_attractions,
    destroy_time_window_violations,
    destroy_expensive_attractions,
    destroy_selected_day,
    
    # Repair operators
    repair_regret_insertion,
    repair_time_based_insertion,
    repair_balanced_solution
)

# Specify which symbols to export when using "from alns import *"
__all__ = [
    # Core components
    'VRPALNS',
    'VRPSolution',
    
    # Destroy operators
    'destroy_targeted_subsequence',
    'destroy_worst_attractions',
    'destroy_time_window_violations',
    'destroy_expensive_attractions',
    'destroy_selected_day',
    
    # Repair operators
    'repair_regret_insertion',
    'repair_time_based_insertion',
    'repair_balanced_solution'
]