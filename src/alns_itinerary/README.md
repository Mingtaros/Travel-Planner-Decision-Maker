# Travel Itinerary Optimizer (VRP-Based)

## Overview

The Travel Itinerary Optimizer is an advanced Python-based solution for generating optimal multi-day travel itineraries in Singapore. Using a position-based Vehicle Routing Problem (VRP) formulation and Adaptive Large Neighborhood Search (ALNS), this system creates personalized travel plans that maximize satisfaction while balancing cost and travel time.

This implementation specifically addresses time constraint challenges by using a position-based representation that naturally enforces sequence and time window constraints, particularly for meal scheduling.

## Key Features

### Optimization Capabilities
- Multi-objective optimization (cost, travel time, satisfaction)
- Position-based VRP solution representation
- Time window enforcement for meal constraints
- Adaptive Large Neighborhood Search with specialized operators
- Automatic meal scheduling (lunch and dinner at hawker centers)

### Constraints Handling
- Time window management with explicit timing
- Budget constraints with fine-grained control
- Meal requirements (lunch and dinner at hawker centers)
- Transportation mode selection (transit or drive)
- Attraction uniqueness (each attraction visited at most once)
- Daily attraction limits for realistic itineraries

### Export Capabilities
- Detailed JSON itineraries with timing information
- Transportation details between locations
- Cost breakdown by category
- Satisfaction and time metrics

## Problem Domain

The optimizer solves a complex travel planning problem with multiple objectives:
- Minimize total trip cost
- Minimize total travel time
- Maximize trip satisfaction

### Constraints
1. **Time Constraints**
   - Daily start time: 9 AM
   - Hard end time: 10 PM
   - Specific lunch (11 AM - 3 PM) and dinner (5 PM - 9 PM) windows

2. **Location Constraints**
   - Each attraction can be visited only once
   - Mandatory hawker center visits for lunch and dinner
   - Maximum number of attractions per day for realistic scheduling

3. **Budget Constraints**
   - Total trip cost within specified budget
   - Includes transportation, meals, and attraction fees

## Installation

### Prerequisites
- Python 3.8+
- pip

### Dependencies
```bash
pip install -r requirements.txt
```

### Required Environment Variables
Create a `.env` file with:
```
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
```

## Usage

### Basic Example
```python
from main import main

# Run optimization with default parameters
results = main(
    seed=42,
    config_path="./src/alns_itinerary/config.json",
    llm_path="./src/alns_itinerary/llm.json",
    max_attractions=16,
    max_hawkers=12
)
```

### Configuration Files
The system uses two main configuration files:

#### `config.json`
Contains algorithm parameters and constraints:
```json
{
    "MAX_ITERATIONS": 5000,
    "SEGMENT_SIZE": 100,
    "TIME_LIMIT": 3600,
    "EARLY_TERMINATION_ITERATIONS": 500,
    "MAX_ATTRACTION_PER_DAY": 4,
    "START_TIME": 540,
    "HARD_LIMIT_END_TIME": 1320,
    "LUNCH_START": 660,
    "LUNCH_END": 900,
    "DINNER_START": 1020,
    "DINNER_END": 1260,
    "WEIGHTS_DESTROY": [1.0, 1.0, 1.0, 1.0, 1.0],
    "WEIGHTS_REPAIR": [1.0, 1.0, 1.0],
    "OBJECTIVE_WEIGHTS": [0.3, 0.3, 0.4],
    "INFEASIBLE_PENALTY": 10.0,
    "RICH_THRESHOLD": 100,
    "AVG_HAWKER_COST": 15,
    "RATING_MAX": 10,
    "MEAL_BUFFER_TIME": 90,
    "APPROX_HOTEL_TRAVEL_COST": 10,
    "WEIGHTS_SCORES": [3, 2, 1, 0],
    "DESTROY_REMOVE_PERCENTAGE": 0.3,
    "DESTROY_DISTANT_LOC_WEIGHTS": [0.5, 0.5],
    "DESTROY_EXPENSIVE_THRESHOLD": 0.9,
    "DESTROY_DAY_HAWKER_PRESERVE": 0.7,
    "REPAIR_TRANSIT_WEIGHTS": [0.5, 0.5],
    "REPAIR_SATISFACTION_WEIGHTS": [0.5, 0.5]
}
```

#### `llm.json`
Contains trip-specific parameters:
```json
{
    "HOTEL_NAME": "Marina Bay Sands",
    "BUDGET": 750,
    "NUM_DAYS": 3
}
```

### Examining Results

After optimization completes, you can find the itineraries in the `results` directory:
- `initial_itinerary_TIMESTAMP.json`: The initial solution before optimization
- `best_itinerary_TIMESTAMP.json`: The optimized itinerary

## Project Structure
```
itinerary_optimizer/
│
├── alns/                 # ALNS algorithm implementation
│   ├── vrp_alns.py       # Enhanced ALNS for VRP representation
│   ├── vrp_operators.py  # Destroy and repair operators
│   └── vrp_solution.py   # Position-based solution representation
│
├── problem/              # Problem definition
│   └── itinerary_problem.py  # Problem class with constraints
│
├── data/                 # Data management
│   ├── transport_utils.py    # Transport matrix handling
│   ├── location_utils.py     # Location data processing
│   └── cache_manager.py      # Caching for API responses
│
├── utils/                # Utility modules
│   ├── export_json_itinerary.py  # JSON export functionality
│   ├── google_maps_client.py     # Google Maps API client
│   └── config.py              # Configuration loading
│
├── main.py               # Main entry point
├── config.json           # Algorithm configuration
└── llm.json              # Trip parameters
```

## Algorithm Overview

### Position-Based VRP Representation
- Each solution is represented as a sequence of locations with arrival and departure times
- Natural handling of time windows and sequence constraints
- Direct enforcement of meal timing requirements

### Adaptive Large Neighborhood Search (ALNS)
- **Destroy Operators**:
  - `destroy_targeted_subsequence`: Removes a subsequence of attractions while preserving meals
  - `destroy_worst_attractions`: Removes attractions with poor satisfaction-to-cost ratio
  - `destroy_distant_locations`: Removes locations requiring excessive travel
  - `destroy_expensive_attractions`: Removes costly attractions to improve budget utilization
  - `destroy_selected_day`: Restructures an entire day while preserving meals

- **Repair Operators**:
  - `repair_regret_insertion`: Uses regret-based insertion with meal scheduling
  - `repair_transit_efficient_insertion`: Prioritizes travel time efficiency
  - `repair_satisfaction_driven_insertion`: Prioritizes overall satisfaction

- **Simulated Annealing**: Temperature-based acceptance of worse solutions to explore solution space

## Time Window Management

The system handles time windows explicitly:

- **Meal Windows**: Lunch and dinner slots are enforced through direct feasibility checks
- **Time Propagation**: When a location is inserted or removed, times are recalculated for all subsequent locations
- **Rest Periods**: The system accounts for waiting time at hawker centers if arriving before opening

## Constraint Violations

The system provides detailed information about constraint violations when solutions are infeasible:
- Budget constraints
- Time window constraints
- Meal scheduling constraints
- Attraction uniqueness constraints

## Performance Optimization
- Caching of routes between locations
- Efficient constraint checking during insertion
- Adaptive operator selection based on performance
- Early termination when no improvement is found

## Limitations
- Requires Google Maps API access for accurate travel times
- Performance depends on the size of the problem instance
- Quality of results depends on the accuracy of location data

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments
- VRP formulation inspired by classical vehicle routing problems
- ALNS methodology based on academic literature in combinatorial optimization
- Utilizes Google Maps API for accurate transport timing and costs