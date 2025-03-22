# Travel Itinerary Optimizer (VRP-Based)

## Overview

The Travel Itinerary Optimizer is an advanced Python-based solution for generating optimal multi-day travel itineraries in Singapore. Using a position-based Vehicle Routing Problem (VRP) formulation and Adaptive Large Neighborhood Search (ALNS), this project creates personalized travel plans that maximize satisfaction while minimizing cost and travel time.

This implementation specifically addresses time constraint challenges by using a position-based representation that naturally enforces sequence and time window constraints.

## Key Features

### Optimization Capabilities
- Multi-objective optimization (cost, travel time, satisfaction)
- Position-based VRP solution representation
- Time window enforcement for meal constraints
- Adaptive Large Neighborhood Search algorithm

### Constraints Handling
- Time window management with explicit timing
- Budget constraints
- Meal requirements (lunch and dinner at hawker centers)
- Transportation mode selection (transit or drive)
- Attraction visit limits (each attraction visited at most once)

### Visualization
- Detailed timeline plots
- Cost breakdown charts
- Satisfaction rating visualizations

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
   - At least 2 hawker center visits per day

3. **Budget Constraints**
   - Total trip cost within specified budget
   - Includes hotel, transportation, meals, and attraction fees

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
MYSQL_HOST=your_mysql_host
MYSQL_USER=your_mysql_username
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=your_database_name
```

## Usage

### Basic Example
```python
from main_vrp import main

# Run optimization with default parameters
results = main(
    hotel_name="Marina Bay Sands",
    budget=500,
    num_days=3,
    max_attractions=10,
    max_hawkers=5
)
```

### Advanced Configuration
```python
# Customize optimization parameters
results = main(
    hotel_name="Raffles Hotel",
    budget=750,
    num_days=4,
    max_attractions=15,
    max_hawkers=8
)

# Visualize results
from utils import SolutionVisualizer

# Generate comprehensive visualizations
visualizations = SolutionVisualizer.generate_comprehensive_visualization(
    problem, results['best_solution_vector']
)
```

## Project Structure
```
travel_itinerary/
│
├── vrp_solution.py       # Position-based VRP solution representation
├── vrp_operators.py      # VRP-specific destroy and repair operators
├── vrp_alns.py           # Modified ALNS for VRP representation
├── main_vrp.py           # Main entry point using VRP approach
│
├── problem/              # Problem definition and utilities
│   ├── itinerary_problem.py
│   ├── constraints.py
│   └── utils.py
│
├── data/                 # Data management
│   ├── transport_utils.py
│   ├── location_utils.py
│   └── cache_manager.py
│
└── utils/                # Utility modules
    ├── export_itinerary.py
    ├── google_maps_client.py
    └── visualization.py
```

## Algorithm Overview

### Position-Based VRP Representation
- Each solution is represented as a sequence of locations with arrival and departure times
- Natural handling of time windows and sequence constraints
- Direct enforcement of meal timing requirements

### Adaptive Large Neighborhood Search (ALNS)
- **Destroy Operators**: Partially disassemble solutions
  - Random day subsequence removal
  - Attraction removal based on value
  - Time window violation removal
  - Day shuffling

- **Repair Operators**: Reconstruct solutions
  - Greedy insertion
  - Regret-based insertion
  - Time-based insertion

- **Simulated Annealing**: Allows exploration of solution space

## Time Window Management

One of the key improvements in this implementation is the explicit handling of time windows:

- **Meal Windows**: Lunch (11 AM - 3 PM) and dinner (5 PM - 9 PM) slots are enforced through direct feasibility checks
- **Time Propagation**: When a location is inserted or removed, times are recalculated for all subsequent locations
- **Feasibility Validation**: Every insertion is checked for time window feasibility before being accepted

## Performance Optimization
- Time-aware insertion and removal operations
- Efficient constraint checking
- Adaptive operator selection

## Limitations
- Requires Google Maps API access
- Performance depends on computational resources
- Accuracy of results relies on input data quality

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments
- VRP formulation inspired by classical vehicle routing problems
- ALNS methodology based on academic literature
- Utilizes Google Maps API for routing