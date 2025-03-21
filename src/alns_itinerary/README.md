# Travel Itinerary Optimizer

## Overview

The Travel Itinerary Optimizer is an advanced Python-based solution for generating optimal multi-day travel itineraries in Singapore. Utilizing Adaptive Large Neighborhood Search (ALNS) and sophisticated constraint satisfaction techniques, this project creates personalized travel plans that maximize satisfaction while minimizing cost and travel time.

## Key Features

### Optimization Capabilities
- Multi-objective optimization
- Constraint-based solution generation
- Adaptive search strategy
- Comprehensive solution evaluation

### Constraints
- Time window management
- Budget constraints
- Meal requirements
- Transportation mode selection
- Attraction visit limits

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
from alns_itinerary.main import main

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
from alns_itinerary.utils import SolutionVisualizer

# Generate comprehensive visualizations
visualizations = SolutionVisualizer.generate_comprehensive_visualization(
    problem, solution
)
```

## Project Structure
```
alns_itinerary/
│
├── alns/                # ALNS algorithm implementation
│   ├── alns_core.py     # Core ALNS algorithm
│   ├── destroy_operators.py
│   └── repair_operators.py
│
├── problem/             # Problem definition and utilities
│   ├── itinerary_problem.py
│   ├── constraints.py
│   └── utils.py
│
├── data/                # Data management
│   ├── transport_utils.py
│   └── cache_manager.py
│
└── utils/               # Utility modules
    ├── export_itinerary.py
    ├── google_maps_client.py
    └── visualization.py
```

## Algorithm Overview

### Adaptive Large Neighborhood Search (ALNS)
- **Destroy Operators**: Partially disassemble solutions
  - Random day removal
  - Attraction removal
  - Route modification

- **Repair Operators**: Reconstruct solutions
  - Greedy repair
  - Random repair
  - Satisfaction-based repair

- **Simulated Annealing**: Allows exploration of solution space

## Logging and Debugging
- Comprehensive logging
- Detailed constraint violation reporting
- Solution export and import capabilities

## Performance Optimization
- Caching mechanism for route computations
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

## License
[Specify your license here]

## Contact
For support or inquiries, please open an issue on GitHub or contact [your contact information]

## Acknowledgments
- Inspired by real-world travel planning challenges
- Utilizes advanced optimization techniques
- Leverages Google Maps API for routing
```

This README provides:
- Comprehensive project overview
- Installation instructions
- Usage examples
- Project structure
- Algorithm explanation
- Contribution guidelines

Would you like me to modify or expand on any section of the README?