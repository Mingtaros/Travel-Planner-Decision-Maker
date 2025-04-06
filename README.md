# Singapore Travel Itinerary Project

A personalized AI-powered travel planner for Singapore using Multi-Agent Retrieval-Augmented Generation (RAG) and Adaptive Large Neighborhood Search (ALNS) with a position-based Vehicle Routing Problem (VRP) formulation.

## Overview

The Singapore Travel Itinerary Optimizer is a hybrid AI system that intelligently generates optimized, multi-day travel plans. It combines:
	•	Multi-Agent RAG for intelligent POI retrieval and user query understanding
	•	ALNS with position-based VRP formulation for optimizing itineraries based on budget, time, and satisfaction constraints

This end-to-end pipeline transforms a user’s travel query into a cost-effective and satisfying itinerary using LLM agents, search, and combinatorial optimization.

## Key Features

### Optimization Capabilities
- 🤖 Multi-Agent RAG System:
    - Supervisor Agent: Understands whether the user wants food or attractions
    - Attraction & Hawker Agents: Retrieve relevant POIs from PgVector DB
    - Itinerary Planner Agent: Generates travel plans with cost, time, and satisfaction estimates
- Multi-objective optimization (cost, travel time, satisfaction) & Multi-Day Travel Itineraries
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

## Prerequisites

- Python 3.8+
- Google Maps API Key
- OpenAI API Key
- Groq API Key
- Required Python packages (install via `pip install -r requirements.txt`)

## Tech Stack for Mutli Agent RAG
- Agno Framework for LLM-based agent orchestration
- PgVectorDB
- Docker for building container image

## Datasets
<table>
    <tr>
        <th>Data Type</th><th>Description</th><th>Source</th><th>Path</th>
    </tr><tr>
        <td>Singapore Travel Destinations</td><td>Geolocation and descriptions of Travel Destinations in Singapore. Entrance fee and costs are obtained using LLMs.</td><td>https://data.gov.sg/datasets/d_0f2f47515425404e6c9d2a040dd87354/view</td><td>data/attractions.csv</td>
    </tr><tr>
        <td>Singapore Hawker Centers</td><td>Geolocation and names of Hawker Centers in Singapore</td><td>https://data.gov.sg/datasets/d_4a086da0a5553be1d89383cd90d07ecd/view</td><td>data/hawkers.csv</td>
    </tr>
</table>

## Directory Structure

```
travel-planner-decision-maker/
├── log/  (will be created automatically)
├── results/ (all itinerary results are generated here)
├── data/
│   ├── locationData/
│   │   └── csv/
│   │       ├── attractions.csv
│   │       └── hawkers.csv
│   ├── waypointData/  (will be created automatically)
│   │   └── waypoints.json
│   └── routeData/  (will be created automatically)
│       ├── route_matrix_morning.json
│       ├── route_matrix_midday.json
│       ├── route_matrix_evening.json
│       └── route_matrix_night.json
└── src/
    ├── route_matrix/
    |   ├── google_maps_client.py
    |   ├── waypoint_generator.py
    |   └── generate_route_matrix.py
    └── alns_itinerary/
        ├── alns/                 # ALNS algorithm implementation
        │   ├── vrp_alns.py               # Enhanced ALNS for VRP representation
        │   ├── vrp_operators.py          # Destroy and repair operators
        │   └── vrp_solution.py           # Position-based solution representation
        ├── problem/              # Problem definition
        │   └── itinerary_problem.py      # Problem class with constraints
        ├── data/                 # Data management
        │   ├── transport_utils.py        # Transport matrix handling
        │   ├── location_utils.py         # Location data processing
        │   ├── cache_manager.py          # Caching for API responses
        │   └── trip_detail.py            # Getting location details for trips
        ├── utils/                # Utility modules
        │   ├── export_json_itinerary.py  # JSON export functionality
        │   ├── google_maps_client.py     # Google Maps API client
        │   └── config.py                 # Configuration loading
        ├── agentic/              # Agentic Modules
        │   └── multiagent.py             # Agents and Tools Declaration
        ├── streamlit_app.py      # Main entry point for streamlit app
        ├── alns_main.py          # Main entry point for ALNS algorithm
        ├── config.json           # Algorithm configuration
        └── llm.json              # Trip parameters
```

## Setup Instructions
1. Clone this repository
2. Create a `.env` file in the root directory with your Google Maps,OpenAI, and Groq API key:
   ```
   GOOGLE_MAPS_API_KEY=your_api_key_here
   OPENAI_API_KEY=your_api_key_here
   GROQ_API_KEY=your_api_key_here
   ```
   See `.env.example` file for the template.
3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Turn on Docker service and run the Posgres VectorDB (PGVector):
    ```bash
    docker run -d \
        -e POSTGRES_DB=ai \
        -e POSTGRES_USER=ai \
        -e POSTGRES_PASSWORD=ai \
        -e PGDATA=/var/lib/postgresql/data/pgdata \
        -v pgvolume:/var/lib/postgresql/data \
        -p 5532:5432 \
        --name pgvector \
        agnohq/pgvector:16
    ```

## Data Preparation

The itinerary optimization requires two preprocessing steps:

### Step 1: Generate Waypoints

Run the following command to geocode all locations and store them in JSON format:

```bash
python ./src/route_matrix/waypoint_generator.py --attractions data/locationData/attractions.csv --hawkers data/locationData/hawkers.csv --output data/waypointData/waypoints.json
```

This script:
- Reads attractions and hawker centers from CSV files
- Geocodes them using the Google Maps API
- Stores the coordinates in a JSON file (`data/waypointData/waypoints.json`)
- Only needs to be run once (unless new locations are added)

### Step 2: Generate Route Matrix

Run the following command to generate route matrices for different times of day:

**NOTE: PLEASE CHECK API USAGE LIMITS FOR FREE TIER AND ADJUST NUMBER OF LOCATIONS USED ACCORDINGLY**

```bash
python ./src/route_matrix/generate_route_matrix.py --waypoints data/waypointData/waypoints.json --output-dir data/routeData
```

This script:
- Loads waypoints from the JSON file
- Processes them in batches to comply with Google Maps API limits (100 elements per request)
- Computes transit and driving routes between all locations for different times of day
- Calculates fares for each route type
- Saves the results in JSON files (`data/routeData/route_matrix_*.json`)

## Streamlit Application

### Basic Usage
```bash
streamlit run ./src/alns_itinerary/streamlit_app.py
```

Enter the itinerary details and click on Generate Itinerary

## Evaluation

### Basic Usage
Update the dictionary `user_queries` and update line 115 with the query key.
```bash
python ./src/alns_itinerary/evalute_alns_agentic_rag.py
```

### `config.json`
Contains algorithm parameters and constraints:
```json
{
    "MAX_ATTRACTION_PER_DAY": 4, 
    "RICH_THRESHOLD": 100,
    "MEAL_BUFFER_TIME": 240,
    "AVG_HAWKER_COST": 20,
    "RATING_MAX": 5,
    "START_TIME": 540,
    "HARD_LIMIT_END_TIME": 1320,
    "LUNCH_START": 720,
    "LUNCH_END": 900,
    "DINNER_START": 1080,
    "DINNER_END": 1260,
    "HAWKER_TIME_SPENT": 60,
    "APPROX_HOTEL_TRAVEL_COST": 15,
    "MAX_ITERATIONS": 2000,
    "SEGMENT_SIZE": 100,
    "TIME_LIMIT": 600,
    "EARLY_TERMINATION_ITERATIONS": 500,
    "WEIGHTS_DESTROY": [1.0, 1.0, 1.0, 1.0, 1.0],
    "WEIGHTS_REPAIR": [1.0, 1.0, 1.0],
    "OBJECTIVE_WEIGHTS": [0.33, 0.33, 0.33],
    "WEIGHTS_SCORES": [3, 2, 1, 0],
    "INFEASIBLE_PENALTY": 10,
    "DESTROY_REMOVE_PERCENTAGE": 0.3,
    "DESTROY_DISTANT_LOC_WEIGHTS": [0.7, 0.3],
    "DESTROY_DISTANT_LOC_PERCENTAGE": 0.3,
    "DESTROY_EXPENSIVE_THRESHOLD": 0.9,
    "DESTROY_DAY_HAWKER_PRESERVE": 0.7,
    "REPAIR_TRANSIT_WEIGHTS": [0.7, 0.3],
    "REPAIR_SATISFACTION_WEIGHTS": [0.3, 0.7],
    "BUDGET_OBJECTIVE_LIMIT": 0.7,
    "TRAVEL_TIME_OBJECTIVE_LIMIT": 0.3,
    "SATISFACTION_OBJECTIVE_LIMIT": 1.0,
    "TRAVEL_TIME_REFERENCE": 150
}
```

### Examining Results

After ALNS optimization completes, you can find the itineraries in the `results` directory:
- `initial_itinerary_TIMESTAMP.json`: The initial solution before optimization
- `best_itinerary_TIMESTAMP.json`: The optimized itinerary

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

## Notes and Assumptions
- Routes data is obtained through Google Maps API
- Assume that Hotel Prices and Entrance fee prices are static
- Performance depends on the size of the problem instance
- Quality of results depends on the accuracy of location data

## Members
- Daniel James (https://github.com/danieljames96)
- Leonardo (https://github.com/Mingtaros)
- Tai Jing Shen (https://github.com/sciencenerd880)
- Valerian Yap (https://github.com/valerianyap)
- Yuan Shengbo (https://github.com/CKAB693Yuan)