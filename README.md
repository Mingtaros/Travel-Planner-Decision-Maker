# Singapore Travel Itinerary Project

Creating a Travel Planner mainly using Constraint Satisfaction Problem (CSP) for AI Planning and Decision Making.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- Python 3.8+
- Google Maps API key

## Datasets
<table>
    <tr>
        <th>Data Type</th><th>Description</th><th>Source</th><th>Path</th>
    </tr><tr>
        <td>Bus Stop Routes</td><td>Returns detailed route information for all services currently in operation, including:  all bus stops along each route, first/last bus timings for each stop.</td><td>https://datamall.lta.gov.sg/content/datamall/en/dynamic-data.html</td><td>data/bus_routes.csv</td>
    </tr><tr>
        <td>Fare Type</td><td>Singapore Public Transport Prices differ based on whether the rider is an Adult, or a Student, or a Senior or Disabled, etc.</td><td>https://www.lta.gov.sg/content/ltagov/en/map/fare-calculator.html</td><td>data/fare_type.json</td>
    </tr><tr>
        <td>Singapore MRT Stations</td><td>Details of MRT Stations in Singapore</td><td>https://www.kaggle.com/datasets/shengjunlim/singapore-mrt-lrt-stations-with-coordinates, https://www.lta.gov.sg/content/ltagov/en/map/fare-calculator.html</td><td>data/mrt_stations.csv</td>
    </tr><tr>
        <td>Singapore Bus Stops</td><td>Details of Bus Stops in Singapore</td><td>https://datamall2.mytransport.sg/ltaodataservice/BusStops, https://www.lta.gov.sg/content/ltagov/en/map/fare-calculator.html</td><td>data/bus_stops.csv</td>
    </tr><tr>
        <td>Singapore Travel Destinations</td><td>Geolocation and descriptions of Travel Destinations in Singapore. Entrance fee and costs are obtained using LLMs.</td><td>https://data.gov.sg/datasets/d_0f2f47515425404e6c9d2a040dd87354/view</td><td>data/attractions.csv</td>
    </tr><tr>
        <td>Singapore Hawker Centers</td><td>Geolocation and names of Hawker Centers in Singapore</td><td>https://data.gov.sg/datasets/d_4a086da0a5553be1d89383cd90d07ecd/view</td><td>data/hawker_centers.csv</td>
    </tr><tr>
        <td>Singapore Hotels</td><td>Geolocation and names of Hotels in Singapore. Per-night costs are obtained using LLMs</td><td>https://data.gov.sg/datasets/d_654e22f14e5bb817423f0e0c9ac4f632/view</td><td>data/hotels.csv</td>
    </tr>
</table>

## Directory Structure

```
travel-planner-decision-maker/
├── docker-compose.yml
├── init/
│   └── init.sql
├── data/
│   ├── attractions.csv
│   ├── tourists.csv
│   ├── foodcentre.csv
│   ├── preference.csv
│   └── routeData/  (will be created automatically)
├── mysql-data/  (will be created automatically)
├── src/
│   ├── load_data.py
│   ├── store_waypoints.py
│   ├── generate_route_matrix.py
│   └── utils/
│       └── google_maps_client.py
└── log/  (will be created automatically)
```

## Setup Instructions

1. Clone this repository
2. Place your CSV files in the `data` directory
3. Create a `.env` file in the root directory with the following variables:
   ```
   MYSQL_USER=<user>
   MYSQL_PASSWORD=<password>
   MYSQL_DATABASE=<database>
   MYSQL_HOST=<host>
   MYSQL_PORT=<port>
   GOOGLE_MAPS_API_KEY=your_api_key_here
   ```
4. Start the database container:
   ```bash
   docker-compose up -d
   ```
5. Verify the database is running:
   ```bash
   docker-compose ps
   ```
6. Connect to the database:
   ```bash
   docker exec -it <container_name> mysql -u <user> -p<password>
   ```
7. Run the data loading script:
   ```bash
   python ./src/load_data.py
   ```
8. Verify data was loaded:
   ```sql
   USE ai_planning_project;
   SELECT * FROM attractions LIMIT 5;
   SELECT * FROM foodcentre LIMIT 5;
   ```

## Route Matrix Generation

The route matrix generation process involves two main steps:

1. **Store waypoint data** (geocoding locations - only needs to be done once)
2. **Generate route matrices** (can be run for different times of day)

### Step 1: Store Waypoints

Run the following command to geocode all locations and store them in the database:

```bash
python ./src/store_waypoints.py
```

This script:
- Fetches attractions and foodcentres from the database
- Geocodes them using the Google Maps API
- Stores the coordinates in a `waypoints` table
- Only needs to be run once (unless new locations are added)

Verify the waypoints were stored:
```sql
SELECT * FROM waypoints LIMIT 5;
```

### Step 2: Generate Route Matrix

Run the following command to generate route matrices for different times of day:

```bash
python ./src/generate_route_matrix.py
```

This script:
- Fetches waypoints from the database
- Processes them in batches to comply with Google Maps API limits (100 elements per request)
- Computes transit and driving routes between all locations
- Calculates fares for each route type
- Saves the results in JSON files (`data/routeData/route_matrix_*.json`)
- Stores the results in the database (`route_matrix` table)

You can modify the departure times in the script to generate matrices for different times of day.

Verify the route matrix was stored:
```sql
SELECT * FROM route_matrix LIMIT 5;
```

## Exporting and Importing the Database

### Exporting the Database (for sharing with collaborators)

1. Export as SQL dump file:
   ```bash
   docker exec -i ai_planning_project_db sh -c 'mysqldump --no-tablespaces -u planner -p"plannerpassword" ai_planning_project' > ai_planning_project_dump.sql
   ```

### Importing the Database

Import a SQL dump file:
   ```bash
    docker exec -i ai_planning_project_db sh -c 'mysql -u planner -p"plannerpassword" -e "CREATE DATABASE IF NOT EXISTS ai_planning_project;"'

    docker exec -i ai_planning_project_db sh -c 'mysql -u planner -p"plannerpassword" ai_planning_project' < ai_planning_project_dump.sql
   ```

## Constraint Optimization Problem

### Inequality Constraints (G)

1. **Attraction Visit Limits**:
   - Each attraction must be visited at most once as a source
   - Each attraction must be visited at most once as a destination

2. **Time Constraints**:
   - For each route chosen (x_var[i,j,k,l] = 1), the finish time at destination l must be at least the finish time at source k plus transport duration plus activity duration at l

3. **Hawker Visit Requirements**:
   - Each day must include at least 2 visits to hawker centers
   - Each day must include at least 1 hawker visit during lunch hours (11 AM to 3 PM)
   - Each day must include at least 1 hawker visit during dinner hours (5 PM to 9 PM)

4. **Transport Mode Constraints**:
   - For each route, only one transport type can be chosen (can't use both transit and drive for the same route)

5. **Budget Constraint**:
   - Total cost of hotel, attractions, hawker meals, and transportation must be within the specified budget

6. **Minimum and Maximum Visits**:
   - Total visits must be at least the minimum (2 hawker visits per day)
   - Total visits must be at most the maximum (6 visits per day × number of days)

### Equality Constraints (H)

1. **Hotel Starting Point**:
   - Each day must start from the hotel (u_var[i,0] must be the smallest time value for each day)

2. **Flow Conservation**:
   - For each location and each day, the number of incoming routes must equal the number of outgoing routes

3. **Return to Hotel Requirement**:
   - At the end of each day, must return to the hotel

4. **Attraction Visit Symmetry**:
   - If an attraction is visited as a source, it must also be visited as a destination (and vice versa)

### Time Windows

The problem also enforces time windows:
- Daily start time: 9 AM (540 minutes)
- Hard limit end time: 10 PM (1320 minutes)
- Lunch window: 11 AM to 3 PM (660-900 minutes)
- Dinner window: 5 PM to 9 PM (1020-1260 minutes)

### Optimization Objectives

The problem has three objectives being minimized:
1. Total cost (hotel, attractions, food, transport)
2. Total travel time
3. Negative satisfaction (maximizing satisfaction by minimizing its negative)

These constraints collectively ensure a feasible and optimal travel itinerary that respects time windows, budget limitations, and visit requirements while maximizing user satisfaction.

## Modules
### Public Transport Fare Calculator
Provided Starting station / bus stop, and ending station / bus stop, get fares and distances.

Data Required:
- Singapore Bus Stops
- Singapore MRT Stations
- Fare Type (Rider types)

Features:
- Different pricing for different rider types.
- Supports multi-trips, (therefore trip 2 can continue pricing from trip 1).

### Trip Details
Given the origin location and destination location of where the user is going to go, return the possible routes (trips) including their price and arrival time

Modules required:
- Public Transport Fare Calculator
- Google Maps Get Direction

## Notes and Assumptions
- Routes data is obtained through Google Maps API.
- Assume that Hotel Prices and Entrance fee prices are static.

## Members
- Daniel James (https://github.com/danieljames96)
- Leonardo (https://github.com/Mingtaros)
- Tai Jing Shen (https://github.com/sciencenerd880)
- Valerian Yap (https://github.com/valerianyap)
- Yuan Shengbo (https://github.com/CKAB693Yuan)