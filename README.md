# Travel Planner & Decision Maker
Creating a Travel Planner mainly using Constraint Satisfaction Problem (CSP) for AI Planning and Decision Making

## Datasets
<table>
    <tr>
        <th>Data Type</th><th>Description</th><th>Source</th><th>Path</th>
    </tr><tr>
        <td>Bus Stop IDs</td><td>Bus Stop IDs that is used by LTA fare calculator API.</td><td>https://www.lta.gov.sg/content/ltagov/en/map/fare-calculator.html</td><td>data/bus_stop_to_id.json</td>
    </tr><tr>
        <td>MRT Station IDs</td><td>MRT Station IDs that is used by LTA fare calculator API.</td><td>https://www.lta.gov.sg/content/ltagov/en/map/fare-calculator.html</td><td>data/mrt_stop_to_id.json</td>
    </tr><tr>
        <td>Bus Stop Routes</td><td>Returns detailed route information for all services currently in operation, including:  all bus stops along each route, first/last bus timings for each stop.</td><td>https://datamall.lta.gov.sg/content/datamall/en/dynamic-data.html</td><td>data/bus_routes.csv</td>
    </tr><tr>
        <td>Fare Type</td><td>Singapore Public Transport Prices differ based on whether the rider is an Adult, or a Student, or a Senior or Disabled, etc.</td><td>https://www.lta.gov.sg/content/ltagov/en/map/fare-calculator.html</td><td>data/fare_type.json</td>
    </tr><tr>
        <td>Singapore MRT Station Geolocation</td><td>Geolocation of MRT Stations in Singapore</td><td>https://www.kaggle.com/datasets/shengjunlim/singapore-mrt-lrt-stations-with-coordinates</td><td>data/mrt_stations.csv</td>
    </tr><tr>
        <td>Singapore Bus Stops Geolocation</td><td>Geolocation of Bus Stops in Singapore</td><td>https://datamall2.mytransport.sg/ltaodataservice/BusStops</td><td>data/bus_stops.csv</td>
    </tr><tr>
        <td>Singapore Travel Destinations</td><td>Geolocation and descriptions of Travel Destinations in Singapore. Entrance fee and costs are obtained using LLMs.</td><td>https://data.gov.sg/datasets/d_0f2f47515425404e6c9d2a040dd87354/view</td><td>data/attractions.csv</td>
    </tr><tr>
        <td>Singapore Hawker Centers</td><td>Geolocation and names of Hawker Centers in Singapore</td><td>https://data.gov.sg/datasets/d_4a086da0a5553be1d89383cd90d07ecd/view</td><td>data/hawker_centers.csv</td>
    </tr><tr>
        <td>Singapore Hotels</td><td>Geolocation and names of Hotels in Singapore. Per-night costs are obtained using LLMs</td><td>https://data.gov.sg/datasets/d_654e22f14e5bb817423f0e0c9ac4f632/view</td><td>data/hotels.csv</td>
    </tr>
</table>

## Features
### Public Transport Fare Calculator
Provided Starting station / bus stop, and ending station / bus stop, get fares and distances.

Data Required:
- Bus Stop IDs
- MRT Station IDs
- Fare Type

Features:
- Different pricing for different fare types.
- Multiple-trips, (therefore trip 2 can continue pricing from trip 1).

Source: https://www.lta.gov.sg/content/ltagov/en/map/fare-calculator.html

## Notes and Assumptions
- Routes data is obtained through Google Maps API.
- Assume that Hotel Prices and Entrance fee prices are static.

## Members
- Daniel James (https://github.com/danieljames96)
- Leonardo (https://github.com/Mingtaros)
- Tai Jing Shen (https://github.com/sciencenerd880)
- Valerian Yap (https://github.com/valerianyap)
- Yuan Shengbo (https://github.com/CKAB693Yuan)
