# Travel Planner & Decision Maker
Creating a Travel Planner mainly using Constraint Satisfaction Problem (CSP) for AI Planning and Decision Making

## Public Transport Fare Calculator
Provided Starting station / bus stop, and ending station / bus stop, get fares and distances.

Data Required:
<table>
    <tr>
        <th>Data Type</th><th>Definition</th><th>Source</th><th>Path</th>
    </tr><tr>
        <td>Bus Stop IDs</td><td>Bus Stop IDs that is used by LTA fare calculator API.</td><td>https://www.lta.gov.sg/content/ltagov/en/map/fare-calculator.html</td><td>data/bus_stop_to_id.json</td>
    </tr><tr>
        <td>MRT Station IDs</td><td>MRT Station IDs that is used by LTA fare calculator API.</td><td>https://www.lta.gov.sg/content/ltagov/en/map/fare-calculator.html</td><td>data/mrt_stop_to_id.json</td>
    </tr><tr>
        <td>Fare Type</td><td>Singapore Public Transport Prices differ based on whether the rider is an Adult, or a Student, or a Senior or Disabled, etc.</td><td>https://www.lta.gov.sg/content/ltagov/en/map/fare-calculator.html</td><td>data/fare_type.json</td>
</table>

Features:
- Different pricing for different fare types.
- Multiple-trips, (therefore trip 2 can continue pricing from trip 1).

Source: https://www.lta.gov.sg/content/ltagov/en/map/fare-calculator.html

## Members
- Daniel James (https://github.com/danieljames96)
- Leonardo (https://github.com/Mingtaros)
- Tai Jing Shen (https://github.com/sciencenerd880)
- Valerian Yap (https://github.com/valerianyap)
- Yuan Shengbo (https://github.com/CKAB693Yuan)
