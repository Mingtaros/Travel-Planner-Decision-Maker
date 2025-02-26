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
    </tr>
</table>

Features:
- Multiple-trips, (therefore trip 2 can continue pricing from trip 1)

Source from: https://www.lta.gov.sg/content/ltagov/en/map/fare-calculator.html

## Members
- Daniel James
- Leonardo
- Tai Jing Shen
- Valerian Yap
- Yuan Shengbo
