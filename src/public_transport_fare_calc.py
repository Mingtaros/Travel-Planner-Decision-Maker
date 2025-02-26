import json
import requests


mrt_base_url = "https://www.lta.gov.sg/content/ltagov/en/map/fare-calculator/jcr:content/map2-content/farecalculator.mrtget.html"
bus_base_url = "https://www.lta.gov.sg/content/ltagov/en/map/fare-calculator/jcr:content/map2-content/farecalculator.busget.html"

fare_type = {
    "Adult": '30',
    "Disabled": '37',
    "Workfare Concession": '38',
    "Senior Citizen": '39',
    "Student": '40',
}

with open("data/bus_stop_to_id.json", 'r') as f:
    bus_stop_to_id = json.load(f)

with open("data/mrt_stop_to_id.json", 'r') as g:
    mrt_stop_to_id = json.load(g)

def calculate_mrt_fare(source, destination, rider_type="Adult", tripInfo="", addTripInfo=""):
    # create payload
    payload = {
        "fare": fare_type[rider_type],
        "from": mrt_stop_to_id[source],
        "to": mrt_stop_to_id[destination],
    }

    # trip info can include previous rides that the customer did before this ride, so it's considered single trip, the price accumulated.
    payload["tripInfo"] = tripInfo if tripInfo else "usiAccumulatedDistance1=0-usiAccumulatedDistance2=0-usiAccumulatedDistance3=0-usiAccumulatedDistance4=0-usiAccumulatedDistance5=0-usiAccumulatedDistance6=0-usiAccumulatedFare1=0-usiAccumulatedFare2=0-usiAccumulatedFare3=0-usiAccumulatedFare4=0-usiAccumulatedFare5=0-usiAccumulatedFare6=0",
    payload["addTripInfo"] = addTripInfo if addTripInfo else "0",

    # make requestx
    response = requests.post(
        url=mrt_base_url,
        data=payload,
    )

    return response.json()


def calculate_bus_fare(source, destination, bus_number, rider_type="Adult", tripInfo="", addTripInfo=""):
    # create payload
    payload = {
        "fare": fare_type[rider_type],
        "from": bus_stop_to_id[source],
        "to": bus_stop_to_id[destination],
        "bus": bus_number,
    }

    # trip info can include previous rides that the customer did before this ride, so it's considered single trip, the price accumulated.
    payload["tripInfo"] = tripInfo if tripInfo else "usiAccumulatedDistance1=0-usiAccumulatedDistance2=0-usiAccumulatedDistance3=0-usiAccumulatedDistance4=0-usiAccumulatedDistance5=0-usiAccumulatedDistance6=0-usiAccumulatedFare1=0-usiAccumulatedFare2=0-usiAccumulatedFare3=0-usiAccumulatedFare4=0-usiAccumulatedFare5=0-usiAccumulatedFare6=0",
    payload["addTripInfo"] = addTripInfo if addTripInfo else "0",

    # make request
    response = requests.post(
        url=bus_base_url,
        data=payload,
    )

    return response.json()


if __name__ == "__main__":
    """
    Suppose case where a user is going from Dhoby Ghaut to Changi Airport Terminal 1
    transit at Punggol. The trip starts from Dhoby Ghaut MRT, and ends at Changi
    Airport Terminal 1 Bus Stop. Therefore there are 2 trips, 1 MRT trip, 1 bus trip.
    
    Expected Output:
        Fares:
            First Trip Fare: $1.93
            Second Trip Fare: $0.41
            Total Fare: $2.34
        Distance:
            First Trip Distance: 13.9 km
            Second Trip Distance: 14.2 km
            Total Distance: 28.1 km
    """

    first_trip = calculate_mrt_fare(
        source="Dhoby Ghaut (NS24 / NE6 / CC1)",
        destination="Punggol (NE17 / PTC)",
    )
    second_trip = calculate_bus_fare(
        source="65199 - Aft Punggol Rd",
        destination="95029 - Changi Airport Ter 1",
        bus_number='858',
        tripInfo=first_trip["tripInfo"],
        addTripInfo=first_trip["addTripInfo"],
    )

    # fares are returned as string, need to convert first
    # fares are returned in cents
    first_trip_fare = int(first_trip["fare"]) / 100
    second_trip_fare = int(second_trip["fare"]) / 100
    total_fare = first_trip_fare +  second_trip_fare

    # distances are also returned as string, which we need to convert
    # same as fares, returned in decameters, which needs to be converted to kilometers
    first_trip_distance = int(first_trip['distance']) / 100
    second_trip_distance = int(second_trip['distance']) / 100
    total_distance = first_trip_distance + second_trip_distance

    print("Fares:\n" \
          f"    First Trip Fare: ${first_trip_fare}\n" \
          f"    Second Trip Fare: ${second_trip_fare}\n" \
          f"    Total Fare: ${total_fare}\n" \
          "Distance:\n" \
          f"    First Trip Distance: {first_trip_distance} km\n" \
          f"    Second Trip Distance: {second_trip_distance} km\n" \
          f"    Total Distance: {total_distance} km")
