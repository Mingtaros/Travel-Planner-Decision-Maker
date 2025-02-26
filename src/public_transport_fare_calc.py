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

def calculate_mrt_fare(source, destination, rider_type="Adult", tripInfo="", addTripInfo=""):
    # create payload
    payload = {
        "fare": fare_type[rider_type],
        "from": source,
        "to": destination,
    }

    # trip info can include previous rides that the customer did before this ride, so it's considered single trip, the price accumulated.
    payload["tripInfo"] = tripInfo if tripInfo else "usiAccumulatedDistance1=0-usiAccumulatedDistance2=0-usiAccumulatedDistance3=0-usiAccumulatedDistance4=0-usiAccumulatedDistance5=0-usiAccumulatedDistance6=0-usiAccumulatedFare1=0-usiAccumulatedFare2=0-usiAccumulatedFare3=0-usiAccumulatedFare4=0-usiAccumulatedFare5=0-usiAccumulatedFare6=0",
    payload["addTripInfo"] = addTripInfo if addTripInfo else "0",

    # make request
    response = requests.post(
        url=mrt_base_url,
        data=payload,
    )

    return response.json()


def calculate_bus_fare(source, destination, bus_number, rider_type="Adult", tripInfo="", addTripInfo=""):
    # create payload
    payload = {
        "fare": fare_type[rider_type],
        "from": source,
        "to": destination,
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
    first_trip = calculate_mrt_fare('10', '101') # TODO: find mapping of station & bus stop names to alightID and boardID
    second_trip = calculate_bus_fare('5704', '5120', '858', tripInfo=first_trip["tripInfo"], addTripInfo=first_trip["addTripInfo"])

    # fares are returned as string, need to convert first
    # fares are returned with dollars and cents not separated
    first_trip_fare = int(first_trip["fare"]) / 100
    second_trip_fare = int(second_trip["fare"]) / 100
    total_fare = first_trip_fare +  second_trip_fare

    print("Fares:\n" \
          f"    First Trip Fare: ${first_trip_fare}\n" \
          f"    Second Trip Fare: ${second_trip_fare}\n" \
          f"    Total Fare: ${total_fare}")
