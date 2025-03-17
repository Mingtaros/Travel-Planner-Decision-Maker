import json

def get_transport_matrix():
    BASE_PATH = "data/routeData/"
    daytimes = [("morning", 8), ("midday", 12), ("evening", 16), ("night", 20)]
    transport_matrix = {}

    for time_in_day, hour in daytimes:
        filepath = BASE_PATH + f"route_matrix_{time_in_day}.json"

        with open(filepath, 'r') as f:
            route_matrix = json.load(f)

        for route in route_matrix["routes"]:
            this_route = route_matrix["routes"][route]
            origin = this_route["origin_name"]
            destination = this_route["destination_name"]
            transport_matrix[(origin, destination, hour)] = {
                "transit": {
                    "duration": this_route["transit"]["duration_minutes"],
                    "price": this_route["transit"]["fare_sgd"],
                },
                "drive": {
                    "duration": this_route["drive"]["duration_minutes"],
                    "price": this_route["drive"]["fare_sgd"],
                }
            }
    
    return transport_matrix


def get_all_locations():
    BASE_PATH = "data/routeData/"

    with open(BASE_PATH + "route_matrix_morning.json", 'r') as f:
        route_matrix = json.load(f)
    
    locations = [route_matrix["locations"][location_id] for location_id in route_matrix["locations"]]
    return locations


def get_transport_hour(transport_time):
    # because the transport_matrix is only bracketed to 4 groups, we find the earliest it happens
    brackets = [8, 12, 16, 20]
    transport_hour = transport_time // 60

    for bracket in reversed(brackets):
        if transport_hour >= bracket:
            return bracket

    return brackets[-1] # from 8 PM to 8 AM next day
