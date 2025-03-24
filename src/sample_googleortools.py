from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import random

def create_data_model(num_days=2):
    data = {}

    # 10 POIs (0 = depot), symmetric distance matrix
    data['distance_matrix'] = [
        [0, 20, 40, 35, 25, 50, 30, 55, 60, 45],
        [20, 0, 25, 30, 35, 60, 50, 40, 70, 55],
        [40, 25, 0, 20, 30, 45, 35, 50, 65, 40],
        [35, 30, 20, 0, 15, 30, 40, 60, 55, 45],
        [25, 35, 30, 15, 0, 20, 25, 50, 60, 35],
        [50, 60, 45, 30, 20, 0, 15, 25, 40, 30],
        [30, 50, 35, 40, 25, 15, 0, 20, 35, 25],
        [55, 40, 50, 60, 50, 25, 20, 0, 30, 15],
        [60, 70, 65, 55, 60, 40, 35, 30, 0, 10],
        [45, 55, 40, 45, 35, 30, 25, 15, 10, 0],
    ]

    data['rewards'] = [0, 80, 60, 70, 50, 90, 100, 75, 85, 95]  # Rewards for nodes 1–9
    data['num_locations'] = len(data['distance_matrix'])
    data['num_vehicles'] = num_days
    data['depot'] = 0
    data['daily_time_budget'] = 480  # 8 hours/day
    return data

def main(num_days=2):
    data = create_data_model(num_days)

    manager = pywrapcp.RoutingIndexManager(
        data['num_locations'],
        data['num_vehicles'],
        data['depot']
    )

    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add dimension to enforce per-day (vehicle) time budget
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        data['daily_time_budget'],
        True,
        'Time'
    )

    time_dimension = routing.GetDimensionOrDie('Time')

    # Allow skipping POIs, and penalize it using reward
    for node in range(1, data['num_locations']):  # Exclude depot (0)
        index = manager.NodeToIndex(node)
        reward = data['rewards'][node]
        penalty = 1000 - reward  # Missing a high-reward POI is costly
        routing.AddDisjunction([index], penalty)

    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(10)

    # Solve
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        total_reward = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route = []
            route_reward = 0
            print(f"\nDay {vehicle_id + 1} Itinerary:")
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                if node != data['depot']:
                    route_reward += data['rewards'][node]
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            print(f"Route: {route}")
            print(f"Reward collected: {route_reward}")
            total_reward += route_reward

        print(f"\nTotal reward across {num_days} days: {total_reward}")
    else:
        print("No solution found.")

if __name__ == '__main__':
    # Change the number of days here
    main(num_days=2)
