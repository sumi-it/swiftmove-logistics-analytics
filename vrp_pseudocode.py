# NOTE: pseudocode-level illustration. Requires `pip install ortools`; not executed in Week 1.
# [[w1_vrp]]
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# dist_matrix : n x n integer distances in metres (depot = index 0), from OSRM / OpenStreetMap
# demands     : load (kg) at each stop, 0 for the depot
# capacities  : one capacity (kg) per vehicle, e.g. [800]*8 + [2000]*10 + [5000]*7
manager = pywrapcp.RoutingIndexManager(len(dist_matrix), len(capacities), 0)
routing = pywrapcp.RoutingModel(manager)

def distance_cb(i, j):
    return dist_matrix[manager.IndexToNode(i)][manager.IndexToNode(j)]
routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(distance_cb))

def demand_cb(i):
    return demands[manager.IndexToNode(i)]
routing.AddDimensionWithVehicleCapacity(routing.RegisterUnaryTransitCallback(demand_cb),
                                        0, capacities, True, "Capacity")

params = pywrapcp.DefaultRoutingSearchParameters()
params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
params.time_limit.seconds = 30

solution = routing.SolveWithParameters(params)
# Read the route of each vehicle from `solution`; compare total km with current manual routes
# [[/w1_vrp]]
