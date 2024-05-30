# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 R. Berkay Bozkurt <resitberkaybozkurt@gmail.com>

import heapq
import random

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

PLOT_PAUSE = 1


def G_map_from_address(address: str = "350 5th Ave, New York, New York"):
    """
    This function plots the map of the city.
    """
    return ox.graph_from_address(address, network_type="drive")


def G_map_from_place(place: str = "Berlin, Germany"):
    """
    This function plots the map of the city.
    """
    return ox.graph_from_place(place, network_type="drive")


def get_random_nodes(graph, with_data=False):
    """
    This function returns two random nodes from the graph.
    """
    nodes = list(graph.nodes(data=with_data))
    return random.choice(nodes)


import heapq

import networkx as nx


def find_best_matches(G, vehicle_node_ids, destinations_node_id):
    """
    This function finds the best matches between vehicles and destinations.
    """
    best_matches = []
    assigned_destinations = set()
    for vehicle_node_id in vehicle_node_ids:
        distances = []
        for dest_node_id in destinations_node_id:
            if (
                vehicle_node_id in G
                and dest_node_id in G
                and dest_node_id not in assigned_destinations
            ):
                try:
                    distance = nx.shortest_path_length(G, vehicle_node_id, dest_node_id)
                    heapq.heappush(distances, (distance, dest_node_id))
                except nx.NetworkXNoPath:
                    continue
        if distances:
            best_match = heapq.heappop(distances)[1]
            best_matches.append((vehicle_node_id, best_match))
            assigned_destinations.add(best_match)
    return best_matches


def find_routes(G, source_dest_pairs: []):
    """
    This function finds the best routes between vehicles and destinations.
    """
    routes = []

    for source, dest in source_dest_pairs:
        routes.append(nx.shortest_path(G, source, dest))
    return routes


def find_route(G, source_node_id, target_node_id):
    """
    This function finds the best route between two nodes.
    """
    return nx.shortest_path(G, source_node_id, target_node_id)


def find_distance(G, source_node_id, target_node_id):
    """
    This function calculates the price of the ride.
    """
    return nx.shortest_path_length(G, source_node_id, target_node_id)


def draw_plain_map(G, pre_ax=None):
    """
    Draws a plain map without any nodes or routes.

    Parameters:
    - G: networkx.MultiDiGraph
        Input graph

    Returns:
    - fig, ax: tuple
        Matplotlib figure and axis objects
    """
    fig, ax = ox.plot_graph(G, ax=pre_ax, node_size=0)
    plt.draw()
    plt.pause(PLOT_PAUSE)
    return fig, ax


def draw_outgoing_roads_to_neighbors(
    G, node_id, road_color="green", road_linewidth=3, road_alpha=0.9, pre_ax=None
):
    """
    Draws outgoing roads to neighbors of a node.

    Parameters:
    - G: networkx.MultiDiGraph
        Input graph
    - node_id: int
        ID of the node

    Returns:
    - fig, ax: tuple
        Matplotlib figure and axis objects
    """
    neighbor_node_ids = list(G[node_id].keys())
    for neighbor_node_id in neighbor_node_ids:
        edge_data = G.get_edge_data(node_id, neighbor_node_id).values()
        x, y = [], []
        for data in edge_data:
            if "geometry" in data:
                xs, ys = data["geometry"].xy
                x.extend(xs)
                y.extend(ys)
            else:
                x.extend((G.nodes[node_id]["x"], G.nodes[neighbor_node_id]["x"]))
                y.extend((G.nodes[node_id]["y"], G.nodes[neighbor_node_id]["y"]))
            pre_ax.plot(x, y, c=road_color, lw=road_linewidth, alpha=road_alpha)
    plt.draw()
    plt.pause(PLOT_PAUSE)


def draw_passengers_and_vehicles(
    G,
    vehicles_node_ids,
    passengers_node_ids,
    node_colors={"vehicles": "yellow", "passengers": "red"},
    node_sizes={"vehicles": 20, "passengers": 30},
    pre_ax=None,
):
    """
    Draws nodes on the map.

    Parameters:
    - G: networkx.MultiDiGraph
        Input graph
    - passengers: list
        List of passenger nodes
    - vehicles: list
        List of vehicle nodes
    - node_colors: dict, optional
        Colors of the nodes
    - node_sizes: dict, optional
        Sizes of the nodes

    Returns:
    - fig, ax: tuple
        Matplotlib figure and axis objects
    """
    vehicles_x_y = find_x_y_coordinates_of_nodes(G, vehicles_node_ids)
    passengers_x_y = find_x_y_coordinates_of_nodes(G, passengers_node_ids)
    pre_ax.scatter(
        *zip(*vehicles_x_y), color=node_colors["vehicles"], s=node_sizes["vehicles"]
    )
    pre_ax.scatter(
        *zip(*passengers_x_y),
        color=node_colors["passengers"],
        s=node_sizes["passengers"]
    )
    plt.draw()
    plt.pause(PLOT_PAUSE)


def draw_route_on_map(
    G, routes, pre_ax, route_color="red", route_linewidth=2, route_alpha=0.6
):
    """
    Draws a route on the map.

    Parameters:
    - G: networkx.MultiDiGraph
        Input graph
    - routes: list of lists
        Routes as a list of lists of node IDs
    - route_color: str, optional
        Color of the route
    - route_linewidth: int, optional
        Width of the route line
    - route_alpha: float, optional
        Opacity of the route line

    Returns:
    - fig, ax: tuple
        Matplotlib figure and axis objects
    """
    for route in routes:
        x, y = [], []
        for u, v in zip(route[:-1], route[1:]):
            data = min(G.get_edge_data(u, v).values(), key=lambda d: d["length"])
            if "geometry" in data:
                xs, ys = data["geometry"].xy
                x.extend(xs)
                y.extend(ys)
            else:
                x.extend((G.nodes[u]["x"], G.nodes[v]["x"]))
                y.extend((G.nodes[u]["y"], G.nodes[v]["y"]))
            pre_ax.plot(x, y, c=route_color, lw=route_linewidth, alpha=route_alpha)
        plt.draw()
        plt.pause(PLOT_PAUSE)


def show_map():
    """
    This function shows the map.
    """
    plt.show()


def find_x_y_coordinates_of_node(G, node_id: int):
    """
    This function returns the x and y coordinates of the node.

    Parameters:
    - G: networkx.MultiDiGraph
        Input graph
    - node_id: int
        ID of the node

    Returns:
    - x, y: tuple
        x and y coordinates of the node
    """
    node = G.nodes[node_id]
    return node["x"], node["y"]


def find_closest_node(G, x, y):
    """
    This function returns the closest node to the given x and y coordinates.

    Parameters:
    - G: networkx.MultiDiGraph
        Input graph
    - x: float
        x coordinate
    - y: float
        y coordinate

    Returns:
    - node_id: int
        ID of the closest node
    """
    print("x", x)
    print("y", y)
    print(type(x))
    return ox.distance.nearest_nodes(G, x, y, return_dist=False)


def find_x_y_coordinates_of_nodes(G, node_ids: list):
    """
    This function returns the x and y coordinates of the nodes.

    Parameters:
    - G: networkx.MultiDiGraph
        Input graph
    - node_ids: list
        IDs of the nodes

    Returns:
    - coordinates: list of tuples
        x and y coordinates of the nodes
    """
    return [find_x_y_coordinates_of_node(G, id) for id in node_ids]
