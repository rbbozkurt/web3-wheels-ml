# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 R. Berkay Bozkurt <resitberkaybozkurt@gmail.com>

import random

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox


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


def get_random_nodes(graph, withData=True):
    """
    This function returns two random nodes from the graph.
    """
    nodes = list(graph.nodes(data=withData))
    return random.choice(nodes)


def find_best_matches(G, vehicles, destinations):
    """
    This function finds the best matches between vehicles and destinations.
    """
    best_matches = []
    for target_node_id, _ in vehicles:
        best_match = None
        best_distance = float("inf")
        for dest_node_id, _ in destinations:
            if nx.has_path(G, target_node_id, dest_node_id) == True:
                distance = nx.shortest_path_length(G, target_node_id, dest_node_id)
                if distance < best_distance:
                    best_distance = distance
                    best_match = dest_node_id
        best_matches.append((target_node_id, best_match))
    return best_matches


def find_routes(G, matches):
    """
    This function finds the best routes between vehicles and destinations.
    """
    routes = []

    for match in matches:
        vehicle, dest = match
        routes.append(nx.shortest_path(G, vehicle, dest))
    return routes


def draw_plain_map(G):
    """
    Draws a plain map without any nodes or routes.

    Parameters:
    - G: networkx.MultiDiGraph
        Input graph

    Returns:
    - fig, ax: tuple
        Matplotlib figure and axis objects
    """
    fig, ax = ox.plot_graph(G, node_size=0)
    plt.draw()
    plt.pause(0.5)
    return fig, ax


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
    plt.pause(0.5)


def draw_route_on_map(
    G, routes, pre_ax, route_color="blue", route_linewidth=2, route_alpha=0.2
):
    """
    Draws a route on the map.

    Parameters:
    - G: networkx.MultiDiGraph
        Input graph
    - route: list
        Route as a list of node IDs
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
    print(routes)
    x = []
    y = []
    for route in routes:
        for u, v in zip(route[:-1], route[1:]):
            # if there are parallel edges, select the shortest in length
            data = min(G.get_edge_data(u, v).values(), key=lambda d: d["length"])
            if "geometry" in data:
                # if geometry attribute exists, add all its coords to list
                xs, ys = data["geometry"].xy
                x.extend(xs)
                y.extend(ys)
            else:
                # otherwise, the edge is a straight line from node to node
                x.extend((G.nodes[u]["x"], G.nodes[v]["x"]))
                y.extend((G.nodes[u]["y"], G.nodes[v]["y"]))
            pre_ax.plot(x, y, c=route_color, lw=route_linewidth, alpha=route_alpha)
        x.clear()
        y.clear()
        plt.draw()
        plt.pause(0.5)


def show_map(fig):
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
    x, y = None, None
    node_list = list(G.nodes(data=True))
    for id, node in node_list:
        print(id, node)
        if id == node_id:
            x = node["x"]
            y = node["y"]
    return x, y


def find_x_y_coordinates_of_nodes(G, node_ids: list):
    """
    This function returns the x and y coordinates of the nodes.

    Parameters:
    - G: networkx.MultiDiGraph
        Input graph
    - node_ids: list
        IDs of the nodes

    Returns:
    - x, y: tuple
        x and y coordinates of the nodes
    """
    x, y = None, None
    node_list = list(G.nodes(data=True))
    matching_nodes = []
    for id, node in node_list:
        if id in node_ids:
            x = node["x"]
            y = node["y"]
            matching_nodes.append((x, y))
    return matching_nodes
