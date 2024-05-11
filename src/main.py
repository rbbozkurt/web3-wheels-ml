# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 R. Berkay Bozkurt <resitberkaybozkurt@gmail.com>
import matplotlib.pyplot as plt

from demos import (
    Action_To_String,
    GraphNavigationEnv,
    GridWorldEnv,
    graph_world_350_5th_Ave_New_York_New_York,
    grid_world_5_x_5,
)
from openstreetsmap_api import *


def print_actions(actions: list):
    """
    This function prints the actions in a human-readable format.
    """
    for index, action in enumerate(actions):
        print(
            "Vehicle {} taking the action: {}".format(index, Action_To_String[action])
        )


def parse_print_info(info: dict):
    """
    This function parses and prints the information dictionary.
    """
    print(" ------ Info ------ ")
    print("Vehicles:")
    for id, vehicle in info["vehicles"].items():
        print(f"ID: {id}")
        for key, value in vehicle.items():
            print(f"  {key}: {value}")
        print()

    print("Passengers:")
    for id, passenger in info["passengers"].items():
        print(f"ID: {id}")
        for key, value in passenger.items():
            print(f"  {key}: {value}")
        print()

    print("Gas Stations:")
    for id, gas_station in info["gas_stations"].items():
        print(f"ID: {id}")
        for key, value in gas_station.items():
            print(f"  {key}: {value}")
        print()


def main():
    """
    This is the main function of the program.
    It serves as the entry point for the application.
    """
    grid_world = GridWorldEnv(
        map_string=grid_world_5_x_5, n_vehicle=1, n_passenger=1, n_gas_station=1
    )
    observation, info = grid_world.reset(seed=42)
    print(" ------ Initial Info ------ ")
    print()
    print(grid_world.render())
    parse_print_info(info)
    for _ in range(1000):
        actions = grid_world.action_space.sample()
        print_actions(actions)
        observation, reward, terminated, truncated, info = grid_world.step(actions)
        grid_world.render()
        parse_print_info(info)
        if terminated or truncated:
            observation, info = grid_world.reset()
    grid_world.close()


def simulate_map_passenger_vehicle(
    n_vehicle=4,
    n_passenger=15,
):
    """
    This function plots the map of the city.
    """
    map_G = G_map_from_address()

    vehicle_nodes = []
    for _ in range(n_vehicle):
        vehicle_nodes.append(get_random_nodes(map_G))

    passenger_nodes = []
    for _ in range(n_passenger):
        passenger_nodes.append(get_random_nodes(map_G))
    # best_matches = find_best_matches(map_G, vehicle_nodes, passenger_nodes)
    # routes = find_routes(map_G, best_matches)
    # _ , ax = draw_plain_map(map_G)
    # draw_passengers_and_vehicles(map_G,pre_ax=ax,vehicles=vehicle_nodes, passengers=passenger_nodes)
    # draw_route_on_map(map_G, routes, ax)


def test_graph_navigation_env():
    """
    This function tests the GraphNavigationEnv class.
    """
    env = GraphNavigationEnv(graph_world_350_5th_Ave_New_York_New_York)
    observation = env.reset()
    print(observation)
    for _ in range(1000):
        action = env.action_space.sample()
        print("action: ", action)
        observation_space, reward, done, truncated, info = env.step(action)
        print(observation)
        env.render()
    env.close()


if __name__ == "__main__":
    test_graph_navigation_env()
    plt.ion()
