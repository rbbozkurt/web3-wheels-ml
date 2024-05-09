# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>

import gymnasium as gym
import networkx as nx
import osmx as ox
from gymnasium import Env, spaces, utils


class RideShareEnv(Env):
    def __init__(self, map_area="Piedmont, California, USA"):
        """
        Initialize the ride-sharing environment.
        - Download the street network data using OSMnx
        - Convert the data into a graph representation (nodes and edges)
        - Define the observation and action spaces
        """
        self.map_network = ox.graph_from_place(map_area, network_type="drive")

        pass

    def add_agent(self, agent):
        """
        Gets called when Owner adds car from frontend
        - Add a new agent to the environment.
        - Uses agent's location to assign closes node on graph
        """
        pass

    def reset(self):
        """
        Reset the environment to its initial state.
        - Reset agent positions, fuel levels, etc.
        - Generate a new ride request (pickup and drop-off nodes)
        - Return the initial observation
        """
        pass

    def step(self, actions):
        """
        Perform one step in the environment based on the agents' actions.
        - Update agent positions based on their actions
        - Check for ride completion, collisions, or other events
        - Calculate and return rewards, next observations, done flags, and info
        """
        pass

    def get_route(self, start_node, end_node):
        """
        Provides Agent with shortest path between two nodes
        - Uses networkx to find shortest path
        - Can be modified to include traffic data
        """
        pass

    def render(self, mode="human"):
        """
        Render the current state of the environment.
        - Visualize the agents, ride requests, and street network graph
        """
        # fig, ax = ox.plot_graph(G)
        pass
