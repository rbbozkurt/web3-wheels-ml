# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from gymnasium import Env

from agents import Passenger, TaxiAgent


class RideShareEnv(Env):
    def __init__(self, map_area="Piedmont, California, USA"):
        """
        Initialize the ride-sharing environment.
        - Download the street network data using OSMnx
        - Convert the data into a graph representation (nodes and edges)
        - Define the observation and action spaces
        """
        self.map_network = ox.graph_from_place(map_area, network_type="drive")
        self.taxi_agents = []
        self.passengers = []

    def add_agent(self, agent: TaxiAgent):
        """
        Gets called when Owner adds car from frontend
        - Add a new agent to the environment.
        - Uses agent's position to assign closest node on graph
        """
        closest_node = ox.distance.nearest_nodes(
            self.map_network, agent.position["longitude"], agent.position["latitude"]
        )
        agent.position = closest_node
        self.taxi_agents.append(agent)

    def remove_taxi_agent(self, agent: TaxiAgent):
        self.taxi_agents.remove(agent)

    def add_passenger(self, passenger: Passenger):
        """
        Gets called when Owner adds passenger from frontend
        - Add a new passenger to the environment.
        - Uses passenger's position to assign closest node on graph
        """
        # TODO: Check that passenger destination is not equal to pickup location
        closest_node = ox.distance.nearest_nodes(
            self.map_network,
            passenger.pickup_location["longitude"],
            passenger.pickup_location["latitude"],
        )
        passenger.position = closest_node
        self.passengers.append(passenger)

    def remove_passenger(self, passenger: Passenger):
        """
        Passenger is removed from environment when dropped off
        """

        self.passengers.remove(passenger)

    def reset(self):
        """
        Reset the environment to its initial state.
        - Reset agent positions, fuel levels, etc.
        - Generate a new ride request (pickup and drop-off nodes)
        - Return the initial observation
        """
        pass

    def step(self, timestep):
        """
        Perform one time step in the environment
        - Update agent positions based on their actions
        - Check for ride completion, collisions, or other events
        - Calculate and return rewards, next observations, done flags, and info
        """
        for agent in self.agents:
            agent.action_move(timestep)

        # Check for passenger pickup and drop-off events
        for agent in self.agents:
            for passenger in self.passengers:
                if (
                    agent.position == passenger.position
                    and not passenger.is_picked_up()
                ):
                    # Passenger pickup event
                    agent.action_pickup(passenger)
                    passenger.set_picked_up(True)
                elif (
                    agent.position == passenger.destination and passenger.is_picked_up()
                ):
                    # Passenger drop-off event
                    agent.action_dropoff(passenger)
                    passenger.set_completed(True)

        # Calculate rewards based on passenger waiting time and ride completion
        rewards = self._calculate_rewards()

        # Get the next observation
        next_observation = self._get_observation()

        # Check if the episode is done
        done = self._is_done()

        # Provide additional information if needed
        info = {}

        return next_observation, rewards, done, info

    def get_route(self, start_node, end_node):
        """
        Provides Agent with shortest path between two nodes
        - Uses networkx to find shortest path
        - Can be modified to include traffic data
        """
        return nx.shortest_path(self.map_network, start_node, end_node)

    def get_path_distance(self, path):
        """
        Gets the total distance of path
        """
        edge_lengths = ox.routing.route_to_gdf(self.map_network, path)["length"]
        total_distance = round(sum(edge_lengths))
        return total_distance

    def render(
        self, mode="human", ax=None, route=None, output_file="test_environment.png"
    ):
        """
        Render the current state of the environment.
        - Visualize the taxi_agents, ride requests, street network graph, and route (if provided)
        """
        if ax is None:
            fig, ax = plt.subplots()

        # Plot the street network graph
        ox.plot_graph(
            self.map_network,
            node_size=0,
            bgcolor="white",
            edge_linewidth=0.5,
            edge_color="gray",
            ax=ax,
        )

        # Plot the route (if provided)
        if route is not None:
            ox.plot_graph_route(self.map_network, route, node_size=0, ax=ax)

        # Plot taxi_agents on the map
        for agent in self.taxi_agents:
            node = self.map_network.nodes[agent.position]
            x, y = node["x"], node["y"]
            ax.scatter(x, y, color="blue", marker="o", s=50, label="Agent")

        # Plot passengers on the map
        for passenger in self.passengers:
            node = self.map_network.nodes[passenger.position]
            x, y = node["x"], node["y"]
            ax.scatter(x, y, color="green", marker="o", s=50, label="Passenger")

        ax.legend()

        if output_file is not None:
            # Save the plot to the specified output file
            plt.savefig(output_file)
            plt.close(fig)
