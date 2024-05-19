# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
from gymnasium import Env, spaces

from agents import Passenger, TaxiAgent, reward_function_basic


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
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

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

    def _get_observation_space(self):
        # Define the observation space based on your problem
        # Include the positions of agents and passengers, and the destinations of passengers

        num_agents = len(self.taxi_agents)
        num_passengers = len(self.passengers)

        observation_space = spaces.Dict(
            {
                "num_agents": spaces.Discrete(num_agents + 1),
                "num_passengers": spaces.Discrete(num_passengers + 1),
                "agent_positions": spaces.Box(
                    low=np.array([[-180, -90]] * num_agents),
                    high=np.array([[180, 90]] * num_agents),
                    dtype=np.float32,
                ),
                "passenger_positions": spaces.Box(
                    low=np.array([[-180, -90]] * num_passengers),
                    high=np.array([[180, 90]] * num_passengers),
                    dtype=np.float32,
                ),
                "passenger_destinations": spaces.Box(
                    low=np.array([[-180, -90]] * num_passengers),
                    high=np.array([[180, 90]] * num_passengers),
                    dtype=np.float32,
                ),
                # Add other relevant observations
            }
        )
        return observation_space

    def _get_action_space(self):
        # Define the action space based on your problem
        # Example: Selecting a destination node for each agent
        max_nodes = 1000  # Adjust this value based on your requirements
        action_space = spaces.Discrete(max_nodes)
        return action_space

    def _get_observation(self):
        observation = {
            "num_agents": len(self.taxi_agents),
            "num_passengers": len(self.passengers),
            "agent_positions": np.array(
                [list(agent.position.values()) for agent in self.taxi_agents]
            ),
            "passenger_positions": np.array(
                [
                    list(passenger.pickup_location.values())
                    for passenger in self.passengers
                ]
            ),
            "passenger_destinations": np.array(
                [
                    list(passenger.dropoff_location.values())
                    for passenger in self.passengers
                ]
            ),
            # Add other relevant observations
        }
        return observation

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
        - Generate new passengers with pickup and dropoff locations
        - Return the initial observation
        """
        # Reset taxi agents
        self.taxi_agents = []
        # Reset passengers
        self.passengers = []
        # Create the initial observation
        observation = self._get_observation()
        return observation

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
        rewards = reward_function_basic(self)

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
