# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
from gymnasium import Env, spaces

from ..agents import (
    AICoordinator,
    Passenger,
    TaxiAgent,
    reward_function_2,
    reward_function_basic,
)


class RideShareEnv(Env):
    def __init__(
        self, config, map_area="Piedmont, California, USA", max_time_steps=500
    ):
        """
        Initialize the ride-sharing environment.
        - Download the street network data using OSMnx
        - Convert the data into a graph representation (nodes and edges)
        - Define the observation and action spaces
        """
        self.map_network = ox.graph_from_place(map_area, network_type="drive")
        self.taxi_agents = []
        self.passengers = []
        self.max_agents = config["max_taxis"]
        self.max_passengers = config["max_passengers"]

        self.action_space = self._get_action_space()
        self.map_bounds = self.get_map_bounds()
        self.action_to_node_mapping = self._create_action_to_node_mapping()
        self.current_time_step = 0
        self.max_time_steps = max_time_steps
        self.observation_space = self._get_observation_space()
        self.coordinator = []

    def add_agent(self, agent: TaxiAgent):
        """
        Gets called when Owner adds car from frontend
        - Add a new agent to the environment.
        - Uses agent's position to assign closest node on graph
        """
        # closest_node = ox.distance.nearest_nodes(
        #     self.map_network, agent.position["longitude"], agent.position["latitude"]
        # )
        # agent.position = closest_node #TODO: Understand when to use node_id and when to use geolocation
        self.taxi_agents.append(agent)

    def _is_done(self):
        # Check if the maximum time step limit has been reached
        if self.current_time_step >= self.max_time_steps:
            return True

        # Check if all passengers have reached their destinations
        for passenger in self.passengers:
            if not passenger.completed:
                return False

        return False

    def _get_observation_space(self):
        observation_space = spaces.Dict(
            {
                "num_agents": spaces.Box(
                    low=0, high=self.max_agents, shape=(1,), dtype=np.int32
                ),
                "num_passengers": spaces.Box(
                    low=0, high=self.max_passengers, shape=(1,), dtype=np.int32
                ),
                "agent_positions": spaces.Box(
                    low=-180, high=180, shape=(self.max_agents, 2), dtype=np.float32
                ),
                "passenger_positions": spaces.Box(
                    low=-180, high=180, shape=(self.max_passengers, 2), dtype=np.float32
                ),
                "passenger_destinations": spaces.Box(
                    low=-180, high=180, shape=(self.max_passengers, 2), dtype=np.float32
                ),
                # Add other relevant observations based on TaxiAgent and Passenger properties
            }
        )
        return observation_space

    def get_map_bounds(self):
        # Get the bounding box of the map network
        nodes, edges = ox.utils_graph.graph_to_gdfs(self.map_network)
        bounds = nodes.total_bounds
        return bounds

    def _get_action_space(self):
        return spaces.Box(low=-1, high=1, shape=(self.max_agents, 2), dtype=float)

    def _create_action_to_node_mapping(self):
        action_to_node_mapping = {}
        nodes = list(self.map_network.nodes())
        num_nodes = len(nodes)

        # Get the longitude and latitude coordinates of each node
        node_coords = np.array(
            [
                (self.map_network.nodes[node]["x"], self.map_network.nodes[node]["y"])
                for node in nodes
            ]
        )

        # Normalize the node coordinates to the range [-1, 1] based on the map bounds
        normalized_coords = (node_coords - self.map_bounds[[0, 1]]) / (
            self.map_bounds[[2, 3]] - self.map_bounds[[0, 1]]
        ) * 2 - 1

        # Create the action-to-node mapping using the normalized coordinates
        for i in range(num_nodes):
            action_to_node_mapping[tuple(normalized_coords[i])] = nodes[i]

        return action_to_node_mapping

    def _get_observation(self):
        num_agents = len(self.taxi_agents)
        num_passengers = len(self.passengers)

        # Filter taxi agents without passengers
        taxi_agents_without_passengers = [
            agent for agent in self.taxi_agents if not agent.passengers
        ]
        num_agents_without_passengers = len(taxi_agents_without_passengers)

        agent_positions = np.zeros((self.observation_space["agent_positions"].shape))
        for i, agent in enumerate(taxi_agents_without_passengers):
            agent_positions[i] = np.array(
                [agent.position["longitude"], agent.position["latitude"]]
            )

        # Filter passengers not picked up
        passengers_not_picked_up = [
            passenger for passenger in self.passengers if not passenger.picked_up
        ]
        num_passengers_not_picked_up = len(passengers_not_picked_up)

        passenger_positions = np.zeros(
            (self.observation_space["passenger_positions"].shape)
        )
        for i, passenger in enumerate(passengers_not_picked_up):
            if i >= num_passengers_not_picked_up:
                break
            pickup_location = passenger.pickup_location
            passenger_positions[i] = np.array(
                [pickup_location["longitude"], pickup_location["latitude"]]
            )

        passenger_destinations = np.zeros(
            (self.observation_space["passenger_destinations"].shape)
        )
        for i, passenger in enumerate(passengers_not_picked_up):
            if i >= num_passengers_not_picked_up:
                break
            destination = passenger.destination
            passenger_destinations[i] = np.array(
                [destination["longitude"], destination["latitude"]]
            )

        # Input observation to model. Do not change keys
        observation = {
            "num_agents": np.array([num_agents_without_passengers]),
            "num_passengers": np.array([num_passengers_not_picked_up]),
            "agent_positions": agent_positions,
            "passenger_positions": passenger_positions,
            "passenger_destinations": passenger_destinations,
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
        closest_node_pickup = ox.distance.nearest_nodes(
            self.map_network,
            passenger.pickup_location["longitude"],
            passenger.pickup_location["latitude"],
        )
        passenger.position["node"] = closest_node_pickup
        closest_node_destination = ox.distance.nearest_nodes(
            self.map_network,
            passenger.destination["longitude"],
            passenger.destination["latitude"],
        )
        passenger.destination["node"] = closest_node_destination
        self.passengers.append(passenger)

    def remove_passenger(self, passenger: Passenger):
        """
        Passenger is removed from environment when dropped off
        """

        self.passengers.remove(passenger)

    def reset(self, map_area="Piedmont, California, USA"):
        """
        Reset the environment to its initial state.
        - Reset agent positions, fuel levels, etc.
        - Generate new passengers with pickup and dropoff locations
        - Return the initial observation
        """

        self.map_network = ox.graph_from_place(map_area, network_type="drive")
        self.taxi_agents = []
        self.passengers = []
        self.action_space = self._get_action_space()
        self.map_bounds = self.get_map_bounds()
        self.action_to_node_mapping = self._create_action_to_node_mapping()
        self.current_time_step = 0
        self.observation_space = self._get_observation_space()

    def step(self, time_interval=0.5):
        """
        Perform one time step in the environment
        - Update agent positions based on their actions
        - Check for ride completion, collisions, or other events
        - Calculate and return rewards, next observations, done flags, and info
        """
        for taxi in self.taxi_agents:
            taxi.action_move(time_interval)

        # Check for passenger pickup and drop-off events
        for taxi in self.taxi_agents:
            for passenger in self.passengers:
                if (
                    taxi.position["node"] == passenger.position["node"]
                    and not passenger.is_picked_up()
                ):
                    # Passenger pickup event
                    if not taxi.passengers:  # If the taxi does not have a passenger
                        taxi.action_pickup(passenger)

                elif (
                    taxi.position["node"] == passenger.destination["node"]
                    and passenger.which_taxi == taxi.name
                ):
                    # Passenger drop-off event
                    taxi.action_dropoff(passenger)
                    passenger.set_completed(True)

        # Get the next observation
        next_observation = self._get_observation()

        # Filter taxi agents without passengers
        taxi_agents_without_passengers = [
            agent for agent in self.taxi_agents if not agent.passengers
        ]

        # Check if any taxi does not have passengers
        if taxi_agents_without_passengers:
            # Call the AICoordinator to get the actions for the taxi agents
            actions = self.coordinator.get_action(next_observation)

            # Assign destinations to taxi agents based on the actions
            for taxi, action in zip(taxi_agents_without_passengers, actions):
                # Find the closest node based on the continuous action value
                closest_node_id = None
                min_distance = float("inf")
                for normalized_action, node_id in self.action_to_node_mapping.items():
                    distance = np.sqrt(
                        (action[0] - normalized_action[0]) ** 2
                        + (action[1] - normalized_action[1]) ** 2
                    )
                    if distance < min_distance:
                        min_distance = distance
                        closest_node_id = node_id
                destination = {
                    "node": closest_node_id,
                    "longitude": self.map_network.nodes[closest_node_id]["x"],
                    "latitude": self.map_network.nodes[closest_node_id]["y"],
                }
                taxi.set_destination(destination["node"])
        else:
            actions = None

        # Calculate rewards based on passenger waiting time and ride completion
        rewards = reward_function_2(self)

        # Check if the episode is done
        done = self._is_done()
        self.current_time_step += 1
        # print("Time step: ", self.current_time_step)

        # Provide additional information if needed
        info = {}

        return next_observation, actions, rewards, done, info

    def get_route(self, start_node, end_node):
        """
        Provides Agent with shortest path between two nodes
        - Uses networkx to find shortest path
        - Can be modified to include traffic data
        """
        try:
            route = nx.shortest_path(self.map_network, start_node, end_node)
            return route
        except nx.NetworkXNoPath:
            # No path exists between the start and end nodes
            return None

    def get_path_distance(self, path):
        """
        Gets the total distance of path
        """
        if len(path) == 1:
            # If the path consists of a single node, the distance is 0
            return 0
        else:
            edge_lengths = ox.routing.route_to_gdf(self.map_network, path)["length"]
            total_distance = round(sum(edge_lengths))
            return total_distance

    def render(self, mode="human", ax=None, output_file="test_environment.png"):
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

        # Plot taxi_agents on the map
        for taxi in self.taxi_agents:
            node = self.map_network.nodes[taxi.position["node"]]
            x, y = node["x"], node["y"]
            if taxi.passengers:  # If the taxi has a passenger, plot it in red
                ax.scatter(x, y, color="red", marker="o", s=30, label="TaxiAgent")
            else:  # Otherwise, plot it in blue
                ax.scatter(x, y, color="blue", marker="o", s=30, label="TaxiAgent")

            # Plot the path for the taxi agent if it has a destination
            if taxi.destination:
                path = self.get_route(taxi.position["node"], taxi.destination)
                if path:
                    if taxi.passengers:  # If the taxi has a passenger, plot it in red
                        ox.plot_graph_route(
                            self.map_network,
                            path,
                            node_size=0,
                            ax=ax,
                            route_linewidth=3,
                            route_alpha=0.3,
                            orig_dest_size=1,
                            route_color="red",
                        )
                    else:
                        ox.plot_graph_route(
                            self.map_network,
                            path,
                            node_size=0,
                            ax=ax,
                            route_linewidth=3,
                            route_alpha=0.3,
                            orig_dest_size=1,
                            route_color="blue",
                        )

        # Plot passengers on the map
        for passenger in self.passengers:
            if not passenger.picked_up:
                node = self.map_network.nodes[passenger.position["node"]]
                x, y = node["x"], node["y"]
                ax.scatter(x, y, color="green", marker="o", s=50, label="Passenger")

        # ax.legend()

        if output_file is not None:
            # Save the plot to the specified output file
            plt.savefig(output_file)
            plt.close(fig)
