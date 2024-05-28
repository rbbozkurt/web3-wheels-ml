# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>

import osmnx as ox


class TaxiAgent:
    def __init__(self, env, carInfo):
        """
        Initialize an agent with its properties (car type, tankCapacity level, etc.).
        - Set the initial position of the agent on the graph
        - Backend will initialize each agent with Info from User (or fetched from an API using VIN)
        """
        self.name = carInfo["name"]
        self.description = carInfo["description"]
        self.vin = carInfo["vin"]
        closest_node = ox.distance.nearest_nodes(
            env.map_network,
            carInfo["position"]["longitude"],
            carInfo["position"]["latitude"],
        )
        # How to handle osm nodes and gps coords:
        #    ML model uses gps coords as observation and output destination. (Convert node to gps if needed)
        #   Convert to node using nearest node for position and osm navigation
        self.position = {
            "longitude": carInfo["position"]["longitude"],
            "latitude": carInfo["position"]["latitude"],
            "node": closest_node,
        }
        self.mileage_km = carInfo.get(
            "mileage_km", 0.1
        )  # Default to 0.1 if not provided
        self.tankCapacity = carInfo.get(
            "tankCapacity", 50
        )  # Default to 50 if not provided
        self.currentFuel = self.tankCapacity
        self.reputation = carInfo.get("reputation", 90)  # from 1-100. Start at 90

        self.distance_from_node = 0  # distance to closest node when between nodes
        self.passengers = []  # list of passengers in car

        self.destination = []  # Destination of Agent NODE ID
        self.path = []  # Path of Agent at any given time
        self.env = env  # Environment in which the agent is operating

    def set_destination(self, node):
        """
        Set the destination for the agent.
        This can be used to set the passenger pickup destination, drop off destination,
        or some other destination by a central planning agent.
        """
        self.destination = node

    def action_move(self, timestep=0.5):
        """
        Moves agent towards the destination based on the agent's speed and the time step.

        """

        # Calculate the distance the agent can travel in one time step
        average_speed = 30  # km/h #TODO use speed limit of road as speed
        distance_per_step = average_speed * timestep

        if self.destination:
            destination = self.destination  # Get the first destination in the list

            if self.position["node"] == destination:
                # If the current position is the same as the destination, no movement is needed
                print("Already at the destination")
                self.destination = None  # Reset the destination
                return

            # Get the shortest path from the agent's current position to the destination
            path = self.env.get_route(self.position["node"], destination)
            if path is None:
                # Handle the case when no route is found
                print("No route found")
                self.destination = None  # Reset the destination
                return
            else:
                # Process the route
                distance = self.env.get_path_distance(path)
                # print(f"Route found with distance {distance}")
                self.path = path
            # Calculate the total distance of the path
            total_distance = self.env.get_path_distance(path)

            # Check if the agent can reach the destination within the current time step
            if total_distance <= distance_per_step:
                # Agent can reach the destination in this time step
                self.position["node"] = destination
                self.distance_from_node = 0  # Agent is at the destination node
                self.update_agent_position(destination)
            else:
                # Agent cannot reach the destination in this time step
                # Move the agent along the path, to next node, based on the distance per step
                current_node = self.position["node"]  # same as path[0]
                next_node = path[1]
                edge_data = self.env.map_network.get_edge_data(current_node, next_node)
                edge_distance = edge_data[0]["length"]
                remaining_distance = edge_distance - self.distance_from_node

                if distance_per_step >= remaining_distance:
                    # Agent can move to the next node
                    self.update_agent_position(next_node)
                else:
                    # Agent cannot reach the next node, stop at the current position
                    self.distance_from_node += distance_per_step
                    self.update_agent_position(current_node, self.distance_from_node)

        else:
            # If no destination is set, the agent stays still
            pass

    def update_agent_position(self, node, distance_from_node=0):
        self.position["node"] = node
        self.position["longitude"] = self.env.map_network.nodes[node]["x"]
        self.position["latitude"] = self.env.map_network.nodes[node]["y"]
        self.distance_from_node = distance_from_node

    def action_pickup(self, passenger):
        """
        Checks if agent is in same node as passenger.
        If so, passenger is picked up and added to
        agent's list of passengers
        """
        if self.position["node"] == passenger.position["node"]:
            # Add the passenger to the agent's list of passengers
            self.passengers.append(passenger)
        self.set_destination(passenger.destination["node"])
        passenger.set_picked_up(True)
        passenger.which_taxi = self.name
        print(self.name, "picked up passenger", passenger.passenger_id)

    def action_dropoff(self, passenger):
        """
        Check if agent is in same node as passenger's destination.
        If so, passenger is dropped off and removed from
        agent's list of passengers
        """
        if self.position["node"] == passenger.destination["node"]:
            # Remove the passenger from the agent's list of passengers
            self.passengers.remove(passenger)

            # Mark the passenger as dropped off in the environment
            self.env.remove_passenger(passenger)
            passenger.set_completed(True)
            print(self.name, "dropped off passenger", passenger.passenger_id)
            self.destination = None  # Reset the destination
