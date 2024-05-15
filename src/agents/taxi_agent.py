# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>

from envs import RideShareEnv


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
        self.position = {
            "longitude": carInfo["position"]["longitude"],
            "latitude": carInfo["position"]["latitude"],
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

        self.destination = []  # Destination of Agent at any given time
        self.path = []  # Path of Agent at any given time
        self.env = env  # Environment in which the agent is operating

    def set_destination(self, destination):
        """
        Set the destination for the agent.
        This can be used to set the passenger pickup destination, drop off destination,
        or some other destination by a central planning agent.
        """
        self.destination.append(destination)

    def get_observation(self):
        """
        Get the current observation for the agent based on the environment state.
        - Observation could include agent's position, nearby agents, traffic conditions, etc.
        """
        # Get the agent's current position (node) in the environment
        current_position = self.env.get_agent_position(self)

        # Get nearby agents within a certain radius
        nearby_agents = self.env.get_nearby_agents(
            self, radius=1.0
        )  # Adjust the radius as needed

        # Create the observation dictionary
        observation = {
            "position": current_position,
            "nearby_agents": nearby_agents,
        }

        return observation

    def get_action(self, observation):
        """
        Get the action for the agent based on its current observation (to be learned by the RL algorithm).
        - Action could be moving to an adjacent node, staying put, etc.
        """
        # Make a function for each action. This get_action function will choose from possible actions

        pass

    def action_move(self):
        """
        Moves agent towards the destination based on the agent's speed and the time step.
        """
        # Set the time step (in hours)
        time_step = 0.5  # Each step represents x hours

        # Calculate the distance the agent can travel in one time step
        average_speed = 30  # km/h #TODO use speed limit of road as speed
        distance_per_step = average_speed * time_step

        if len(self.destination) > 0:
            destination = self.destination[0]  # Get the first destination in the list

            # Get the shortest path from the agent's current position to the destination
            path = self.env.get_route(self.position, destination)
            self.path = path
            # Calculate the total distance of the path
            total_distance = self.env.get_path_distance(path)

            # Check if the agent can reach the destination within the current time step
            if total_distance <= distance_per_step:
                # Agent can reach the destination in this time step
                self.position = destination
                self.distance_from_node = 0  # Agent is at the destination node
                self.update_agent_position(destination)
                self.destination.pop(0)  # Remove the reached destination from the list
            else:
                # Agent cannot reach the destination in this time step
                # Move the agent along the path, to next node, based on the distance per step
                current_node = self.position  # same as path[0]
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

    def update_agent_position(self, position, distance_from_node=0):
        self.position = position
        self.distance_from_node = distance_from_node

    def action_pickup(self, agent, passenger):
        """
        Checks if agent is in same node as passenger.
        If so, passenger is picked up and added to
        agent's list of passengers
        """
        if agent.position == passenger.position:
            # Add the passenger to the agent's list of passengers
            agent.passengers.append(passenger)

            # Remove the passenger from the environment
            RideShareEnv.remove_passenger(passenger)

    def action_dropoff(self, agent, passenger):
        """
        Check if agent is in same node as passenger's destination.
        If so, passenger is dropped off and removed from
        agent's list of passengers
        """
        if agent.position == passenger.destination:
            # Remove the passenger from the agent's list of passengers
            agent.passengers.remove(passenger)

            # Mark the passenger as dropped off in the environment
            RideShareEnv.mark_passenger_dropped_off(passenger)

    def update(self, action, reward, next_observation):
        """
        Update the agent's state and policy based on the action taken, reward received, and next observation.
        - This function will be called by the RL algorithm during training
        """
        RideShareEnv.step(
            self, action, reward, next_observation
        )  # Update the environment
        pass
