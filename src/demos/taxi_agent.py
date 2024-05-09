# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>

from ..demos.envs.maps.osm_env import RideShareEnv


class Agent:
    def __init__(self, env, carInfo):
        """
        Initialize an agent with its properties (car type, fuel level, etc.).
        - Set the initial position of the agent on the graph
        - Backend will initialize each agent with Info from User (or fetched from an API using VIN)
        """
        self.name = carInfo.name
        self.vin = carInfo.vin
        self.year = carInfo.year
        self.mileage = carInfo.mileage
        self.fuel = carInfo.fuel
        self.reputation = 90  # from 1-100. Start at 90
        self.model = carInfo.model
        self.location.longitude = carInfo.location.longitude
        self.location.latitude = carInfo.location.latitude
        self.passengers = []  # list of passengers in car
        self.average_speed = 10  # km/h
        RideShareEnv.add_agent(self)  # Add agent to the environment

    def get_observation(self, env):
        """
        Get the current observation for the agent based on the environment state.
        - Observation could include agent's position, nearby agents, traffic conditions, etc.
        """
        # Get the agent's current position (node) in the environment
        current_position = env.get_agent_position(self)

        # Get nearby agents within a certain radius
        nearby_agents = env.get_nearby_agents(
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

    def action_move(self, agent, destination):
        """
        Moves agent towards the destination based on the agent's speed and the time step.
        """
        # Set the time step (in hours)
        time_step = 0.1  # Each step represents 0.1 hours (6 minutes)

        # Calculate the distance the agent can travel in one time step
        distance_per_step = agent.average_speed * time_step

        # Get the shortest path from the agent's current position to the destination
        path = RideShareEnv.get_shortest_path(agent.location, destination)

        # Calculate the total distance of the path
        total_distance = RideShareEnv.get_path_distance(path)

        # Check if the agent can reach the destination within the current time step
        if total_distance <= distance_per_step:
            # Agent can reach the destination in this time step
            agent.location = destination
            agent.distance_from_node = 0  # Agent is at the destination node
            RideShareEnv.update_agent_position(agent, destination)
        else:
            # Agent cannot reach the destination in this time step
            # Move the agent along the path based on the distance per step
            remaining_distance = distance_per_step
            current_node = agent.location

            for i in range(1, len(path)):
                next_node = path[i]
                edge_distance = RideShareEnv.get_edge_distance(current_node, next_node)

                if remaining_distance >= edge_distance:
                    # Agent can move to the next node
                    current_node = next_node
                    remaining_distance -= edge_distance
                else:
                    # Agent cannot reach the next node, stop at the current position
                    agent.location = current_node
                    agent.distance_from_node = remaining_distance / edge_distance
                    break

            # Update the agent's position in the environment
            RideShareEnv.update_agent_position(
                agent, agent.location, agent.distance_from_node
            )

    def action_pickup(self, agent, passenger):
        """
        Checks if agent is in same node as passenger. If so, passenger is picked up and added to agent's list of passengers
        """
        if agent.location == passenger.location:
            # Add the passenger to the agent's list of passengers
            agent.passengers.append(passenger)

            # Remove the passenger from the environment
            RideShareEnv.remove_passenger(passenger)

    def action_dropoff(self, agent, passenger):
        """
        Check if agent is in same node as passenger's destination. If so, passenger is dropped off and removed from agent's list of passengers
        """
        if agent.location == passenger.destination:
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
