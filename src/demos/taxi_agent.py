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

        RideShareEnv.add_agent(self)  # Add agent to the environment

    def get_observation(self, env):
        """
        Get the current observation for the agent based on the environment state.
        - Observation could include agent's position, nearby agents, traffic conditions, etc.
        """

        pass

    def get_action(self, observation):
        """
        Get the action for the agent based on its current observation (to be learned by the RL algorithm).
        - Action could be moving to an adjacent node, staying put, etc.
        """
        # Make a function for each action. This get_action function will choose from possible actions

        pass

    def action_move(self, agent, destination):
        """
        Moves agent one step, equal to 0.1 km on graph?

        """

    def action_pickup(self, agent, passenger):
        """
        Checks if agent is in same node as passenger. If so, passenger is picked up and added to agent's list of passengers
        """

    def action_dropoff(self, agent, passenger):
        """
        Check if agent is in same node as passenger's destination. If so, passenger is dropped off and removed from agent's list of passengers
        """

    def update(self, action, reward, next_observation):
        """
        Update the agent's state and policy based on the action taken, reward received, and next observation.
        - This function will be called by the RL algorithm during training
        """
        RideShareEnv.step(
            self, action, reward, next_observation
        )  # Update the environment
        pass
