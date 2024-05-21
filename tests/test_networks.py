# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>
import os
import sys

import numpy as np
from gymnasium import spaces

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.coordinator_agent import AICoordinator
from src.agents.passenger import Passenger
from src.agents.taxi_agent import TaxiAgent
from src.envs.osm_env import RideShareEnv


def test_observation_space():
    # Create a sample ride-sharing environment
    map_area = "Piedmont, California, USA"
    env = RideShareEnv(map_area)

    # Add sample taxi agents to the environment
    taxi_info1 = {
        "name": "Taxi 1",
        "vin": "VIN1",
        "description": "Sample Taxi 1",
        "mileage_km": 5000,
        "tankCapacity": 50,
        "position": {"latitude": 37.824454, "longitude": -122.231589},
    }
    taxi_agent1 = TaxiAgent(env, taxi_info1)
    env.add_agent(taxi_agent1)

    taxi_info2 = {
        "name": "Taxi 2",
        "vin": "VIN2",
        "description": "Sample Taxi 2",
        "mileage_km": 8000,
        "tankCapacity": 60,
        "position": {"latitude": 37.825123, "longitude": -122.232456},
    }
    taxi_agent2 = TaxiAgent(env, taxi_info2)
    env.add_agent(taxi_agent2)

    # Add sample passengers to the environment
    passenger1 = Passenger(
        passenger_id=1,
        pickup_location={"latitude": 37.823567, "longitude": -122.230987},
        destination={"latitude": 37.828901, "longitude": -122.234567},
    )

    env.add_passenger(passenger1)

    passenger2 = Passenger(
        passenger_id=1,
        pickup_location={"latitude": 37.824789, "longitude": -122.231234},
        destination={"latitude": 37.828901, "longitude": -122.234567},
    )

    env.add_passenger(passenger2)

    # Get the observation space
    observation_space = env.observation_space

    # Check the properties of the observation space
    assert "num_agents" in observation_space.spaces
    assert "num_passengers" in observation_space.spaces
    assert "agent_positions" in observation_space.spaces
    assert "passenger_positions" in observation_space.spaces
    assert "passenger_destinations" in observation_space.spaces

    # Check the data types and shapes of the observation components
    assert isinstance(observation_space["num_agents"], spaces.Discrete)
    assert observation_space["num_agents"].n == 3  # 2 agents + 1

    assert isinstance(observation_space["num_passengers"], spaces.Discrete)
    assert observation_space["num_passengers"].n == 3  # 2 passengers + 1

    assert isinstance(observation_space["agent_positions"], spaces.Box)
    assert observation_space["agent_positions"].shape == (2, 2)

    assert isinstance(observation_space["passenger_positions"], spaces.Box)
    assert observation_space["passenger_positions"].shape == (2, 2)

    assert isinstance(observation_space["passenger_destinations"], spaces.Box)
    assert observation_space["passenger_destinations"].shape == (2, 2)

    print("Observation space test passed!")
