# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>
import os
import sys

import numpy as np
import yaml

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
    observation = env._get_observation()

    # Check the properties of the observation space
    # Check if the observation is in the expected format
    assert "num_agents" in observation
    assert "num_passengers" in observation
    assert "agent_positions" in observation
    assert "passenger_positions" in observation
    assert "passenger_destinations" in observation

    print("Observation test passed!")


def test_actor_network():
    # Create a sample ride-sharing environment
    # Load training parameters from ppo_config.yml file
    with open("src/training/ppo_config.yml", "r") as f:
        config = yaml.safe_load(f)
    map_area = "Piedmont, California, USA"
    env = RideShareEnv(map_area)

    # Create an AICoordinator instance
    coordinator = AICoordinator(env, config)
    env.coordinator = coordinator
    # Reset the environment
    env.reset()

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
        passenger_id=2,
        pickup_location={"latitude": 37.824789, "longitude": -122.231234},
        destination={"latitude": 37.827234, "longitude": -122.235678},
    )
    env.add_passenger(passenger2)

    # Get the initial observation
    observation = env._get_observation()

    # Check if the observation is in the expected format
    assert "num_agents" in observation
    assert "num_passengers" in observation
    assert "agent_positions" in observation
    assert "passenger_positions" in observation
    assert "passenger_destinations" in observation

    # Get a sample action from the coordinator
    action = coordinator.get_action(observation)

    # Check if the action is in the expected format
    assert isinstance(action, np.ndarray)
    assert action.shape[0] <= coordinator.max_agents
    assert action.shape[1] == 2

    # Check if the action values are within the valid range
    assert np.all(action >= -1) and np.all(action <= 1)

    # Check if the actions are correctly mapped to OSM nodes
    for taxi, action_value in zip(env.taxi_agents, action):
        # Find the closest node based on the action value
        closest_node_id = None
        min_distance = float("inf")
        for normalized_action, node_id in env.action_to_node_mapping.items():
            distance = np.sqrt(
                (action_value[0] - normalized_action[0]) ** 2
                + (action_value[1] - normalized_action[1]) ** 2
            )
            if distance < min_distance:
                min_distance = distance
                closest_node_id = node_id

        # Check if the closest node is a valid OSM node
        assert closest_node_id in env.map_network.nodes()
