# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>

import osmnx as ox

from src.demos.envs.maps.osm_env import RideShareEnv
from src.demos.taxi_agent import Agent


def test_adding_agents():
    # Create a ride-sharing environment
    map_area = "Piedmont, California, USA"
    env = RideShareEnv(map_area)

    # Create some sample car information
    car_info_1 = {
        "name": "Car 1",
        "vin": "ABC123",
        "year": 2022,
        "mileage": 10000,
        "fuel": 50,
        "model": "Sedan",
        "position": {"latitude": 37.824454, "longitude": -122.231589},
    }

    car_info_2 = {
        "name": "Car 2",
        "vin": "XYZ789",
        "year": 2021,
        "mileage": 15000,
        "fuel": 60,
        "model": "SUV",
        "position": {"latitude": 37.821592, "longitude": -122.234797},
    }

    # Create agent instances using the car information
    agent_1 = Agent(env, car_info_1)
    agent_2 = Agent(env, car_info_2)

    # Add the agents to the environment
    env.add_agent(agent_1)
    env.add_agent(agent_2)

    # Assert that the agents are added to the environment correctly
    assert len(env.agents) == 2
    assert agent_1 in env.agents
    assert agent_2 in env.agents

    env.render()
