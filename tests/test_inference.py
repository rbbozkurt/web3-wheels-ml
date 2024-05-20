# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>
import numpy as np
import osmnx as ox

from agents import AICoordinator, Passenger, TaxiAgent
from envs import RideShareEnv


def test_get_action():
    # Create a sample ride-sharing environment
    map_area = "Piedmont, California, USA"
    env = RideShareEnv(map_area)

    # Create a sample AICoordinator
    coordinator = AICoordinator(env)

    # Create a sample observation
    observation = {
        "num_agents": 2,
        "num_passengers": 3,
        "agent_positions": np.array(
            [[37.824454, -122.231589], [37.821592, -122.234797]]
        ),
        "passenger_positions": np.array(
            [
                [37.825000, -122.232000],
                [37.823000, -122.233000],
                [37.822000, -122.235000],
            ]
        ),
        "passenger_destinations": np.array(
            [
                [37.820000, -122.235000],
                [37.819000, -122.236000],
                [37.818000, -122.237000],
            ]
        ),
    }

    # Get the action from the AICoordinator
    action = coordinator.get_action(observation)

    # Check if the action is of the expected shape and data type
    assert isinstance(action, np.ndarray)
    assert action.shape == env.action_space.shape
    assert action.dtype == env.action_space.dtype

    # Check if the action values are within the valid range
    assert np.all(action >= env.action_space.low)
    assert np.all(action <= env.action_space.high)
