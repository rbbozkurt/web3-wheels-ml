# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>
import numpy as np

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


def test_pickup_dropoff():
    # Create a sample ride-sharing environment
    map_area = "Piedmont, California, USA"
    env = RideShareEnv(map_area)

    # Create a sample taxi agent
    car_info = {
        "name": "Car 1",
        "vin": "ABC123",
        "description": "Sedan",
        "mileage_km": 10000,
        "tankCapacity": 50,
        "position": {"latitude": 37.824454, "longitude": -122.231589},
    }
    agent = TaxiAgent(env, car_info)

    # Create a sample passenger
    passenger = Passenger(
        passenger_id=1,
        pickup_location={"latitude": 37.824454, "longitude": -122.231589},
        dropoff_location={"latitude": 37.821592, "longitude": -122.234797},
    )

    # Test pickup action
    assert passenger not in agent.passengers
    agent.action_pickup(passenger)
    assert passenger in agent.passengers
    assert agent.destination == passenger.destination

    # Test dropoff action
    agent.position = passenger.destination
    agent.action_dropoff(passenger)
    assert passenger not in agent.passengers
    assert passenger not in env.passengers

    # Test pickup and dropoff with multiple passengers
    passenger2 = Passenger(
        passenger_id=2,
        pickup_location={"latitude": 37.824454, "longitude": -122.231589},
        dropoff_location={"latitude": 37.820000, "longitude": -122.235000},
    )
    agent.action_pickup(passenger)
    agent.action_pickup(passenger2)
    assert passenger in agent.passengers
    assert passenger2 in agent.passengers

    agent.position = passenger.destination
    agent.action_dropoff(passenger)
    assert passenger not in agent.passengers
    assert passenger2 in agent.passengers

    agent.position = passenger2.destination
    agent.action_dropoff(passenger2)
    assert passenger2 not in agent.passengers
    assert len(agent.passengers) == 0
