# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.passenger import Passenger
from src.agents.taxi_agent import TaxiAgent
from src.envs.osm_env import RideShareEnv


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
    env.add_agent(agent)
    # Create a sample passenger
    passenger = Passenger(
        passenger_id=1,
        pickup_location={"latitude": 37.824454, "longitude": -122.231589},
        destination={"latitude": 37.821592, "longitude": -122.234797},
    )
    env.add_passenger(passenger)
    # Test pickup action
    assert passenger not in agent.passengers
    agent.action_pickup(passenger)
    assert passenger in agent.passengers
    assert agent.destination == passenger.destination["node"]
    assert passenger.picked_up == True
    # Test dropoff action
    agent.position["node"] = passenger.destination["node"]
    agent.action_dropoff(passenger)
    assert passenger not in agent.passengers
    assert passenger not in env.passengers
