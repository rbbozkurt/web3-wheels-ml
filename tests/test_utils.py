# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.agents.passenger import Passenger
from src.agents.taxi_agent import TaxiAgent
from src.envs.osm_env import RideShareEnv


def test_get_map_bounds():
    # Create an instance of the RideShareEnv with a specific map area
    map_area = "Piedmont, California, USA"
    env = RideShareEnv(map_area)

    # Call the get_map_bounds method
    bounds = env.get_map_bounds()

    # Check if the bounds are returned as expected
    assert len(bounds) == 4
    assert all(isinstance(coord, float) for coord in bounds)

    # Check if the bounds are in the correct order
    min_lat, min_lng, max_lat, max_lng = bounds
    assert min_lat < max_lat
    assert min_lng < max_lng

    # Print the bounds for visual inspection
    print("Map Bounds:")
    print(f"  Min Latitude: {min_lat}")
    print(f"  Min Longitude: {min_lng}")
    print(f"  Max Latitude: {max_lat}")
    print(f"  Max Longitude: {max_lng}")

    print("get_map_bounds test passed!")
