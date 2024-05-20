# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>
from envs import RideShareEnv


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

    # Optionally, you can compare the bounds with expected values
    # based on the specific map area used for testing
    expected_bounds = (37.8159, -122.2442, 37.8343, -122.2202)
    assert bounds == expected_bounds

    print("get_map_bounds test passed!")
