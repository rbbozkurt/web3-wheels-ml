# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>


def test_coordinator_agent():
    # Create a ride-sharing environment
    map_area = "Piedmont, California, USA"
    env = RideShareEnv(map_area)

    # Create some sample car information
    car_info_1 = {
        "name": "Car 1",
        "vin": "ABC123",
        "description": "Sedan",
        "mileage_km": 10000,
        "tankCapacity": 50,
        "position": {"latitude": 37.824454, "longitude": -122.231589},
    }

    # Create agent instances using the car information
    agent_1 = TaxiAgent(env, car_info_1)

    # Add the agents to the environment
    env.add_agent(agent_1)

    # Set a destination for the agent #TODO: This is manual setting of agent destination. use AICoordinator to obtain destinations
    destination = ox.distance.nearest_nodes(env.map_network, -122.234797, 37.821592)
    agent_1.set_destination(destination)
