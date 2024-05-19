# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import osmnx as ox
from gymnasium import spaces

from agents import Passenger, TaxiAgent
from envs import RideShareEnv


def test_adding_taxi_agents():
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

    car_info_2 = {
        "name": "Car 2",
        "vin": "XYZ789",
        "description": "Sedan",
        "mileage_km": 15000,
        "fuel": 60,
        "tankCapacity": "SUV",
        "position": {"latitude": 37.821592, "longitude": -122.234797},
    }

    # Create agent instances using the car information
    agent_1 = TaxiAgent(env, car_info_1)
    agent_2 = TaxiAgent(env, car_info_2)

    # Add the agents to the environment
    env.add_agent(agent_1)
    env.add_agent(agent_2)

    # Assert that the agents are added to the environment correctly
    assert len(env.taxi_agents) == 2
    assert agent_1 in env.taxi_agents
    assert agent_2 in env.taxi_agents

    env.render()


def test_agent_movement():
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

    # Add the taxi_agents to the environment
    env.add_agent(agent_1)

    # Set a destination for the agent
    destination = ox.distance.nearest_nodes(env.map_network, -122.234797, 37.821592)
    agent_1.set_destination(destination)

    # Step through time until the agent reaches the destination
    state = {"reached_destination": False}
    num_steps = 0
    max_steps = 100  # Maximum number of steps to prevent infinite loop

    # Create a figure and axis for the animation
    fig, ax = plt.subplots()

    # Define the update function for the animation
    def update(frame):
        # Check if the agent has reached the destination
        if agent_1.position == destination:
            state["reached_destination"] = True

        else:
            # Clear the previous plot
            ax.clear()
            # Perform the agent's action for the current frame
            agent_1.action_move()

            # Render the environment for the current frame
            env.render(ax=ax, route=path, output_file=None)

            # Set the title for the current frame
            ax.set_title(f"Step: {frame}")

    # Get the shortest path from the agent's current position to the destination
    path = env.get_route(agent_1.position, destination)

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=max_steps, interval=500)

    # Save the animation as a GIF
    ani.save("agent_movement.gif", writer="pillow")

    # Close the figure
    plt.close(fig)

    # Print the number of steps taken to reach the destination
    print(f"Agent reached the destination in {num_steps} steps")

    # Assert that the agent reached the destination
    assert state[
        "reached_destination"
    ], "Agent did not reach the destination within the maximum number of steps"


def test_adding_passengers():
    map_area = "Piedmont, California, USA"
    env = RideShareEnv(map_area)

    passenger = Passenger(
        passenger_id=1,
        pickup_location={"latitude": 37.824454, "longitude": -122.231589},
        dropoff_location={"latitude": 37.821592, "longitude": -122.234797},
    )
    env.add_passenger(passenger)

    assert passenger in env.passengers
    env.render()


def test_observation_space():
    # Create a sample ride-sharing environment
    map_area = "Piedmont, California, USA"
    env = RideShareEnv(map_area)

    # Create some sample taxi agents
    car_info_1 = {
        "name": "Car 1",
        "vin": "ABC123",
        "description": "Sedan",
        "mileage_km": 10000,
        "tankCapacity": 50,
        "position": {"latitude": 37.824454, "longitude": -122.231589},
    }
    car_info_2 = {
        "name": "Car 2",
        "vin": "XYZ789",
        "description": "SUV",
        "mileage_km": 5000,
        "tankCapacity": 60,
        "position": {"latitude": 37.821592, "longitude": -122.234797},
    }
    agent_1 = TaxiAgent(env, car_info_1)
    agent_2 = TaxiAgent(env, car_info_2)
    env.add_agent(agent_1)
    env.add_agent(agent_2)

    # Create some sample passengers
    passenger_1 = Passenger(
        passenger_id=1,
        pickup_location={"latitude": 37.825000, "longitude": -122.232000},
        dropoff_location={"latitude": 37.820000, "longitude": -122.235000},
    )
    passenger_2 = Passenger(
        passenger_id=2,
        pickup_location={"latitude": 37.823000, "longitude": -122.233000},
        dropoff_location={"latitude": 37.819000, "longitude": -122.236000},
    )
    env.add_passenger(passenger_1)
    env.add_passenger(passenger_2)

    # Get the observation space
    observation_space = env._get_observation_space()

    # Check the correctness of the observation space
    assert isinstance(observation_space, spaces.Dict)
    assert "num_agents" in observation_space.spaces
    assert "num_passengers" in observation_space.spaces
    assert "agent_positions" in observation_space.spaces
    assert "passenger_positions" in observation_space.spaces
    assert "passenger_destinations" in observation_space.spaces

    # Check the correctness of the observation space dimensions
    assert observation_space["num_agents"].n == 3  # 2 agents + 1
    assert observation_space["num_passengers"].n == 3  # 2 passengers + 1
    assert observation_space["agent_positions"].shape == (2, 2)
    assert observation_space["passenger_positions"].shape == (2, 2)
    assert observation_space["passenger_destinations"].shape == (2, 2)

    # Reset the environment and check the initial observation
    observation = env.reset()
    assert observation["num_agents"] == 0
    assert observation["num_passengers"] == 0


def test_action_space():
    map_area = "Piedmont, California, USA"
    env = RideShareEnv(map_area)

    assert isinstance(env.action_space, spaces.Discrete)
