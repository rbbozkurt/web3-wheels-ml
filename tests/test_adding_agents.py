# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>

import os
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import osmnx as ox

# Add the parent directory of 'tests' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.passenger import Passenger
from src.agents.taxi_agent import TaxiAgent
from src.envs.osm_env import RideShareEnv


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
    map_area = "Manhattan, New York City, New York, USA"
    env = RideShareEnv(map_area)

    # Create some sample car information
    car_info_1 = {
        "name": "Car 1",
        "vin": "ABC123",
        "description": "Sedan",
        "mileage_km": 10000,
        "tankCapacity": 50,
        "position": {"latitude": 40.712776, "longitude": -74.005974},
    }

    # Create agent instances using the car information
    agent_1 = TaxiAgent(env, car_info_1)

    # Add the taxi_agents to the environment
    env.add_agent(agent_1)

    # Set a destination for the agent
    destination = ox.distance.nearest_nodes(env.map_network, 40.718407, -74.007068)
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
        if agent_1.position["node"] == destination:
            state["reached_destination"] = True

        else:
            # Clear the previous plot
            ax.clear()
            # Perform the agent's action for the current frame
            agent_1.action_move()

            # Render the environment for the current frame
            env.render(ax=ax, output_file=None)

            # Set the title for the current frame
            ax.set_title(f"Step: {frame}")

    # Get the shortest path from the agent's current position to the destination
    # path = env.get_route(agent_1.position["node"], destination)

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
        destination={"latitude": 37.821592, "longitude": -122.234797},
    )
    env.add_passenger(passenger)

    assert passenger in env.passengers
    env.render()
