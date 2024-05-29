# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>

import os
import random
import sys
from datetime import datetime

import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from src.agents.coordinator_agent import AICoordinator
from src.agents.passenger import Passenger
from src.agents.taxi_agent import TaxiAgent
from src.envs.osm_env import RideShareEnv

# Load evaluation parameters from ppo_config.yml file
with open("src/training/ppo_config.yml", "r") as f:
    config = yaml.safe_load(f)


def evaluate(coordinator, num_episodes):
    total_reward = 0

    for episode in range(num_episodes):
        # Pick a city randomly from the list of cities for evaluation
        city = random.choice(config["cities"])
        env = RideShareEnv(config, city)
        env.coordinator = coordinator
        done = False
        episode_reward = 0

        # Create random taxis and place them in random locations in the city
        num_taxis = random.randint(1, config["max_taxis"])
        for _ in range(num_taxis):
            taxi_info = {
                "name": f"Taxi_{_}",
                "vin": f"VIN_{_}",
                "description": "Random Taxi",
                "mileage_km": random.randint(1000, 10000),
                "tankCapacity": random.randint(40, 60),
                "position": {
                    "latitude": random.uniform(env.map_bounds[1], env.map_bounds[3]),
                    "longitude": random.uniform(env.map_bounds[0], env.map_bounds[2]),
                },
            }
            taxi_agent = TaxiAgent(env, taxi_info)
            env.add_agent(taxi_agent)

        # Create a figure and axis for the animation
        fig, ax = plt.subplots()

        # Define the update function for the animation
        def update(frame):
            nonlocal done, episode_reward

            # Clear the previous plot
            ax.clear()

            # Probabilistically add passengers to random locations on the map
            if random.random() < config["passenger_spawn_probability"]:
                if len(env.passengers) < config["max_passengers"]:
                    passenger_info = {
                        "passenger_id": len(env.passengers) + 1,
                        "pickup_location": {
                            "latitude": random.uniform(
                                env.map_bounds[1], env.map_bounds[3]
                            ),
                            "longitude": random.uniform(
                                env.map_bounds[0], env.map_bounds[2]
                            ),
                        },
                        "destination": {
                            "latitude": random.uniform(
                                env.map_bounds[1], env.map_bounds[3]
                            ),
                            "longitude": random.uniform(
                                env.map_bounds[0], env.map_bounds[2]
                            ),
                        },
                    }
                    passenger = Passenger(**passenger_info)
                    env.add_passenger(passenger)

            next_observation, actions, rewards, done, _ = env.step(
                time_interval=config["time_interval"]
            )
            episode_reward += sum(rewards)

            # Render the environment for the current frame
            env.render(ax=ax, output_file=None)

            # Set the title for the current frame
            ax.set_title(f"Episode {episode+1} - Step: {frame}")

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=100, interval=500)
        now = datetime.now()
        folder_name = now.strftime("%Y-%m-%d")
        filename = now.strftime("%H-%M-%S")
        if not os.path.exists(f"results/{folder_name}"):
            os.makedirs(f"results/{folder_name}")
        # Save the file to the new folder
        ani.save(
            f"results/{folder_name}/{filename}_episode_{episode+1}.gif", writer="pillow"
        )

        # Close the figure
        plt.close(fig)

        total_reward += episode_reward
        print(f"Episode {episode+1}: Reward = {episode_reward}")

    average_reward = total_reward / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {average_reward}")


if __name__ == "__main__":
    # Load the trained coordinator
    trained_coordinator = AICoordinator(RideShareEnv(), config)
    trained_coordinator.model = trained_coordinator.model.load(
        "src/training/saved_models/trained_coordinator"
    )

    num_episodes = config["eval_num_episodes"]
    evaluate(trained_coordinator, num_episodes)
