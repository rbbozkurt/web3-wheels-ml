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


def make_movies(num_episodes, num_steps):
    env = RideShareEnv(config)
    coordinator = AICoordinator(env, config)

    # Load the trained model
    saved_model_path = "src/training/saved_models/trained_coordinator"
    if os.path.exists(saved_model_path + ".zip"):
        coordinator.model.load(saved_model_path, env=env)
        print("Loaded trained model.")
    else:
        print("No trained model found. Please train the model first.")
        return
    env.coordinator = coordinator

    episode_rewards = []
    episode_passengers_delivered = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        episode_reward = 0
        passengers_delivered = 0

        # Create a figure and axis for the animation
        fig, ax = plt.subplots()

        # Define the update function for the animation
        def update(frame):
            nonlocal obs, episode_reward, passengers_delivered

            # Clear the previous plot
            ax.clear()

            # Probabilistically add passengers to random locations on the map

            action, _ = coordinator.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            passengers_delivered = info["passengers_delivered"]
            # Render the environment for the current frame
            env.render(ax=ax, output_file=None)
            ax.set_title(f"Episode {episode+1} - Step: {frame}")

        episode_rewards.append(episode_reward)
        episode_passengers_delivered.append(passengers_delivered)

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=500)
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

        print(
            f"Episode {episode + 1}: Reward = {episode_reward}, Passengers Delivered = {passengers_delivered}"
        )

    average_reward = sum(episode_rewards) / num_episodes
    average_passengers_delivered = sum(episode_passengers_delivered) / num_episodes

    print(f"\nAverage Reward over {num_episodes} episodes: {average_reward}")
    print(
        f"Average Passengers Delivered over {num_episodes} episodes: {average_passengers_delivered}"
    )


if __name__ == "__main__":
    num_episodes = 1  # Specify the number of episodes to evaluate
    num_steps = 300  # Specify the number of steps in each episode
    make_movies(num_episodes, num_steps)
