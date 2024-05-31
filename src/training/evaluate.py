# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>

import os
import random
import sys

import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.agents.coordinator_agent import AICoordinator
from src.agents.passenger import Passenger
from src.agents.taxi_agent import TaxiAgent
from src.envs.osm_env import RideShareEnv

# Load evaluation parameters from ppo_config.yml file
with open("src/training/ppo_config.yml", "r") as f:
    config = yaml.safe_load(f)


def evaluate(num_episodes, num_steps):
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

        for step in range(num_steps):
            action, _ = coordinator.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            passengers_delivered = info["passengers_delivered"]

        episode_rewards.append(episode_reward)
        episode_passengers_delivered.append(passengers_delivered)

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
    num_episodes = 10  # Specify the number of episodes to evaluate
    num_steps = 100  # Specify the number of steps in each episode
    evaluate(num_episodes, num_steps)
