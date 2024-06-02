# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>

import os
import random
import sys

import yaml
from memory import ReplayBuffer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

import numpy as np
import tensorflow as tf
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.agents.coordinator_agent import AICoordinator
from src.agents.passenger import Passenger
from src.agents.taxi_agent import TaxiAgent
from src.envs.osm_env import RideShareEnv

# Load training parameters from ppo_config.yml file
with open("src/training/ppo_config.yml", "r") as f:
    config = yaml.safe_load(f)


def make_env():
    def _init():
        env = RideShareEnv(config)
        return env

    return _init


def train(num_episodes, batch_size, replay_buffer_capacity, num_training_steps):
    num_envs = 1  # Number of parallel environments
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    coordinator = AICoordinator(env, config)
    # Check if a saved model exists
    saved_model_path = "src/training/saved_models/trained_coordinator"
    if os.path.exists(saved_model_path + ".zip"):
        # Load the saved model
        coordinator.model.load(saved_model_path, env=env)
        print("Loaded previously trained model.")

    env.coordinator = coordinator
    replay_buffer = ReplayBuffer(replay_buffer_capacity)
    summary_writer = tf.summary.create_file_writer("src/training/logger")

    for episode in range(num_episodes):
        # Pick a city randomly from the list of cities for training
        env.reset()
        done = False
        episode_reward = 0
        passenger_id = 1
        # Create random taxis and place them in random locations in the city
        num_taxis = random.randint(5, config["max_taxis"])
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

        while not done:
            # Probabilistically add passengers to random locations on the map

            if random.random() < config["passenger_spawn_probability"]:
                if len(env.passengers) < config["max_passengers"]:
                    passenger_info = {
                        "passenger_id": passenger_id,
                        "request_time": env.current_time_step,
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
                    passenger_id += 1
            observation = env._get_observation()
            next_observation, actions, rewards, done, info = coordinator.step(
                config["time_interval"]
            )

            if actions is not None:
                replay_buffer.push(
                    observation, actions, np.sum(rewards), next_observation, done
                )
            episode_reward += rewards

        print(f"Episode {episode+1}: Reward = {episode_reward}")
        # Calculate percentage of passengers picked up and delivered
        num_passengers = passenger_id
        passengers_delivered = info["passengers_delivered"]
        ratio = passengers_delivered / num_passengers
        # Log relevant data
        with summary_writer.as_default():
            tf.summary.scalar("train/reward", episode_reward, step=episode)
            tf.summary.scalar("train/Percent_trips_completed", ratio, step=episode)
        coordinator.logger.dump(step=episode)
        # Train the coordinator after each episode
        if len(replay_buffer) >= batch_size:
            for _ in range(num_training_steps):
                experiences = replay_buffer.sample(batch_size)
                coordinator.train(experiences)

    return coordinator


def train2():
    num_envs = 12  # Number of parallel environments
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    coordinator = AICoordinator(env, config)

    # Check if a saved model exists
    saved_model_path = "src/training/saved_models/trained_coordinator"
    if os.path.exists(saved_model_path + ".zip"):
        # Load the saved model
        coordinator.model.load(saved_model_path, env=env)
        print("Loaded previously trained model.")

    env.coordinator = coordinator

    coordinator.model.learn(
        total_timesteps=config["total_time_steps"],
        log_interval=100,
        tb_log_name="parallel_training",
        callback=coordinator._on_step,
        progress_bar=True,
    )
    return coordinator


if __name__ == "__main__":
    trained_coordinator = train2()

    # Save the trained coordinator if needed
    print("saving trained coordinator")
    trained_coordinator.model.save("src/training/saved_models/trained_coordinator")
