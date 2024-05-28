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

from src.agents.coordinator_agent import AICoordinator
from src.agents.passenger import Passenger
from src.agents.taxi_agent import TaxiAgent
from src.envs.osm_env import RideShareEnv

# Load training parameters from ppo_config.yml file
with open("src/training/ppo_config.yml", "r") as f:
    config = yaml.safe_load(f)


def train(num_episodes, batch_size, replay_buffer_capacity, num_training_steps):
    city = random.choice(config["cities"])
    max_time_steps = config["max_time_steps"]
    env = RideShareEnv(city, max_time_steps)
    coordinator = AICoordinator(
        env,
        config,
    )
    env.coordinator = coordinator
    replay_buffer = ReplayBuffer(replay_buffer_capacity)

    for episode in range(num_episodes):
        # Pick a city randomly from the list of cities for training
        city = random.choice(config["cities"])
        env.reset(city)
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

        while not done:
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
            observation = env._get_observation()
            next_observation, actions, rewards, done, _ = env.step(time_interval=3)

            if actions is not None:
                replay_buffer.push(
                    observation, actions, np.sum(rewards), next_observation, done
                )
            episode_reward += np.sum(rewards)

        print(f"Episode {episode+1}: Reward = {episode_reward}")

        # Train the coordinator after each episode
        if len(replay_buffer) >= batch_size:
            for _ in range(num_training_steps):
                experiences = replay_buffer.sample(batch_size)
                coordinator.train(experiences)

    return coordinator


if __name__ == "__main__":
    num_episodes = config["num_episodes"]
    batch_size = config["batch_size"]
    replay_buffer_capacity = config["replay_buffer_capacity"]

    trained_coordinator = train(
        num_episodes, batch_size, replay_buffer_capacity, num_training_steps=3
    )
    # Save the trained coordinator if needed
    trained_coordinator.model.save("src/training/saved_models/trained_coordinator")
