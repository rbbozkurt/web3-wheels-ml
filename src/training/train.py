# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>

import os
import random
import sys

import yaml
from memory import ReplayBuffer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from agents.coordinator_agent import AICoordinator
from agents.passenger import Passenger
from agents.taxi_agent import TaxiAgent
from envs.osm_env import RideShareEnv

# Load training parameters from ppo_config.yml file
with open("ppo_config.yml", "r") as f:
    config = yaml.safe_load(f)


def train(num_episodes, batch_size, replay_buffer_capacity):
    env = RideShareEnv()
    coordinator = AICoordinator(
        env, max_agents=config["max_taxis"], max_passengers=config["max_passengers"]
    )
    replay_buffer = ReplayBuffer(replay_buffer_capacity)

    for episode in range(num_episodes):
        # Pick a city randomly from the list of cities for training
        city = random.choice(config["cities"])
        state = env.reset(city)
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
                        "dropoff_location": {
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

            action = coordinator.get_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(replay_buffer) >= batch_size:
                experiences = replay_buffer.sample(batch_size)
                coordinator.train(experiences)

        print(f"Episode {episode+1}: Reward = {episode_reward}")

    return coordinator


if __name__ == "__main__":
    num_episodes = config["num_episodes"]
    batch_size = config["batch_size"]
    replay_buffer_capacity = config["replay_buffer_capacity"]

    trained_coordinator = train(num_episodes, batch_size, replay_buffer_capacity)
    # Save the trained coordinator if needed
