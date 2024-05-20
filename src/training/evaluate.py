# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>

import random

import yaml

from agents import AICoordinator, Passenger, TaxiAgent
from envs import RideShareEnv

# Load evaluation parameters from ppo_config.yml file
with open("ppo_config.yml", "r") as f:
    config = yaml.safe_load(f)


def evaluate(coordinator, num_episodes):
    total_reward = 0

    for episode in range(num_episodes):
        # Pick a city randomly from the list of cities for evaluation
        city = random.choice(config["cities"])
        env = RideShareEnv(city)
        state = env.reset()
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
                    "latitude": random.uniform(env.map_bounds[0], env.map_bounds[2]),
                    "longitude": random.uniform(env.map_bounds[1], env.map_bounds[3]),
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
                                env.map_bounds[0], env.map_bounds[2]
                            ),
                            "longitude": random.uniform(
                                env.map_bounds[1], env.map_bounds[3]
                            ),
                        },
                        "dropoff_location": {
                            "latitude": random.uniform(
                                env.map_bounds[0], env.map_bounds[2]
                            ),
                            "longitude": random.uniform(
                                env.map_bounds[1], env.map_bounds[3]
                            ),
                        },
                    }
                    passenger = Passenger(**passenger_info)
                    env.add_passenger(passenger)

            action = coordinator.get_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward

        total_reward += episode_reward
        print(f"Episode {episode+1}: Reward = {episode_reward}")

    average_reward = total_reward / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {average_reward}")


if __name__ == "__main__":
    # Load the trained coordinator
    trained_coordinator = AICoordinator(RideShareEnv())
    # Load the trained weights into the coordinator

    num_episodes = config["eval_num_episodes"]
    evaluate(trained_coordinator, num_episodes)
