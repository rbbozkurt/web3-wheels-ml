# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2023 R. Berkay Bozkurt <resitberkaybozkurt@gmail.com>
from pyvirtualdisplay import Display

from demos import Action_To_String, GridWorldEnv, grid_world_5_x_5

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

import os
import random

import gymnasium as gym
import imageio
import numpy as np
import pickle5 as pickle


def print_actions(actions: list):
    """
    This function prints the actions in a human-readable format.
    """
    for index, action in enumerate(actions):
        print(
            "Vehicle {} taking the action: {}".format(index, Action_To_String[action])
        )


def parse_print_info(info: dict):
    """
    This function parses and prints the information dictionary.
    """
    print(" ------ Info ------ ")
    print("Vehicles:")
    for id, vehicle in info["vehicles"].items():
        print(f"ID: {id}")
        for key, value in vehicle.items():
            print(f"  {key}: {value}")
        print()

    print("Passengers:")
    for id, passenger in info["passengers"].items():
        print(f"ID: {id}")
        for key, value in passenger.items():
            print(f"  {key}: {value}")
        print()

    print("Gas Stations:")
    for id, gas_station in info["gas_stations"].items():
        print(f"ID: {id}")
        for key, value in gas_station.items():
            print(f"  {key}: {value}")
        print()


def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable


def greedy_policy(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    action = np.argmax(Qtable[state][:])

    return action


def epsilon_greedy_policy(Qtable, state, epsilon, env):
    # Randomly generate a number between 0 and 1
    random_num = random.uniform(0, 1)
    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
        # Take the action with the highest value given a state
        # np.argmax can be useful here
        action = greedy_policy(Qtable, state)
    # else --> exploration
    else:
        action = env.action_space.sample()

    return action


def train(
    n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable
):
    for episode in range(n_training_episodes):
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -decay_rate * episode
        )
        # Reset the environment
        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False
        learning_rate = 0.7  # Learning rate
        gamma = 0.95  # Discounting rate
        # repeat
        for step in range(max_steps):
            # Choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(Qtable, state, epsilon, env)

            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, terminated, truncated, info = env.step(action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            Qtable[state][action] = Qtable[state][action] + learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
            )

            # If terminated or truncated finish the episode
            if terminated or truncated:
                break

            # Our next state is the new state
            state = new_state
    return Qtable


def main():
    """
    This is the main function of the program.
    It serves as the entry point for the application.
    """
    grid_world = GridWorldEnv(
        map_string=grid_world_5_x_5, n_vehicle=1, n_passenger=1, n_gas_station=1
    )
    observation, info = grid_world.reset(seed=42)
    print(" ------ Initial Info ------ ")
    print()
    print(grid_world.render())
    parse_print_info(info)
    for _ in range(1000):
        actions = grid_world.action_space.sample()
        print_actions(actions)
        observation, reward, terminated, truncated, info = grid_world.step(actions)
        grid_world.render()
        parse_print_info(info)
        if terminated or truncated:
            observation, info = grid_world.reset()
    grid_world.close()


def main2():
    env = GridWorldEnv(
        map_string=grid_world_5_x_5, n_vehicle=1, n_passenger=1, n_gas_station=1
    )

    state_space = 5 * 5 * 4 * 25
    action_space = 6

    Qtable_taxi = initialize_q_table(state_space, action_space)

    # Training parameters
    n_training_episodes = 100000  # Total training episodes

    # Evaluation parameters
    n_eval_episodes = 99  # Total number of test episodes

    # DO NOT MODIFY EVAL_SEED
    eval_seed = [
        16,
        54,
        165,
        177,
        191,
        191,
        120,
        80,
        149,
        178,
        48,
        38,
        6,
        125,
        174,
        73,
        50,
        172,
        100,
        148,
        146,
        6,
        25,
        40,
        68,
        148,
        49,
        167,
        9,
        97,
        164,
        176,
        61,
        7,
        54,
        55,
        161,
        131,
        184,
        51,
        170,
        12,
        120,
        113,
        95,
        126,
        51,
        98,
        36,
        135,
        54,
        82,
        45,
        95,
        89,
        59,
        95,
        124,
        9,
        113,
        58,
        85,
        51,
        134,
        121,
        169,
        105,
        21,
        30,
        11,
        50,
        65,
        12,
        43,
        82,
        145,
        152,
        97,
        106,
        55,
        31,
        85,
        38,
        112,
        102,
        168,
        123,
        97,
        21,
        83,
        158,
        26,
        80,
        63,
        5,
        81,
        32,
        11,
        28,
        148,
    ]  # Evaluation seed, this ensures that all classmates agents are trained on the same taxi starting position
    # Each seed has a specific starting state

    # Environment parameters
    env_id = "Taxi-v3"  # Name of the environment
    max_steps = 200  # Max steps per episode

    # Exploration parameters
    max_epsilon = 1.0  # Exploration probability at start
    min_epsilon = 0.05  # Minimum exploration probability
    decay_rate = 0.005  # Exponential decay rate for exploration prob

    Qtable_taxi = train(
        n_training_episodes,
        min_epsilon,
        max_epsilon,
        decay_rate,
        env,
        max_steps,
        Qtable_taxi,
    )

    # Define the filename to save the Q-table
    q_table_file = "src/demos/qtable_taxiv3/q_table_taxi.pkl"

    # Save the Q-table to disk
    with open(q_table_file, "wb") as f:
        pickle.dump(Qtable_taxi, f)

    print("Q-table saved to:", q_table_file)


if __name__ == "__main__":
    main2()
