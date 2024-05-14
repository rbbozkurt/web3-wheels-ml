# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2023 R. Berkay Bozkurt <resitberkaybozkurt@gmail.com>
import random

import numpy as np
from pyvirtualdisplay import Display

from demos import Action_To_String, GridWorldEnv, grid_world_5_x_5

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()


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


if __name__ == "__main__":
    main()
