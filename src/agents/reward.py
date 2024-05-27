# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>


# Different reward functions for training


def reward_function_basic(env):
    """
    Calculate the reward for the current state of the environment and agents.
    - Consider factors like waiting time, fuel efficiency, ride completion, etc.
    - Adds reward if passanger is picked up and dropped off at the correct location.
    - Penalizes if passanger is not picked up promptly
    """

    rewards = []
    for passenger in env.passengers:
        if not passenger.is_picked_up():
            # Negative reward for each time step passenger is waiting
            reward = -1

            if passenger.is_picked_up():
                # Positive reward for passenger pickup
                reward += 10
            rewards.append(reward)
        elif passenger.is_completed():
            # Positive reward for successful ride completion
            reward = 50
            rewards.append(reward)
    return rewards
