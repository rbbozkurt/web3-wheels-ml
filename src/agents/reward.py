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


def reward_function_2(env, current_time_step):
    """
    Calculate the reward for the current state of the environment and agents.
    - Consider factors like waiting time, fuel efficiency, ride completion, etc.
    - Adds reward if passenger is picked up and dropped off at the correct location.
    - Penalizes if passenger is not picked up promptly
    - Rewards if the assigned taxi destination matches a passenger pickup position
    """

    total_reward = 0

    for taxi in env.taxi_agents:
        reward = 0

        # Check if the assigned destination matches a passenger pickup position
        for passenger in env.passengers:
            if (
                not passenger.is_picked_up()
                and taxi.destination == passenger.position["node"]
            ):
                reward += 2  # Reward for assigning the correct destination

        # Check if the taxi has picked up a passenger
        if taxi.passengers:
            for passenger in taxi.passengers:
                if passenger.is_picked_up() and not passenger.is_completed():
                    reward += 0.1  # Reward for passenger pickup
                elif passenger.is_completed():
                    reward += 0.5  # Reward for successful ride completion

        # Penalize if the taxi has no assigned destination and there are passengers waiting
        if not taxi.destination and any(not p.is_picked_up() for p in env.passengers):
            reward -= 0.1  # Penalty for not having an assigned destination when passengers are waiting

        total_reward += reward

    # Additional rewards/penalties based on passenger waiting time
    for passenger in env.passengers:
        if not passenger.is_picked_up():
            # Negative reward for each time step passenger is waiting
            # passenger.update_waiting_time()
            waiting_time_minutes = current_time_step - passenger.ride_request_time
            total_reward -= (
                0.00001 * waiting_time_minutes
            )  # Penalty scaled down based on waiting time in minutes

    return total_reward
