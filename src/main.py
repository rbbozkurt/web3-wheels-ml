# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2023 R. Berkay Bozkurt <resitberkaybozkurt@gmail.com>
from demos import Action_To_String, GridWorldEnv, grid_world_5_x_5


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
