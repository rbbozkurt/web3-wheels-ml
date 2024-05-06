# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 R. Berkay Bozkurt <resitberkaybozkurt@gmail.com>

from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .utils import *

FPS = 4


class GridWorldEnv(gym.Env):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels",
        ],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode=None,
        map_string="",
        n_vehicle=1,
        n_passenger=1,
        n_gas_station=1,
        fuel_price=FUEL_PRICE.MEDIUM,
        fuel_consumption=FUEL_CONSUMPTION.MEDIUM,
        tank_capacity=TANK_CAPACITY.MEDIUM,
        damage_level=DAMAGE_LEVEL.MEDIUM,
        damage_consumption=DAMAGE_CONSUMPTION.MEDIUM,
    ):
        self.window_size = 512  # The size of the PyGame window
        self.map_string = map_string

        self.fuel_price = fuel_price
        self.fuel_consumption = fuel_consumption
        self.tank_capacity = tank_capacity
        self.damage_level = damage_level
        self.damage_consumption = damage_consumption

        self.np_random = np.random.default_rng()
        self.grid_map, self.grid_map_h, self.grid_map_w = self._init_map(
            self.map_string
        )
        self.n_gas_station = n_gas_station
        self.n_passenger = n_passenger
        self.n_vehicle = n_vehicle
        self.gas_stations = {}
        self.vehicles = {}
        self.passengers = {}
        self.observation_space = gym.spaces.Space(
            shape=self.grid_map.shape, dtype=self.grid_map.dtype
        )
        # We have 6 actions: up, down, left, right, pick up passenger, drop off passenger
        self.action_space = spaces.MultiDiscrete([len(ACTIONS)] * self.n_vehicle)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _init_map(self, chars_representation):
        """
        Converts a character representation of the world to a numpy array.

        Args:
        - chars_representation (str): String containing the character representation of the world.

        Returns:
        - chars_world (numpy.ndarray): Numpy array representing the world.
        - width (int): Width of the world.
        - height (int): Height of the world.
        """
        map = np.array(
            [list(line.strip()) for line in chars_representation.split("\n")],
            dtype=object,
        )
        height, width = map.shape
        return map, height, width

    def _get_obs(self):
        return self.grid_map

    def _get_info(self):
        return {
            "chars_world": self.grid_map,
            "vehicles": {
                vehicle.id: vehicle.get_info() for vehicle in self.vehicles.values()
            },
            "passengers": {
                passenger.id: passenger.get_info()
                for passenger in self.passengers.values()
            },
            "gas_stations": {
                gas_station.id: gas_station.get_info()
                for gas_station in self.gas_stations.values()
            },
        }

    def _get_not_occupied_positions(self):
        """
        Returns the positions in the grid that are not occupied.

        Returns:
            numpy.ndarray: An array of positions that are not occupied.
        """
        return np.argwhere(self.grid_map == entity_symbols["Free"])

    def _pick_random_not_occupied_position(self):
        """
        Picks a random position that is not occupied by any object in the grid.

        Returns:
        numpy.ndarray: An array representing the coordinates of the randomly picked position.
        """
        not_occupied_positions = self._get_not_occupied_positions()
        random_index = self.np_random.integers(not_occupied_positions.shape[0])
        return not_occupied_positions[random_index]

    def _place_passenges_randomly(self, n_passenger):
        """
        Places passengers randomly on the grid map.

        Args:
            n_passenger (int): The number of passengers to place.

        Returns:
            None
        """
        for _ in range(n_passenger):
            pick_up_loc = self._pick_random_not_occupied_position()
            drop_off_loc = self._pick_random_not_occupied_position()
            while np.array_equal(pick_up_loc, drop_off_loc):
                drop_off_loc = self._pick_random_not_occupied_position()
                pick_up_loc = self._pick_random_not_occupied_position()
            ride_price = np.linalg.norm(drop_off_loc - pick_up_loc)
            passenger = Passenger(pick_up_loc, drop_off_loc, ride_price)
            self.grid_map[passenger.loc[0], passenger.loc[1]] = passenger.grid_mark()
            self.passengers[passenger.id] = passenger

    def _place_vehicles_randomly(self, n_vehicles):
        """
        Places vehicles randomly on the grid map.

        Args:
            n_vehicles (int): The number of vehicles to be placed.

        Returns:
            None
        """
        for _ in range(n_vehicles):
            location = self._pick_random_not_occupied_position()
            vehicle = Vehicle(
                loc=location,
                tank_capacity=self.tank_capacity,
                fuel_consumption=self.fuel_consumption,
                damage_level=self.damage_level,
                damage_consumption=self.damage_consumption,
            )
            self.grid_map[vehicle.loc[0], vehicle.loc[1]] = vehicle.grid_mark()
            self.vehicles[vehicle.id] = vehicle

    def _place_gas_stations_randomly(self, n_gas_stations):
        """
        Randomly places gas stations on the grid map.

        Args:
            n_gas_stations (int): The number of gas stations to place.

        Returns:
            None
        """
        for _ in range(n_gas_stations):
            location = self._pick_random_not_occupied_position()
            gas_station = GasStation(loc=location)
            self.grid_map[
                gas_station.loc[0], gas_station.loc[1]
            ] = gas_station.grid_mark()
            self.gas_stations[gas_station.id] = gas_station

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.

        Parameters:
            seed (int): The random seed for generating the environment state. Default is None.
            options (dict): Additional options for resetting the environment. Default is None.

        Returns:
            observation (object): The initial observation of the environment.
            info (dict): Additional information about the environment state.
        """
        super().reset(seed=seed)
        Vehicle.counter = 0
        Passenger.counter = 0
        GasStation.counter = 0
        self.vehicles.clear()
        self.passengers.clear()
        self.gas_stations.clear()
        self.grid_map, self.grid_map_h, self.grid_map_w = self._init_map(
            self.map_string
        )

        self._place_vehicles_randomly(self.n_vehicle)
        self._place_passenges_randomly(self.n_passenger)
        self._place_gas_stations_randomly(self.n_gas_station)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _is_passenger_around_for_pickup(self, loc: np.ndarray):
        """
        Checks if there is a passenger around for pickup at the given location.

        Args:
            loc (np.ndarray): The location to check for passengers.

        Returns:
            Tuple[bool, int]: A tuple containing a boolean value indicating whether a passenger is around for pickup,
            and the ID of the passenger if one is found. If no passenger is found, the boolean value will be False and
            the ID will be -1.
        """
        for passenger in self.passengers.values():
            if np.linalg.norm(passenger.loc - loc) <= 1:
                return True, passenger.id
        return False, -1

    def _is_in_bounds(self, pos: np.ndarray):
        """
        Check if the given position is within the bounds of the grid.

        Parameters:
        - pos (np.ndarray): The position to check.

        Returns:
        - bool: True if the position is within the bounds, False otherwise.
        """
        return 0 <= pos[0] < self.grid_map_h and 0 <= pos[1] < self.grid_map_w

    def _is_location_occupied(self, pos: np.ndarray):
        """
        Check if the given position is occupied by an entity.

        Parameters:
        - pos (np.ndarray): The position to check.

        Returns:
        - bool: True if the location is occupied, False otherwise.
        """
        return not entity_symbols["Free"] in self.grid_map[pos[0], pos[1]]

    def _is_gas_station_around(self, loc: np.ndarray):
        """
        Checks if there is a gas station located within a distance of 1 unit from the given position.

        Parameters:
        - pos (np.ndarray): The position to check for gas stations.

        Returns:
        - bool: True if a gas station is found within the specified distance, False otherwise.
        """
        for gas_station in self.gas_stations.values():
            if np.linalg.norm(gas_station.loc - loc) <= 1:
                return True
        return False

    def _update_map_after_pickup(self, vehicle, passenger):
        """
        Updates the grid map after a passenger is picked up by a vehicle.

        Args:
            vehicle (Vehicle): The vehicle that picked up the passenger.
            passenger (Passenger): The passenger that was picked up.

        Returns:
            None
        """
        self.grid_map[
            passenger.pick_up_loc[0], passenger.pick_up_loc[1]
        ] = entity_symbols["Free"]
        self.grid_map[
            passenger.drop_off_loc[0], passenger.drop_off_loc[1]
        ] = passenger.grid_mark()
        self.grid_map[vehicle.loc[0], vehicle.loc[1]] = vehicle.grid_mark()
        print(
            "Map updated after pickup:"
            + "\n"
            + "Pick up location : {}".format(
                passenger.pick_up_loc[0], passenger.pick_up_loc[1]
            )
            + "\n"
            + "Drop off location : {}".format(
                passenger.drop_off_loc[0], passenger.drop_off_loc[1]
            )
            + "\n"
            + "Vehicle location : {}".format(vehicle.loc[0], vehicle.loc[1])
        )

    def _update_map_after_dropoff(self, vehicle, passenger):
        """
        Updates the grid map after a passenger is dropped off by a vehicle.

        Args:
            vehicle (Vehicle): The vehicle that dropped off the passenger.
            passenger (Passenger): The passenger that was dropped off.

        Returns:
            None
        """
        self.grid_map[
            passenger.drop_off_loc[0], passenger.drop_off_loc[1]
        ] = entity_symbols["Free"]
        self.grid_map[vehicle.loc[0], vehicle.loc[1]] = vehicle.grid_mark()
        print(
            "Map updated after dropoff:"
            + "\n"
            + "Drop off location : {}".format(
                passenger.drop_off_loc[0], passenger.drop_off_loc[1]
            )
            + "\n"
            + "Vehicle location : {}".format(vehicle.loc[0], vehicle.loc[1])
        )

    def _update_map_after_move(self, vehicle, old_loc):
        """
        Updates the grid map after a vehicle moves to a new location.

        Args:
            vehicle (Vehicle): The vehicle that moved.
            old_loc (tuple): The old location of the vehicle.

        Returns:
            None
        """
        self.grid_map[old_loc[0], old_loc[1]] = entity_symbols["Free"]
        self.grid_map[vehicle.loc[0], vehicle.loc[1]] = vehicle.grid_mark()
        print(
            "Map updated after move:"
            + "\n"
            + "Old loc : {}".format(self.grid_map[old_loc[0], old_loc[1]])
            + "\n"
            + "New location : {}".format(self.grid_map[vehicle.loc[0], vehicle.loc[1]])
        )

    def _step_action_pickup(self, vehicle):
        """
        Picks up a passenger if there is one around the vehicle's location.

        Args:
            vehicle (Vehicle): The vehicle attempting to pick up a passenger.

        Returns:
            bool: True if the passenger was successfully picked up, False otherwise.
        """
        print("Picking up: ")
        is_passenger_around, passenger_id = self._is_passenger_around_for_pickup(
            vehicle.loc
        )
        if not is_passenger_around:
            print("No passenger around: ")
            return False
        passenger = self.passengers[passenger_id]
        # Check if a passenger is already in a vehicle
        if vehicle.passenger is not None:
            print("Another assenger already in vehicle: ")
            return False

        passenger.vehicle = vehicle
        vehicle.passenger = passenger
        vehicle.passenger.loc = vehicle.loc
        print("Passenger {} picked up ".format(passenger.id))
        self._update_map_after_pickup(vehicle, passenger)
        return True

    def _is_at_dropoff_location(self, vehicle):
        """
        Checks if the vehicle is at the drop-off location of the passenger.

        Args:
            vehicle (Vehicle): The vehicle to check.

        Returns:
            bool: True if the vehicle is at the drop-off location, False otherwise.
        """

        return np.linalg.norm(vehicle.loc - vehicle.passenger.drop_off_loc) <= 1

    def _step_action_dropoff(self, vehicle):
        """
        Drops off a passenger at the dropoff location.

        Args:
            vehicle (Vehicle): The vehicle performing the dropoff.

        Returns:
            bool: True if the dropoff was successful, False otherwise.
        """
        print("Dropping off: ")
        # Check if the vehicle has a passenger
        if vehicle.passenger is None:
            print("No passenger in vehicle: ")
            return False
        if not self._is_at_dropoff_location(vehicle):
            print("Not at dropoff location: ")
            return False

        passenger = vehicle.passenger
        vehicle.passenger = None
        self.passengers.pop(passenger.id)
        vehicle.total_rides += 1
        vehicle.total_earnings += passenger.ride_price
        self._update_map_after_dropoff(vehicle, passenger)
        return True

    def _step_action_refuel(self, vehicle):
        """
        Refuels the vehicle if it is at a gas station.

        Args:
            vehicle (Vehicle): The vehicle to refuel.

        Returns:
            bool: True if the refuel was successful, False otherwise.
        """
        if vehicle.fuel_level == vehicle.tank_capacity:
            print("Already full: ")
            return False
        fuel_to_fill = vehicle.tank_capacity - vehicle.fuel_level
        vehicle.fuel_level += fuel_to_fill
        vehicle.total_earnings -= fuel_to_fill
        return True

    def _step_action_move(self, vehicle, new_loc):
        """
        Moves the vehicle to a new location and updates relevant attributes.

        Args:
            vehicle (Vehicle): The vehicle to move.
            new_loc (tuple): The new location to move the vehicle to.

        Returns:
            bool: True if the vehicle was successfully moved, False otherwise.
        """
        if not self._is_in_bounds(new_loc):
            print("Out of bounds: ")
            return False
        old_loc = vehicle.loc.copy()
        vehicle.loc = new_loc
        print("New location: ", vehicle.loc)
        print("Old location: ", old_loc)
        # Update passenger location if there is one
        if vehicle.passenger is not None:
            vehicle.passenger.loc = new_loc
        vehicle.total_km += 1
        vehicle.fuel_level -= vehicle.fuel_consumption
        self._update_map_after_move(vehicle, old_loc)
        return True

    def step(self, actions: List):
        reward = 0
        terminated = False
        truncated = False

        # iterate over all actions for each vehicle
        for index, action in enumerate(actions):
            vehicle = self.vehicles[index]
            # print what action the vehicle is taking
            print(
                "Vehicle {} taking the action: {}".format(
                    index, Action_To_String[action]
                )
            )

            if action == ACTIONS.PICKUP:
                # check if there is a passenger around for pickup
                if self._step_action_pickup(vehicle):
                    reward += REWARDS.SUCCESS_PICKUP
                    print("Picking up was succesful: ")
                else:
                    reward += REWARDS.FAIL_PICKUP

            elif action == ACTIONS.DROPOFF:
                if self._step_action_dropoff(vehicle):
                    print("Dropping off was succesful: ")
                    reward += REWARDS.SUCCESS_DROPOFF
                    print("All passengers are dropped off")
                    terminated = len(self.passengers) == 0
                else:
                    print("Dropping off was not succesful: ")
                    reward += REWARDS.FAIL_DROPOFF

            elif action == ACTIONS.REFUEL:
                if not self._is_gas_station_around(vehicle.loc):
                    print("No gas station around: ")
                    reward += REWARDS.FAIL_REFUEL
                elif self._step_action_refuel(vehicle):
                    print("Refueled ")
                    reward += REWARDS.SUCCESS_REFUEL
                else:
                    print("Refueling failed: ")
                    reward += REWARDS.FAIL_REFUEL

            else:
                initial_vehicle_loc = vehicle.loc.copy()
                direction = Action_To_Direction[action]
                new_vehicle_loc = initial_vehicle_loc + direction
                if not self._is_in_bounds(new_vehicle_loc):
                    print("New location out of bounds: ", new_vehicle_loc)
                    reward += REWARDS.OUT_OF_BOUNDS
                elif self._is_location_occupied(new_vehicle_loc):
                    print("New location occupied: ", new_vehicle_loc)
                    reward += REWARDS.COLLISION
                # Check if the vehicle has enough fuel to move
                elif not (vehicle.fuel_level - vehicle.fuel_consumption >= 0):
                    # Check if the vehicle has a passenger
                    print("No fuel: ")
                    if vehicle.passenger is not None:
                        print("No fuel with passenger: ")
                        reward += REWARDS.INSUFFICIENT_FUEL_W_PASSENGER
                    else:
                        reward += REWARDS.INSUFFICIENT_FUEL
                        # if there is no fuel, check if there is a gas station around
                        if not self._is_gas_station_around(initial_vehicle_loc):
                            truncated = True
                            print("Truncated: ")
                else:
                    print("Moving: ")
                    self._step_action_move(vehicle, new_vehicle_loc)
                    reward += REWARDS.SUCCESS_MOVE

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        print(self.grid_map)
        return self.grid_map

    def close(self):
        return
