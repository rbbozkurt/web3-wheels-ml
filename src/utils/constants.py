# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 R. Berkay Bozkurt <resitberkaybozkurt@gmail.com>

from enum import IntEnum

import numpy as np


class ACTIONS(IntEnum):
    """
    Enum class representing the possible actions in the environment.

    Attributes:
        NORTH (int): Move north action.
        SOUTH (int): Move south action.
        EAST (int): Move east action.
        WEST (int): Move west action.
        PICKUP (int): Pickup passenger action.
        DROPOFF (int): Drop off passenger action.
    """

    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    PICKUP = 4
    DROPOFF = 5
    REFUEL = 6


class DAMAGE_LEVEL(IntEnum):
    """
    Enum class representing the damage levels.

    Attributes:
        LOW (int): Low damage level.
        MEDIUM (int): Medium damage level.
        HIGH (int): High damage level.
    """

    LOW = 50
    MEDIUM = 75
    HIGH = 100


class TANK_CAPACITY(IntEnum):
    """
    Enum class representing the fuel levels.

    Attributes:
        LOW (int): Low fuel level.
        MEDIUM (int): Medium fuel level.
        HIGH (int): High fuel level.
    """

    LOW = 50
    MEDIUM = 75
    HIGH = 100


class FUEL_CONSUMPTION(IntEnum):
    """
    Enum class representing the fuel consumption levels.

    Attributes:
        LOW (float): Low fuel consumption level.
        MEDIUM (float): Medium fuel consumption level.
        HIGH (float): High fuel consumption level.
    """

    LOW = 0.5
    MEDIUM = 1
    HIGH = 2


class DAMAGE_CONSUMPTION(IntEnum):
    """
    Enum class representing the damage consumption levels.

    Attributes:
        LOW (float): Low damage consumption level.
        MEDIUM (float): Medium damage consumption level.
        HIGH (float): High damage consumption level.
    """

    LOW = 0.2
    MEDIUM = 0.5
    HIGH = 1


class FUEL_PRICE(IntEnum):
    """
    Enum class representing the fuel prices.

    Attributes:
        LOW (int): Low fuel price.
        MEDIUM (int): Medium fuel price.
        HIGH (int): High fuel price.
    """

    LOW = 1
    MEDIUM = 2
    HIGH = 3


class REWARDS(IntEnum):
    """
    Enum class representing the rewards for different actions in the environment.

    Attributes:
        SUCCESS_PICKUP (int): Reward for successfully picking up a passenger.
        SUCCESS_DROPOFF (int): Reward for successfully dropping off a passenger.
        FAIL_PICKUP (int): Reward for unsuccessful pıck up action.
        FAIL_DROPOFF (int): Reward for unsuccessful drop off action.
        INSUFFICIENT_FUEL (int): Reward for running out of fuel.
        INSUFFICIENT_FUEL_W_PASSENGER (int): Reward for running out of fuel with a passenger on board.
        UNNEEDED_REFUEL (int): Reward for refueling when not needed.
    """

    SUCCESS_PICKUP = 100
    SUCCESS_DROPOFF = 100
    SUCCESS_MOVE = 10
    SUCCESS_REFUEL = 100

    FAIL_PICKUP = -100
    FAIL_DROPOFF = -100
    FAIL_REFUEL = -100
    INSUFFICIENT_FUEL = -100
    INSUFFICIENT_FUEL_W_PASSENGER = -200
    UNNEEDED_REFUEL = -50
    OUT_OF_BOUNDS = -100
    COLLISION = -100


Action_To_Direction = {
    0: np.array([-1, 0]),  # Go up
    1: np.array([1, 0]),  # Go down
    2: np.array([0, 1]),  # Go right
    3: np.array([0, -1]),  # Go left
    4: None,  # Pick up passenger
    5: None,  # Drop off passenger
    6: None,  # Refuel
}

Action_To_String = {
    0: "NORTH",
    1: "SOUTH",
    2: "EAST",
    3: "WEST",
    4: "PICKUP",
    5: "DROPOFF",
    6: "REFUEL",
}
