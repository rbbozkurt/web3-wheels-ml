# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 R. Berkay Bozkurt <resitberkaybozkurt@gmail.com>

from typing import Dict, List, Optional, Tuple

import numpy as np

entity_symbols = {
    "GasStation": "G",
    "Vehicle": "V",
    "PassengerPickUp": "P",
    "PassengerDropOff": "D",
    "Free": "O",
}


class Entity:
    def __init__(self, loc):
        self.loc = loc

    def reset(self):
        self.loc = np.array([0, 0])

    def grid_mark(self) -> str:
        return "O"

    def get_info(self) -> Dict:
        return {"location": self.loc}


class GasStation(Entity):
    counter = 0

    def __init__(self, loc):
        super().__init__(loc)
        self.id = GasStation.counter
        GasStation.counter += 1

    def reset(self):
        return super().reset()

    def grid_mark(self) -> str:
        return entity_symbols["GasStation"] + str(self.id)

    def get_info(self) -> Dict:
        return super().get_info()


class Passenger(Entity):
    counter = 0

    def __init__(self, pick_up_loc, drop_off_loc, ride_price):
        super().__init__(pick_up_loc)
        self.id = Passenger.counter
        Passenger.counter += 1

        self.pick_up_loc = self.loc.copy()
        self.drop_off_loc = drop_off_loc.copy()
        self.vehicle: Optional[Vehicle] = None
        self.ride_price = ride_price

    def reset(self):
        super.reset()
        self.pick_up_loc = np.array([0, 0])
        self.drop_off_loc = np.array([0, 0])
        self.vehicle = None

    def is_on_board(self) -> bool:
        return self.vehicle is not None

    def grid_mark(self) -> str:
        if self.is_on_board():
            return entity_symbols["PassengerDropOff"] + str(self.id)
        return entity_symbols["PassengerPickUp"] + str(self.id)

    def get_info(self) -> Dict:
        return {
            "location": self.loc,
            "pick_up_location": self.pick_up_loc,
            "drop_off_location": self.drop_off_loc,
            "ride_price": self.ride_price,
            "on_board": self.is_on_board(),
            "vehicle": self.vehicle.id if self.is_on_board() else None,
        }


class Vehicle(Entity):
    counter = 0

    # keep record of total distance
    def __init__(
        self, loc, tank_capacity, fuel_consumption, damage_level, damage_consumption
    ):
        super().__init__(loc)
        self.id = Vehicle.counter
        Vehicle.counter += 1
        self.tank_capacity = tank_capacity
        self.total_km = 0
        self.total_rides = 0
        self.total_earnings = 0
        self.fuel_level = tank_capacity
        self.fuel_consumption = fuel_consumption
        self.damage_level = damage_level
        self.damage_consumption = damage_consumption
        self.passenger: Optional[Passenger] = None

    def is_carrying_passenger(self) -> bool:
        return self.passenger is not None

    def has_gas(self) -> bool:
        return self.fuel_level > 0

    def reset(self):
        super().reset()
        self.fuel_level = self.tank_capacity
        self.damage_level = 0
        self.passenger = None
        self.total_km = 0
        self.total_rides = 0

    def grid_mark(self) -> str:
        if self.is_carrying_passenger():
            return "V" + str(self.id) + "-" + "R" + str(self.passenger.id)
        return "V" + str(self.id)

    def get_info(self) -> Dict:
        return {
            "position": self.loc,
            "fuel_level": self.fuel_level,
            "tank_capacity": self.tank_capacity,
            "damage_level": self.damage_level,
            "total_km": self.total_km,
            "total_rides": self.total_rides,
            "passenger": self.passenger.id if self.is_carrying_passenger() else None,
        }


# Define a mapping from characters to entity classes
Char_To_Entity = {
    "G": GasStation,
    "V": Vehicle,
    "P": Passenger,
}
