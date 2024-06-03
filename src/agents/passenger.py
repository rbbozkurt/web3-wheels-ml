# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>
import time


class Passenger:
    def __init__(
        self,
        passenger_id,
        pickup_location,
        destination,
        request_time=time.time(),
    ):
        self.passenger_id = passenger_id
        self.pickup_location = {
            "node": pickup_location["node"],
            "longitude": pickup_location["longitude"],
            "latitude": pickup_location["latitude"],
        }
        self.destination = {
            "node": destination["node"],
            "longitude": destination["longitude"],
            "latitude": destination["latitude"],
        }

        self.position = pickup_location
        self.ride_request_time = request_time
        self.waiting_time = 0
        self.picked_up = False
        self.which_taxi = []
        self.completed = False

    def update_waiting_time(self):
        if not self.picked_up and not self.completed:
            current_time = time.time()
            self.waiting_time += current_time - self.ride_request_time

    def is_picked_up(self):
        return self.picked_up

    def set_picked_up(self, picked_up):
        self.picked_up = picked_up

    def is_completed(self):
        return self.completed

    def set_completed(self, completed):
        self.completed = completed

    def get_wait_time(self):
        return self.waiting_time

    def __str__(self):
        return f"Passenger {self.passenger_id}: Pickup: {self.pickup_location}, Dropoff: {self.destination}"
