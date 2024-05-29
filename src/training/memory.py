# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        # Check if the keys in state and next_state dictionaries are consistent
        assert set(state.keys()) == set(
            next_state.keys()
        ), "State and next state dictionaries should have the same keys"

        # Check if the shapes of the values in state and next_state dictionaries are consistent
        for key in state.keys():
            assert isinstance(
                state[key], np.ndarray
            ), f"Value for key '{key}' in state should be a numpy array"
            assert isinstance(
                next_state[key], np.ndarray
            ), f"Value for key '{key}' in next_state should be a numpy array"
            assert (
                state[key].shape == next_state[key].shape
            ), f"Shape mismatch for key '{key}' in state and next_state"

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert dictionaries to numpy arrays
        states = {
            key: np.array([state[key] for state in states]) for key in states[0].keys()
        }
        next_states = {
            key: np.array([next_state[key] for next_state in next_states])
            for key in next_states[0].keys()
        }

        return (
            states,
            np.array(actions),
            np.array(rewards),
            next_states,
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)
