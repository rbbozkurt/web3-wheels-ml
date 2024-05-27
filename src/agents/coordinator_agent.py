# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn


class AICoordinator(gym.Env):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.max_agents = config["max_taxis"]
        self.max_passengers = config["max_passengers"]

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        # Configure the logger
        self.logger = configure(
            folder="src/training/logger",
            format_strings=["stdout", "log", "csv", "tensorboard"],
        )

        # Initialize the SAC model with the logger and configuration parameters
        self.model = SAC(
            "MultiInputPolicy",
            self,
            verbose=1,
            tensorboard_log=self.logger.get_dir(),
            learning_rate=config["actor_learning_rate"],
            gamma=config["gamma"],
            tau=config["tau"],
            train_freq=(config["train_every_n_steps"], "step"),
            gradient_steps=config["train_iterations"],
            policy_kwargs={
                "net_arch": {
                    "pi": config["actor_hidden_layers"],
                    "qf": config["critic_hidden_layers"],
                },
            },
        )
        self.model._logger = self.logger
        self.model._custom_logger = True

    def _get_observation_space(self):
        return gym.spaces.Dict(
            {
                "num_agents": gym.spaces.Box(
                    low=0, high=self.max_agents, shape=(1,), dtype=int
                ),
                "num_passengers": gym.spaces.Box(
                    low=0, high=self.max_passengers, shape=(1,), dtype=int
                ),
                "agent_positions": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.max_agents, 2), dtype=float
                ),
                "passenger_positions": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_passengers, 2),
                    dtype=float,
                ),
                "passenger_destinations": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_passengers, 2),
                    dtype=float,
                ),
            }
        )

    def _get_action_space(self):
        num_agents = self.max_agents
        return gym.spaces.Box(low=-1, high=1, shape=(num_agents, 2), dtype=float)

    def reset(self) -> GymObs:
        return self.env.reset()

    def step(self, action: np.ndarray) -> GymStepReturn:
        return self.env.step(action)

    def train(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        self.model.train(0, [states, actions, rewards, next_states, dones])

    def get_action(self, observation):
        return self.model.predict(observation, deterministic=True)[0]

    def get_policy(self):
        return self.model.policy
