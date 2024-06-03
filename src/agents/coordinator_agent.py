# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>
import gymnasium as gym
import numpy as np
import tensorflow as tf
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.type_aliases import GymObs


class AICoordinator(gym.Env):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.max_agents = config["max_taxis"]
        self.max_passengers = config["max_passengers"]
        self.step_counter = 0
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        # Configure the logger
        self.logger = configure(
            folder="src/training/logger",
            format_strings=["log", "csv", "tensorboard"],
        )

        # Initialize the SAC model with the logger and configuration parameters
        self.model = SAC(
            "MultiInputPolicy",
            env,
            verbose=0,
            batch_size=config["batch_size"],
            # tensorboard_log=self.logger.get_dir(),
            learning_rate=config["actor_learning_rate"],
            # gamma=config["gamma"],
            # tau=config["tau"],
            train_freq=(config["train_every_n_steps"], "step"),
            gradient_steps=config["gradient_steps"],
            policy_kwargs={
                "net_arch": {
                    "pi": config["actor_hidden_layers"],
                    "qf": config["critic_hidden_layers"],
                },
            },
        )

    def _get_observation_space(self):
        return self.env.observation_space

    def _get_action_space(self):
        return self.env.action_space

    def _on_step(self, locals, globals):
        # Calculate percentage of passengers picked up and delivered
        self.step_counter += 1
        info = locals["infos"][0]  # Assuming single environment
        num_passengers = info["num_passengers"] + 1
        passengers_delivered = info["passengers_delivered"]
        ratio = passengers_delivered / num_passengers

        # Log relevant data
        self.logger.record(
            "train/reward", locals["rewards"][0]
        )  # Assuming single environment
        self.logger.record("train/Percent_trips_completed", ratio)
        self.logger.dump(step=self.step_counter)
        return True

    def reset(self) -> GymObs:
        return self.env.reset()

    def step(self, time_interval):
        # Log relevant data

        next_observation, actions, rewards, done, info = self.env.step(time_interval)
        return next_observation, actions, rewards, done, info

    def train(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        train_metrics = self.model.train(
            0, [states, actions, rewards, next_states, dones]
        )
        self.logger.record("train/loss_actor", train_metrics[0])
        self.logger.record("train/loss_critic", train_metrics[1])
        self.logger.record("train/entropy_loss", train_metrics[2])
        self.logger.record("train/entropy_coef", train_metrics[3])

    def get_action(self, observation):
        return self.model.predict(observation, deterministic=True)[0]

    def inference(self, observation):
        """Inference for use with python API

        Args:
            observation (dict): dictionary following format of observation space
        Returns: actions (np.ndarray): array of actions for each agent
        """
        # preprocess dictionary to fit format

        # Get actions. Output is numpy array of size (n_taxis,2)
        actions = self.model.predict(observation, deterministic=True)[0]

        # Initialize the destination array
        destination = np.zeros((10, 2))

        # Convert normalized action value to longitude and latitude
        for i, action in enumerate(actions):
            # Find the closest node based on the continuous action value
            closest_node_id = None
            min_distance = float("inf")
            for normalized_action, node_id in self.env.action_to_node_mapping.items():
                distance = np.sqrt(
                    (action[0] - normalized_action[0]) ** 2
                    + (action[1] - normalized_action[1]) ** 2
                )
                if distance < min_distance:
                    min_distance = distance
                    closest_node_id = node_id

            # Update the destination array with longitude and latitude
            destination[i, 0] = self.env.map_network.nodes[closest_node_id]["x"]
            destination[i, 1] = self.env.map_network.nodes[closest_node_id]["y"]

        return destination

    def get_policy(self):
        return self.model.policy
