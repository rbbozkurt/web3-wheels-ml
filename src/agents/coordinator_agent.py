# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tf_agents.agents import DdpgAgent, TFAgent
from tf_agents.networks import Sequential
from tf_agents.specs import TensorSpec
from tf_agents.trajectories import Trajectory


class AICoordinator:
    def __init__(self, env):
        self.env = env

        # Define the observation and action specs
        self.observation_spec = self._get_observation_spec()
        self.action_spec = self._get_action_spec()

        # Define the actor and critic networks
        self.actor_net = self._build_actor_network()
        self.critic_net = self._build_critic_network()

        # Create the DDPG agent
        self.agent = DdpgAgent(
            self.observation_spec,
            self.action_spec,
            actor_network=self.actor_net,
            critic_network=self.critic_net,
            train_step_counter=tf.Variable(0),
        )

        # Initialize the agent
        self.agent.initialize()

    def _get_observation_spec(self):
        # Define the observation spec based on the environment's observation space
        observation_spec = {
            "num_agents": TensorSpec(shape=(), dtype=tf.int32),
            "num_passengers": TensorSpec(shape=(), dtype=tf.int32),
            "agent_positions": TensorSpec(shape=(None, 2), dtype=tf.float32),
            "passenger_positions": TensorSpec(shape=(None, 2), dtype=tf.float32),
            "passenger_destinations": TensorSpec(shape=(None, 2), dtype=tf.float32),
            # Add other relevant observation specs
        }
        return observation_spec

    def _get_action_spec(self):
        # Define the action spec based on the environment
        return TensorSpec(shape=self.env.action_space.shape, dtype=tf.float32)

    def _build_actor_network(self):
        # Define the actor network architecture
        return Sequential(
            [
                Dense(128, activation="relu"),
                Dense(64, activation="relu"),
                Dense(self.action_spec.shape[0], activation="tanh"),
            ]
        )

    def _build_critic_network(self):
        # Define the critic network architecture
        return Sequential(
            [Dense(128, activation="relu"), Dense(64, activation="relu"), Dense(1)]
        )

    def train(self, experience):
        # Train the agent with the collected experience
        self.agent.train(experience)

    def get_action(self, observation):
        # Get the action from the agent based on the observation
        observation = tf.expand_dims(observation, axis=0)
        action_step = self.agent.policy.action(observation)
        return action_step.action.numpy()[0]

    def get_policy(self):
        # Get the current policy of the agent
        return self.agent.policy
