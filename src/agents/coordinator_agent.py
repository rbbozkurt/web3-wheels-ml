# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Harshil Dave <harshil128@gmail.com>
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tf_agents.agents import DdpgAgent
from tf_agents.networks import Sequential
from tf_agents.specs import BoundedTensorSpec, TensorSpec
from tf_agents.trajectories import trajectory


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
        num_actions = self.env.action_space.n
        num_agents = len(self.env.taxi_agents)

        return BoundedTensorSpec(
            shape=(num_agents,), dtype=tf.int32, minimum=0, maximum=num_actions - 1
        )

    def _build_actor_network(self):
        # Define the actor network architecture
        num_agents = self.env.action_space.n

        # Create input layers for each observation component
        num_agents_input = tf.keras.layers.Input(shape=(), name="num_agents")
        num_passengers_input = tf.keras.layers.Input(shape=(), name="num_passengers")
        agent_positions_input = tf.keras.layers.Input(
            shape=(2,), name="agent_positions"
        )
        passenger_positions_input = tf.keras.layers.Input(
            shape=(2,), name="passenger_positions"
        )
        passenger_destinations_input = tf.keras.layers.Input(
            shape=(2,), name="passenger_destinations"
        )

        # Concatenate the input layers
        concatenated = tf.keras.layers.concatenate(
            [
                tf.keras.layers.Reshape((-1,))(num_agents_input),
                tf.keras.layers.Reshape((-1,))(num_passengers_input),
                tf.keras.layers.Reshape((-1,))(agent_positions_input),
                tf.keras.layers.Reshape((-1,))(passenger_positions_input),
                tf.keras.layers.Reshape((-1,))(passenger_destinations_input),
            ]
        )

        # Convert the concatenated shape to TensorShape
        concatenated_shape = tf.TensorShape(concatenated.shape)

        # Build the hidden layers
        x = Dense(128, input_shape=concatenated_shape, activation="relu")(concatenated)
        x = Dense(64, activation="relu")(x)
        outputs = Dense(num_agents, activation="softmax")(x)

        # Create the actor model
        actor_model = tf.keras.Model(
            inputs={
                "num_agents": num_agents_input,
                "num_passengers": num_passengers_input,
                "agent_positions": agent_positions_input,
                "passenger_positions": passenger_positions_input,
                "passenger_destinations": passenger_destinations_input,
            },
            outputs=outputs,
        )

        return actor_model

    def _build_critic_network(self):
        # Define the critic network architecture
        return Sequential(
            [Dense(128, activation="relu"), Dense(64, activation="relu"), Dense(1)]
        )

    def train(self, experiences):
        # Convert experiences to trajectories
        trajectories = [
            trajectory.from_transition(
                observation=tf.constant(exp[0], dtype=tf.float32),
                action=tf.constant(exp[1], dtype=tf.float32),
                reward=tf.constant(exp[2], dtype=tf.float32),
                discount=tf.constant(1.0 - float(exp[4]), dtype=tf.float32),
                next_step_type=tf.constant(0 if exp[4] else 1, dtype=tf.int32),
                observation_step=tf.constant(exp[3], dtype=tf.float32),
            )
            for exp in experiences
        ]

        # Create a dataset from the trajectories
        dataset = trajectory.experience_to_transitions(trajectories)

        # Train the agent with the dataset
        self.agent.train(dataset)

    def get_action(self, observation):
        # Convert the observation dictionary to a nested tensor
        observation = {
            key: tf.expand_dims(tf.convert_to_tensor(value), axis=0)
            for key, value in observation.items()
        }

        # Get the action from the agent based on the observation
        action_step = self.agent.policy.action(observation)
        return action_step.action.numpy()[0]

    def get_policy(self):
        # Get the current policy of the agent
        return self.agent.policy
